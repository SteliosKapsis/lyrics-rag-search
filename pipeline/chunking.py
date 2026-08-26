"""
Lyrics chunking script for RAG pipeline.
Splits cleaned lyrics into semantically coherent chunks for embedding.

Usage (from project root):
    .venv/Scripts/python pipeline/chunking.py

Or with custom parameters:
    .venv/Scripts/python pipeline/chunking.py --max-chunk-size 400 --min-chunk-size 80 --overlap 50

Inputs:  data/raw/cleaned_lyrics.json
Outputs: data/processed/chunks.jsonl (one chunk per line)
"""

import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Regex to detect section headers like [Verse 1], [Chorus: Artist], etc.
SECTION_HEADER_RE = re.compile(r"^\[([^\]]+)\]\s*$")


def split_into_sections(lyrics: str) -> list[dict]:
    """
    Split lyrics into sections using [Header] markers.

    Returns a list of dicts: {"header": str|None, "text": str}
    Songs without any section headers return a single section with header=None.

    Assumptions about lyric structure:
        - Section headers appear on their own line in square brackets
        - Text between headers belongs to that section
        - Text before the first header (if any) is its own section with header=None
        - Blank lines within a section are stanza boundaries (preserved)
    """
    sections = []
    current_header = None
    current_lines = []

    for line in lyrics.split("\n"):
        match = SECTION_HEADER_RE.match(line.strip())
        if match:
            # Save previous section if it has content
            text = "\n".join(current_lines).strip()
            if text:
                sections.append({"header": current_header, "text": text})
            current_header = match.group(0).strip()  # Keep full [Header] string
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    text = "\n".join(current_lines).strip()
    if text:
        sections.append({"header": current_header, "text": text})

    return sections


def split_by_stanzas(text: str) -> list[str]:
    """
    Split text on blank lines (stanza boundaries).
    Used as fallback for songs without section headers, or to break up
    oversized sections.
    """
    stanzas = re.split(r"\n\s*\n", text)
    return [s.strip() for s in stanzas if s.strip()]


def split_by_lines(text: str) -> list[str]:
    """
    Split text into individual non-blank lines.
    Second-tier fallback for oversized sections with no blank-line
    stanza breaks at all (common in rap verses transcribed as one
    continuous block, with no empty line between bars) -- split_by_stanzas()
    alone can't break these up, since it only splits on blank lines.
    """
    return [line.strip() for line in text.split("\n") if line.strip()]


def _pack_pieces(pieces: list[str], sep: str, max_chunk_size: int) -> list[str]:
    """
    Greedily pack `pieces` (stanzas or lines) into strings joined by
    `sep`, starting a new piece whenever adding the next one would push
    the running total over max_chunk_size. Shared accumulation logic for
    both the stanza-level and line-level fallbacks in split_large_section().
    """
    packed = [pieces[0]]
    for piece in pieces[1:]:
        candidate = packed[-1] + sep + piece
        if len(candidate) > max_chunk_size and packed[-1]:
            packed.append(piece)
        else:
            packed[-1] = candidate
    return packed


def merge_small_sections(
    sections: list[dict], min_chunk_size: int, max_chunk_size: int
) -> list[dict]:
    """
    Merge adjacent sections that are below min_chunk_size (in characters).
    Preserves section headers by concatenating them.

    Never merges if the result would exceed max_chunk_size -- the small
    section is left standalone (under min_chunk_size) instead. Without
    this guard, a run of several small stanzas could cascade-merge into
    a chunk far larger than max_chunk_size, silently undoing the size
    cap split_large_section() had already enforced (confirmed in
    practice: 17.5% of chunks exceeded max_chunk_size before this check
    was added).
    """
    if not sections:
        return sections

    merged = [sections[0]]

    for section in sections[1:]:
        prev = merged[-1]
        merged_len = len(prev["text"]) + 2 + len(section["text"])  # +2 for the "\n\n" join
        if len(prev["text"]) < min_chunk_size and merged_len <= max_chunk_size:
            # Merge with previous
            header_parts = []
            if prev["header"]:
                header_parts.append(prev["header"])
            if section["header"]:
                header_parts.append(section["header"])
            merged[-1] = {
                "header": " + ".join(header_parts) if header_parts else None,
                "text": prev["text"] + "\n\n" + section["text"],
            }
        else:
            merged.append(section)

    # Check if the last section is too small and merge it back
    if len(merged) > 1 and len(merged[-1]["text"]) < min_chunk_size:
        prev, last = merged[-2], merged[-1]
        merged_len = len(prev["text"]) + 2 + len(last["text"])
        if merged_len <= max_chunk_size:
            merged.pop()
            header_parts = []
            if prev["header"]:
                header_parts.append(prev["header"])
            if last["header"]:
                header_parts.append(last["header"])
            merged[-1] = {
                "header": " + ".join(header_parts) if header_parts else None,
                "text": prev["text"] + "\n\n" + last["text"],
            }
        # else: leave the undersized last section standalone rather than
        # exceed max_chunk_size by merging it in.

    return merged


def split_large_section(section: dict, max_chunk_size: int) -> list[dict]:
    """
    Split a section that exceeds max_chunk_size into smaller chunks.

    Tries blank-line (stanza) boundaries first, to keep chunks as
    semantically coherent as possible. If a section has no blank-line
    breaks at all, falls back to splitting on individual line boundaries
    instead of returning the whole oversized section untouched -- this
    previously affected ~15,600 sections in this corpus (the large
    majority of chunks that ended up exceeding max_chunk_size in
    practice), almost all of them rap verses transcribed as one
    continuous block with no blank line between bars.

    Only a section with no newlines at all -- nothing left to split on
    without breaking mid-line -- is returned as a single, still-oversized
    chunk. True last resort; confirmed rare (~30 sections in this corpus).
    """
    if len(section["text"]) <= max_chunk_size:
        return [section]

    stanzas = split_by_stanzas(section["text"])
    # Stanza-level splitting only helps if every resulting stanza actually
    # fits -- a section can have multiple stanzas where just one of them
    # (a dense verse with no internal blank line) is still oversized on
    # its own; packing can't shrink an already-oversized single piece, so
    # falling back to the whole section's line-level split is what
    # actually guarantees the cap, not just declaring "multiple stanzas
    # found" and stopping there.
    if len(stanzas) > 1 and all(len(s) <= max_chunk_size for s in stanzas):
        pieces, sep = stanzas, "\n\n"
    else:
        pieces, sep = split_by_lines(section["text"]), "\n"

    if len(pieces) <= 1:
        return [section]

    packed = _pack_pieces(pieces, sep, max_chunk_size)

    chunks = []
    for i, text in enumerate(packed):
        header = section["header"]
        if header and i > 0:
            header = f"{section['header']} (cont.)"
        chunks.append({"header": header, "text": text.strip()})

    return chunks


def chunk_song(
    lyrics: str,
    max_chunk_size: int = 400,
    min_chunk_size: int = 80,
    overlap_lines: int = 0,
) -> list[str]:
    """
    Chunk a single song's lyrics using the hybrid strategy:

    1. Split on section headers ([Verse], [Chorus], etc.)
    2. For songs without headers, split on blank lines (stanza boundaries)
    3. Merge sections that are too small (< min_chunk_size chars)
    4. Split sections that are too large (> max_chunk_size chars) on stanza boundaries
    5. Optionally add overlap (last N lines of previous chunk prepended to next)

    Returns a list of chunk text strings.
    """
    # Step 1: Split into sections
    sections = split_into_sections(lyrics)

    # Step 2: If no section headers were found, the whole song is one section.
    # Split it by stanzas instead.
    if len(sections) == 1 and sections[0]["header"] is None:
        stanzas = split_by_stanzas(sections[0]["text"])
        sections = [{"header": None, "text": s} for s in stanzas]

    # Step 3: Split oversized sections
    expanded = []
    for section in sections:
        expanded.extend(split_large_section(section, max_chunk_size))
    sections = expanded

    # Step 4: Merge undersized sections (never past max_chunk_size -- see
    # merge_small_sections' docstring)
    sections = merge_small_sections(sections, min_chunk_size, max_chunk_size)

    # Step 5: Build final chunk texts (with header prepended if present)
    chunk_texts = []
    for section in sections:
        text = section["text"]
        if section["header"]:
            text = section["header"] + "\n" + text
        chunk_texts.append(text)

    # Step 6: Apply overlap if requested
    if overlap_lines > 0 and len(chunk_texts) > 1:
        overlapped = [chunk_texts[0]]
        for i in range(1, len(chunk_texts)):
            prev_lines = chunk_texts[i - 1].splitlines()
            overlap = prev_lines[-overlap_lines:]
            overlapped.append("\n".join(overlap) + "\n\n" + chunk_texts[i])
        chunk_texts = overlapped

    return chunk_texts


def process_songs(
    songs: list[dict],
    max_chunk_size: int,
    min_chunk_size: int,
    overlap_lines: int,
) -> list[dict]:
    """
    Process all songs and produce chunk objects with metadata.

    Each chunk object has:
        text, title, artist, release_date, album, chunk_index, total_chunks
    """
    all_chunks = []

    for song in songs:
        lyrics = song.get("lyrics", "")
        if not lyrics.strip():
            continue

        chunks = chunk_song(lyrics, max_chunk_size, min_chunk_size, overlap_lines)

        for i, text in enumerate(chunks):
            all_chunks.append({
                "text": text,
                "title": song.get("title", ""),
                "artist": song.get("artist", ""),
                "release_date": song.get("release_date"),
                "album": song.get("album"),
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    return all_chunks


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "data" / "raw" / "cleaned_lyrics.json"
    default_output = project_root / "data" / "processed" / "chunks.jsonl"

    parser = argparse.ArgumentParser(description="Chunk lyrics for embedding")
    parser.add_argument("--input", default=str(default_input), help="Path to cleaned lyrics JSON")
    parser.add_argument("--output", default=str(default_output), help="Path to output JSONL file")
    parser.add_argument("--max-chunk-size", type=int, default=400, help="Max chunk size in characters (default: 400)")
    parser.add_argument("--min-chunk-size", type=int, default=80, help="Min chunk size in characters (default: 80)")
    parser.add_argument("--overlap-lines", type=int, default=0, help="Number of lines from previous chunk to prepend as overlap (default: 0)")
    args = parser.parse_args()

    # Load cleaned data
    with open(args.input, encoding="utf-8") as f:
        songs = json.load(f)
    log.info("Loaded %d songs from %s", len(songs), args.input)
    log.info("Parameters: max_chunk_size=%d, min_chunk_size=%d, overlap_lines=%d",
             args.max_chunk_size, args.min_chunk_size, args.overlap_lines)

    # Chunk all songs
    chunks = process_songs(songs, args.max_chunk_size, args.min_chunk_size, args.overlap_lines)

    # Write JSONL (one chunk per line)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Summary stats
    sizes = [len(c["text"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    log.info(
        "Done. %d chunks from %d songs. Avg chunk size: %.0f chars. "
        "Min: %d, Max: %d. Output: %s",
        len(chunks), len(songs), avg_size, min(sizes, default=0),
        max(sizes, default=0), args.output,
    )


if __name__ == "__main__":
    main()
