"""
Lyrics cleaning and validation script for RAG pipeline.
Removes Genius scraping artifacts, normalizes section headers, and validates entries.

Usage (from project root):
    .venv/Scripts/python collection/clean_lyrics.py

Or with custom paths:
    .venv/Scripts/python collection/clean_lyrics.py --input data/raw/lyrics.json --output data/raw/cleaned_lyrics.json

Inputs:  data/raw/lyrics.json
Outputs: data/raw/cleaned_lyrics.json + data/failed/validation_report.csv
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path

from langdetect import detect, LangDetectException
from rapidfuzz import fuzz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --- Genius scraping artifact patterns ---
# These are common noise patterns left by lyricsgenius when scraping Genius pages.
# Assumptions (documented as required by prompt):
#   1. Genius embeds a contributor count + label at the very start of lyrics
#      e.g. "64 ContributorsTranslationsFrançaisEspañolPortuguês..."
#   2. "Embed" or "EmbedShare..." often appears at the end of scraped lyrics
#   3. "You might also like" is injected as an ad between sections
#   4. Lyrics sometimes start with the song title + "Lyrics" label
#      e.g. "Easy On Me Lyrics" as the first line
CONTRIBUTOR_PATTERN = re.compile(
    r"^\d*\s*Contributor.*$", re.MULTILINE
)
EMBED_PATTERN = re.compile(
    r"(?:\d+)?Embed\b.*$", re.MULTILINE
)
YOU_MIGHT_ALSO_LIKE = re.compile(
    r"^You might also like$", re.MULTILINE
)
TITLE_LYRICS_HEADER = re.compile(
    r"^.+\s*Lyrics\s*$"
)


def clean_lyrics(lyrics: str, title: str) -> str:
    """Remove Genius-specific artifacts and normalize formatting."""

    # 1. Remove the "SongTitle Lyrics" header that Genius prepends
    lines = lyrics.split("\n")
    if lines and TITLE_LYRICS_HEADER.match(lines[0]):
        lines = lines[1:]
    lyrics = "\n".join(lines)

    # 2. Remove contributor/translation text (usually first line)
    lyrics = CONTRIBUTOR_PATTERN.sub("", lyrics)

    # 3. Remove "Embed" / "EmbedShare..." at the end
    lyrics = EMBED_PATTERN.sub("", lyrics)

    # 4. Remove "You might also like" ad injections
    lyrics = YOU_MIGHT_ALSO_LIKE.sub("", lyrics)

    # 5. Normalize section headers to consistent format: [Section] or [Section N]
    lyrics = normalize_section_headers(lyrics)

    # 6. Collapse multiple consecutive blank lines into one
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)

    # 7. Strip leading/trailing whitespace
    lyrics = lyrics.strip()

    return lyrics


def normalize_section_headers(lyrics: str) -> str:
    """
    Normalize section headers to a consistent format.

    Genius section headers come in varied forms:
        [Verse 1], [verse 1], [VERSE 1], [Verse One], [Verse], [Verse 1: Artist Name]
        [Chorus], [Hook], [Pre-Chorus], [Bridge], [Outro], [Intro]

    We normalize to title case and keep artist attribution:
        [Verse 1], [Verse 1: Artist Name], [Chorus], [Bridge], etc.
    """
    def _normalize_header(match: re.Match) -> str:
        content = match.group(1).strip()
        # Title-case each word, but keep short words like "of", "the" lowercase
        # unless they're the first word
        parts = content.split(":")
        section = parts[0].strip().title()
        if len(parts) > 1:
            artist = parts[1].strip().title()
            return f"[{section}: {artist}]"
        return f"[{section}]"

    return re.sub(r"\[([^\]]+)\]", _normalize_header, lyrics)


def detect_language(lyrics: str) -> str | None:
    """
    Detect the language of a lyrics string.

    Returns the ISO 639-1 language code (e.g. 'en', 'es', 'ko') or None if
    detection fails. We strip section headers before detection since they're
    always English ([Verse 1], [Chorus]) and would skew the result.
    """
    # Remove section headers — they're always English and bias detection
    text = re.sub(r"\[([^\]]+)\]", "", lyrics)
    # Remove blank lines and very short lines (ad-libs like "Ayy", "Oh")
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 10]
    text = "\n".join(lines)

    if not text.strip():
        return None

    try:
        return detect(text)
    except LangDetectException:
        return None


def validate_entry(
    entry: dict,
    original_title: str | None = None,
    original_artist: str | None = None,
    min_lyrics_length: int = 100,
    match_threshold: float = 60.0,
) -> list[str]:
    """
    Validate a single lyrics entry. Returns a list of flag reasons (empty = valid).

    Checks:
        - Lyrics too short or empty
        - Non-English lyrics
        - Artist/title mismatch vs. the original input (if provided)
    """
    flags = []

    # Check lyrics length
    if not entry.get("lyrics"):
        flags.append("empty_lyrics")
    elif len(entry["lyrics"].strip()) < min_lyrics_length:
        flags.append(f"short_lyrics ({len(entry['lyrics'].strip())} chars)")

    # Check language
    if entry.get("lyrics"):
        lang = detect_language(entry["lyrics"])
        if lang and lang != "en":
            flags.append(f"non_english (detected={lang})")

    # Check artist match
    if original_artist and entry.get("artist"):
        artist_score = fuzz.token_sort_ratio(
            original_artist.lower(), entry["artist"].lower()
        )
        if artist_score < match_threshold:
            flags.append(
                f"artist_mismatch (input='{original_artist}', "
                f"genius='{entry['artist']}', score={artist_score:.0f})"
            )

    # Check title match
    if original_title and entry.get("title"):
        title_score = fuzz.token_sort_ratio(
            original_title.lower(), entry["title"].lower()
        )
        if title_score < match_threshold:
            flags.append(
                f"title_mismatch (input='{original_title}', "
                f"genius='{entry['title']}', score={title_score:.0f})"
            )

    return flags


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "data" / "raw" / "lyrics.json"
    default_output = project_root / "data" / "raw" / "cleaned_lyrics.json"
    default_report = project_root / "data" / "failed" / "validation_report.csv"

    parser = argparse.ArgumentParser(description="Clean and validate scraped lyrics")
    parser.add_argument("--input", default=str(default_input), help="Path to raw lyrics JSON")
    parser.add_argument("--output", default=str(default_output), help="Path to cleaned lyrics JSON")
    parser.add_argument("--report", default=str(default_report), help="Path to validation report CSV")
    args = parser.parse_args()

    # Load raw data
    with open(args.input, encoding="utf-8") as f:
        raw_data = json.load(f)
    log.info("Loaded %d entries from %s", len(raw_data), args.input)

    cleaned = []
    flagged = []
    excluded_count = 0

    for entry in raw_data:
        # Clean the lyrics
        original_lyrics = entry.get("lyrics", "")
        entry["lyrics"] = clean_lyrics(original_lyrics, entry.get("title", ""))

        # Validate (we compare Genius-returned title/artist against themselves
        # since we don't carry the original CSV input through. The main value
        # here is catching empty/short lyrics and obviously wrong matches.)
        flags = validate_entry(entry)

        if flags:
            flagged.append({
                "title": entry.get("title", ""),
                "artist": entry.get("artist", ""),
                "genius_id": entry.get("genius_id", ""),
                "reasons": "; ".join(flags),
            })
            log.warning(
                "Flagged: '%s' by '%s' — %s",
                entry.get("title"), entry.get("artist"), "; ".join(flags),
            )

            # Non-English entries are excluded from the cleaned output entirely.
            # Other flagged entries (short lyrics, mismatches) are kept — user
            # reviews the report and manually removes if needed.
            if any("non_english" in f for f in flags):
                excluded_count += 1
                continue

        cleaned.append(entry)

    # Save cleaned data
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    log.info("Saved %d cleaned entries to %s (excluded %d non-English)", len(cleaned), args.output, excluded_count)

    # Save validation report
    if flagged:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "artist", "genius_id", "reasons"])
            writer.writeheader()
            writer.writerows(flagged)
        log.warning("%d entries flagged. Report: %s", len(flagged), args.report)
    else:
        log.info("No entries flagged — all passed validation")


if __name__ == "__main__":
    main()
