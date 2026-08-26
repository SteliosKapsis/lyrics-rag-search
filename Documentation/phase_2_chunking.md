# Phase 2 — Chunking

## Overview

Splits cleaned lyrics into semantically coherent chunks suitable for embedding. Uses a hybrid strategy that respects song structure (verses, choruses, bridges) while keeping chunk sizes within a configurable range.

## Script

`pipeline/chunking.py`

## Usage

```bash
# From project root, with default parameters:
.venv\Scripts\python pipeline\chunking.py

# With custom parameters (useful for Phase 6 experimentation):
.venv\Scripts\python pipeline\chunking.py --max-chunk-size 600 --min-chunk-size 100 --overlap-lines 2
```

## Input

- **`data/raw/cleaned_lyrics.json`** — output from Phase 1.5 (cleaning)

## Output

| File | Location | Description |
|------|----------|-------------|
| `chunks.jsonl` | `data/processed/` | One JSON object per line, one chunk per line |

### Output schema (each line)

```json
{
  "text": "[Verse 1]\nThere ain't no gold in this river...",
  "title": "Easy On Me",
  "artist": "Adele",
  "release_date": "2021-10-15",
  "album": "30 (Target Exclusive)",
  "chunk_index": 0,
  "total_chunks": 6
}
```

## Dependencies

No additional dependencies beyond Python standard library.

## Chunking Strategy

### Why not fixed-size windows?

Lyrics have natural structural boundaries — verses, choruses, bridges — marked by section headers. Fixed-size token windows would split mid-verse, destroying the semantic coherence that makes lyrics searchable. A query like "sad song about leaving home" should match a complete verse about that theme, not a fragment.

### Why not pure section-based splitting?

Section sizes vary wildly. A chorus might be 4 lines (~80 chars), while a rap verse can be 40+ lines (~1000 chars). Pure section-based splitting produces chunks too inconsistent for good embedding quality.

### Hybrid approach (implemented)

The script uses a 6-step hybrid strategy:

1. **Split on section headers** — `[Verse 1]`, `[Chorus]`, `[Bridge]`, etc. are used as primary boundaries. Each section becomes a candidate chunk.

2. **Fallback for songs without headers** — if no `[Section]` markers exist, the lyrics are split on blank lines (stanza boundaries). This handles songs where Genius didn't annotate sections.

3. **Split oversized sections** — sections exceeding `max_chunk_size` (default 400 chars) are broken further, trying the coarsest boundary first: stanza (blank-line) breaks. If a stanza-level split would still leave a piece larger than `max_chunk_size` — or the section has no blank lines at all (e.g. a dense rap verse with no gap between bars) — it falls back to splitting on individual line boundaries instead. Only a section with no newlines anywhere (nothing left to split on without breaking mid-line) is kept intact as a single, still-oversized chunk; confirmed rare in practice (~30 sections in the current corpus). This two-tier fallback (stanza → line) is the same hierarchy-of-separators idea as a recursive character text splitter, applied to lyrics' own structural units instead of generic punctuation.

4. **Merge undersized sections** — adjacent sections smaller than `min_chunk_size` (default 80 chars) are merged together, but *never* if doing so would push the result past `max_chunk_size` — a section that's too small to merge safely is left standalone instead. Headers are concatenated (e.g., `[Bridge] + [Chorus]`). The last section gets special handling — if it's too small, it's merged back into the previous chunk, subject to the same size guard. (This guard was added after discovering it was previously possible for several small stanzas to cascade-merge into a single chunk many times larger than `max_chunk_size` — see Design Decisions.)

5. **Prepend section headers** — the section header (e.g., `[Verse 1]`) is prepended to the chunk text. This gives the embedding model structural context.

6. **Optional overlap** — if `--overlap-lines N` is set, the last N lines of each chunk are prepended to the next chunk. Default is 0 (no overlap). This parameter is exposed for Phase 6 experimentation.

### Assumptions about lyric structure

- Section headers appear on their own line in square brackets: `[Section Name]`
- Text between headers belongs to that section
- Blank lines within a section represent stanza boundaries
- Text before the first header (if any) is a preamble section (e.g., producer tags)

## Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-chunk-size` | 400 | Maximum chunk size in characters. Sections larger than this are split on stanza boundaries. |
| `--min-chunk-size` | 80 | Minimum chunk size in characters. Sections smaller than this are merged with adjacent sections. |
| `--overlap-lines` | 0 | Number of lines from the previous chunk prepended to the next as context overlap. |

These parameters are deliberately exposed as CLI arguments so Phase 6 (evaluation notebook) can sweep across different combinations to find the optimal configuration.

## Test Results (5-song sample)

Regenerated after the stanza→line fallback fix (see Design Decisions):

```
5 songs → 46 chunks
Average chunk size: 237 chars
Min: 73 chars, Max: 438 chars

Chunks per song:
  Easy On Me:     6 chunks
  STAY:           8 chunks
  INDUSTRY BABY: 13 chunks
  Fancy Like:    10 chunks
  Bad Habits:     9 chunks

Size distribution:
  <100 chars:    4 chunks
  100-200:      19 chunks
  200-400:      19 chunks
  400-600:       4 chunks
  600+:          0 chunks
```

Previously, INDUSTRY BABY's Jack Harlow verse (no internal blank lines) produced a single 1026-char chunk, kept intact because the splitter had no fallback below stanza-level. It now falls back to line-level splitting for that verse instead, and the corpus-wide max chunk size dropped accordingly (see Design Decisions) — the "keep dense, blank-line-free verses intact" behavior described in older versions of this doc no longer applies except for the small number of sections with no newlines at all.

## Design Decisions

- **Stanza→line fallback splitting (fixed, previously a silent gap)** — the original splitter tried only one fallback level (blank-line stanza boundaries) and gave up if that didn't yield multiple pieces, letting an oversized section through untouched. Checked against the full corpus: this let 17.5% of all chunks exceed `max_chunk_size`, with a worst case of 255,760 characters (later traced separately to non-song contamination in the source data, not this bug alone — see `phase_1.5_cleaning.md`). The fix adds a second-tier fallback to individual line boundaries, and also closes a related gap in the merge step (`merge_small_sections`) that could previously cascade-merge several undersized sections into something far larger than `max_chunk_size`, since it never checked the merged result's size. After both fixes, chunks exceeding `max_chunk_size` dropped to 9.6% of the corpus, and 99.7% of those are within 100 characters of the limit — an expected consequence of greedy line-packing, not a bug. Only sections with literally no newlines anywhere still pass through oversized (~30 in the current corpus).
- **Characters, not tokens** — chunk sizes are measured in characters rather than tokens for simplicity and speed. The embedding model's tokenizer will handle the conversion. At ~4 chars/token, 400 chars ≈ 100 tokens, well within model limits.
- **Section headers included in chunk text** — headers like `[Verse 1]` are prepended to the chunk. This gives the embedding model structural context (the model can learn that "chorus" text is repeated/central, "bridge" text is transitional, etc.).
- **Merged headers use `+` notation** — when small sections are merged, their headers are combined: `[Bridge] + [Chorus]`. This preserves provenance.
- **Oversized split sections marked `(cont.)`** — when a section is split, subsequent chunks get `(cont.)` appended to the header.
- **JSONL format** — one chunk per line makes it easy to stream, count lines, and process in parallel. Unlike JSON arrays, JSONL doesn't require loading the entire file into memory.
