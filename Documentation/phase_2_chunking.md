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

3. **Split oversized sections** — sections exceeding `max_chunk_size` (default 400 chars) are broken further on stanza boundaries (blank lines within the section). If a section has no internal blank lines (e.g., a dense rap verse), it's kept intact rather than splitting mid-line.

4. **Merge undersized sections** — adjacent sections smaller than `min_chunk_size` (default 80 chars) are merged together. Headers are concatenated (e.g., `[Bridge] + [Chorus]`). The last section gets special handling — if it's too small, it's merged back into the previous chunk.

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

```
5 songs → 37 chunks
Average chunk size: 289 chars
Min: 99 chars, Max: 1026 chars

Chunks per song:
  Easy On Me:     6 chunks
  STAY:           7 chunks
  INDUSTRY BABY:  8 chunks
  Fancy Like:     8 chunks
  Bad Habits:     8 chunks

Size distribution:
  <100 chars:   2 chunks
  100-200:     14 chunks
  200-400:     10 chunks
  400-600:     10 chunks
  600+:         1 chunk
```

The single 1026-char chunk is a dense rap verse (Jack Harlow's verse in INDUSTRY BABY) with no internal stanza breaks. The splitter correctly keeps it intact rather than splitting mid-line.

## Design Decisions

- **Characters, not tokens** — chunk sizes are measured in characters rather than tokens for simplicity and speed. The embedding model's tokenizer will handle the conversion. At ~4 chars/token, 400 chars ≈ 100 tokens, well within model limits.
- **Section headers included in chunk text** — headers like `[Verse 1]` are prepended to the chunk. This gives the embedding model structural context (the model can learn that "chorus" text is repeated/central, "bridge" text is transitional, etc.).
- **Merged headers use `+` notation** — when small sections are merged, their headers are combined: `[Bridge] + [Chorus]`. This preserves provenance.
- **Oversized split sections marked `(cont.)`** — when a section is split, subsequent chunks get `(cont.)` appended to the header.
- **JSONL format** — one chunk per line makes it easy to stream, count lines, and process in parallel. Unlike JSON arrays, JSONL doesn't require loading the entire file into memory.
