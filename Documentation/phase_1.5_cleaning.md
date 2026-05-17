# Phase 1.5 — Data Cleaning & Validation

## Overview

Cleans Genius scraping artifacts from raw lyrics, normalizes section headers, detects non-English entries, and validates data quality. Produces a cleaned dataset ready for chunking (Phase 2) and a validation report for manual review.

## Script

`collection/clean_lyrics.py`

## Usage

```bash
# From project root, with default paths:
.venv\Scripts\python collection\clean_lyrics.py

# Or with custom paths:
.venv\Scripts\python collection\clean_lyrics.py --input data/raw/lyrics.json --output data/raw/cleaned_lyrics.json --report data/failed/validation_report.csv
```

## Input

- **`data/raw/lyrics.json`** — raw output from Phase 1 (ingestion)

## Output

| File | Location | Description |
|------|----------|-------------|
| `cleaned_lyrics.json` | `data/raw/` | Cleaned lyrics, same schema as input. Non-English entries excluded. |
| `validation_report.csv` | `data/failed/` | Flagged entries with reasons for review |

### Validation report schema

```csv
title,artist,genius_id,reasons
"Some Song","Some Artist",12345,"non_english (detected=es)"
"Another","Artist",67890,"short_lyrics (43 chars)"
```

## Dependencies

- `rapidfuzz>=3.0.0` — fuzzy string matching for artist/title mismatch detection
- `langdetect>=1.0.9` — language detection to filter non-English lyrics

## Cleaning Steps (in order)

### 1. Remove "SongTitle Lyrics" header

Genius prepends the song title + "Lyrics" as the first line of scraped text. Detected via regex and removed.

### 2. Remove contributor/translation text

Genius embeds contributor counts at the start of lyrics (e.g., `"64 ContributorsTranslationsFrançais..."`). Matched by `^\d*\s*Contributor.*$` and removed.

### 3. Remove "Embed" artifacts

Scraped pages often end with `"123Embed"` or `"EmbedShare..."` text. Matched by `(?:\d+)?Embed\b.*$` and removed.

### 4. Remove "You might also like"

Genius injects ad text `"You might also like"` between song sections. Matched exactly and removed.

### 5. Normalize section headers

Section headers come in inconsistent formats from Genius:

```
[verse 1], [VERSE 1], [Verse 1: artist name], [CHORUS]
```

Normalized to consistent title case:

```
[Verse 1], [Chorus], [Verse 1: Artist Name], [Bridge]
```

Headers are **preserved** (not stripped) because Phase 2 uses them as structural boundaries for chunking.

### 6. Collapse blank lines

Multiple consecutive blank lines (3+) are collapsed to a single blank line.

### 7. Strip whitespace

Leading and trailing whitespace removed.

## Validation Checks

### Empty/short lyrics

Entries with empty lyrics or fewer than 100 characters are flagged as `empty_lyrics` or `short_lyrics`. These remain in the cleaned output for manual review.

### Language detection

Uses `langdetect` to identify the language of each entry. Non-English entries are:
- **Flagged** in `validation_report.csv` with reason `non_english (detected=xx)`
- **Excluded** from `cleaned_lyrics.json` entirely

Language detection preprocessing:
- Section headers are stripped before detection (they're always in English and would bias the result)
- Lines shorter than 10 characters are excluded (ad-libs like "Ayy", "Oh" aren't meaningful for detection)

**Rationale for excluding non-English:** The embedding model (`all-MiniLM-L6-v2`) is primarily trained on English text. Non-English lyrics produce lower-quality embeddings and would degrade retrieval accuracy. Billboard data is overwhelmingly English, so this is a defensive filter.

### Artist/title mismatch

If the Genius-returned artist/title doesn't closely match the original input (fuzzy score < 60 via `rapidfuzz.fuzz.token_sort_ratio`), the entry is flagged. These entries remain in the output — the flag is informational for manual review.

**Note:** This check currently compares Genius-returned fields against themselves (since the original CSV input is not carried through). Its main value is catching obviously broken entries. A future improvement would be to pass original CSV values through the pipeline for comparison.

## Design Decisions

- **Flagged entries stay in output (except non-English)** — the validation report is for human review. Short lyrics or slight mismatches may still be valid and useful for the RAG pipeline.
- **Non-English entries are hard-excluded** — no point embedding them with an English model.
- **Cleaning happens before validation** — artifacts are removed first so that length checks and language detection operate on clean text.

## Assumptions About Genius Scraping Artifacts

As required by the Phase 1.5 prompt, here are the assumptions made:

1. Genius always prepends `"SongTitle Lyrics"` as the first line
2. Contributor counts appear as `"\d+ Contributor..."` at the start
3. `"Embed"` or `"\d+Embed"` appears at the very end of scraped text
4. `"You might also like"` is injected between sections as advertising
5. Section headers always use square bracket notation `[...]`
6. Section headers may contain artist attribution after a colon (`[Verse 1: Artist]`)

These patterns were observed in `lyricsgenius` v3.11.0 output. If the library or Genius's page structure changes, these patterns may need updating.
