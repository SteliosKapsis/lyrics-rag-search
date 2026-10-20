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

Section headers come in inconsistent formats from Genius. Each is normalized to consistent title case:

| Before | After |
|--------|-------|
| `[verse 1]` | `[Verse 1]` |
| `[VERSE 1]` | `[Verse 1]` |
| `[Verse 1: artist name]` | `[Verse 1: Artist Name]` |
| `[CHORUS]` | `[Chorus]` |

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
- Only lines longer than 10 characters are considered — lines of 10 characters or fewer are excluded (ad-libs like "Ayy", "Oh" aren't meaningful for detection)

**Rationale for excluding non-English:** The embedding model (`all-MiniLM-L6-v2`) is primarily trained on English text. Non-English lyrics produce lower-quality embeddings and would degrade retrieval accuracy. Billboard data is overwhelmingly English, so this is a defensive filter.

## Post-Cleaning Data Quality Pass (`scripts/`)

`validate_entry()` originally had an `artist_mismatch`/`title_mismatch` check (fuzzy match against the original CSV title/artist via `rapidfuzz.fuzz.token_sort_ratio`), but it was dead code — `main()` never threaded the original CSV values through to the call, so the check could never fire. It has since been removed rather than wired up, in favor of the two targeted scripts below, run once against `data/raw/cleaned_lyrics.json` after Phase 1.5 and before Phase 2 (chunking), added after auditing the corpus and finding two issues the dead check would likely have caught, had it been reachable:

### `scripts/dedupe_cleaned_lyrics.py`

`cleaned_lyrics.json` had 19,320 entries but only 14,997 unique `(title, artist)` pairs — 4,323 songs present more than once, mostly Billboard chart entries that charted across multiple weeks/years and were kept every time they were re-encountered during ingestion. `pipeline/chunking.py` doesn't deduplicate, so every duplicate was chunked and indexed again, giving duplicated songs a disproportionate share of retrieval candidate slots relative to non-duplicated songs. Confirmed to affect 14 of the 75 golden-set answerable queries directly.

Policy: keep the first occurrence per `(title, artist)` in file order, drop the rest. Backs up the pre-dedup file to `cleaned_lyrics.json.bak` before writing.

### `scripts/filter_non_song_entries.py`

Genius hosts far more than song lyrics — literary excerpts, movie/TV scripts, Supreme Court opinions, political speeches, "Listening Log" blog posts. The ingestion step's fuzzy search occasionally matched a Billboard title to one of these completely unrelated pages instead of the actual song (e.g. a Billboard entry ending up matched to a Marcel Proust chapter, or a McKinley State of the Union address). The lyrics-length distribution makes this easy to isolate: 95% of entries sit under ~4,500 characters, then a long tail runs from ~8,000 up to 825,000 characters that is, on inspection, entirely non-song content. There's also a handful of degenerate 1-11 character entries from deleted/removed Genius pages (`artist: "Deleted Artist"`).

Policy: drop entries with `len(lyrics) > 8000` or `< 15` characters, or whose artist is a known deleted-page placeholder — 238 entries in total (230 too-long, 7 junk-artist, 1 too-short), verified against real short songs (e.g. "PPAP" at 202 chars) to confirm no legitimate song is anywhere near the cutoffs. Backs up to `cleaned_lyrics.json.prefilter_bak` and writes a full audit trail of every dropped entry to `scripts/non_song_entries_removed.json`.

Run order: `dedupe_cleaned_lyrics.py` then `filter_non_song_entries.py`, both before `pipeline/chunking.py`. Final corpus: **14,759 songs** (down from the 14,997 previously documented throughout this project — that figure included both the duplicates and the non-song contamination).

## Design Decisions

- **Flagged entries stay in output (except non-English)** — the validation report is for human review. Short lyrics or slight mismatches may still be valid and useful for the RAG pipeline.
- **Non-English entries are hard-excluded** — no point embedding them with an English model.
- **Cleaning happens before validation** — artifacts are removed first so that length checks and language detection operate on clean text.

## Assumptions About Genius Scraping Artifacts

As required by the Phase 1.5 prompt, here are the assumptions made:

1. Genius always prepends `"SongTitle Lyrics"` as the first line
2. Contributor counts appear as `"\d* Contributor..."` at the start
3. `"Embed"` or `"\d+Embed"` appears at the very end of scraped text
4. `"You might also like"` is injected between sections as advertising
5. Section headers always use square bracket notation `[...]`
6. Section headers may contain artist attribution after a colon (`[Verse 1: Artist]`)

These patterns were observed in `lyricsgenius` v3.11.0 output. If the library or Genius's page structure changes, these patterns may need updating.
