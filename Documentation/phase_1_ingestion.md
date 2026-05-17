# Phase 1 — Data Ingestion

## Overview

Fetches lyrics and metadata from the Genius API for all unique songs in the Billboard charts dataset. This is the first stage of the RAG pipeline — raw data collection.

## Script

`collection/ingest_lyrics.py`

## Usage

```bash
# From project root, with .env configured:
.venv\Scripts\python collection\ingest_lyrics.py --input charts.csv

# Or with explicit token:
.venv\Scripts\python collection\ingest_lyrics.py --input charts.csv --token YOUR_TOKEN

# Adjust delay between requests (default 1.5s):
.venv\Scripts\python collection\ingest_lyrics.py --input charts.csv --delay 1.0
```

## Input

- **`charts.csv`** — Billboard Hot 100 chart data with columns: `date`, `rank`, `song`, `artist`, `last-week`, `peak-rank`, `weeks-on-board`
- The script also accepts CSVs with `title` instead of `song` as the column name
- **29,680 unique songs** in the full dataset

## Output

| File | Location | Description |
|------|----------|-------------|
| `lyrics.json` | `data/raw/` | JSON array of song objects |
| `failed_fetches.csv` | `data/failed/` | Songs that couldn't be fetched after all retries |

### Output schema (`lyrics.json`)

```json
{
  "title": "Easy On Me",
  "artist": "Adele",
  "release_date": "2021-10-15",
  "album": "30 (Target Exclusive)",
  "lyrics": "[Verse 1]\nThere ain't no gold in this river...",
  "genius_id": 7260084
}
```

## Dependencies

- `lyricsgenius>=3.0.1` — Python client for Genius API
- `python-dotenv>=1.0.0` — loads API token from `.env`

## Key Design Decisions

### Primary artist extraction

Billboard lists featured artists in the `artist` column (e.g., `"The Kid LAROI & Justin Bieber"`). Searching Genius with the full string often returns translations or wrong matches. The `_primary_artist()` function extracts only the primary artist before `&`, `Featuring`, `Feat.`, etc.

```
"The Kid LAROI & Justin Bieber" → "The Kid LAROI"
"Drake Featuring Future"        → "Drake"
```

### Translation filtering

Genius's search API sometimes returns translated versions of songs (German, Turkish, etc.) instead of the original English. The script sets `excluded_terms` to filter these out by common translation labels across 8 languages.

### Section headers preserved

`remove_section_headers=False` is set deliberately. Section headers (`[Verse 1]`, `[Chorus]`, etc.) are preserved in the raw data because Phase 2 (chunking) uses them as structural markers for intelligent splitting.

### Rate limiting & exponential backoff

- **1.5s delay** between requests (configurable via `--delay`)
- On failure: retries up to 5 times with exponential backoff (2s, 4s, 8s, 16s, 32s, capped at 120s)
- Estimated runtime for full dataset: **~12 hours** at 1.5s delay

### Resume support

- On startup, the script checks `lyrics.json` for already-fetched songs and skips them
- Checkpoints every 10 songs (writes to disk mid-run)
- Safe to `Ctrl+C` and re-run — picks up where it left off

### Deduplication

The same song appears across multiple weeks in the Billboard chart data. The script deduplicates by `(title, artist)` before fetching, reducing 18MB of chart rows to 29,680 unique songs.

## Known Limitations

- **Lyrics are scraped, not from the API** — Genius's API does not serve lyrics directly. `lyricsgenius` scrapes them from the song's HTML page, which can break if Genius changes their markup.
- **Fuzzy matching** — `search_song()` returns Genius's top search result, not an exact match. Occasional wrong-song matches are caught in Phase 1.5 (cleaning/validation).
- **`release_date`** — may be `null` for some songs if Genius doesn't have the data.
- **`album`** — may be `null`, or may return a variant name (e.g., "Target Exclusive" editions).

## Bugs Fixed During Development

| Bug | Cause | Fix |
|-----|-------|-----|
| `'Song' object has no attribute 'year'` | `lyricsgenius` v3.11 removed `.year` attribute | Changed to `song.to_dict().get("release_date")` |
| Deprecated `verbose` parameter warning | v3.11 deprecated the constructor param | Replaced with `logging.getLogger("lyricsgenius").setLevel(logging.WARNING)` |
| Genius returning German/Turkish translations | Fuzzy search matched translation pages for featured-artist queries | Added translation terms to `excluded_terms` + `_primary_artist()` to search by primary artist only |
