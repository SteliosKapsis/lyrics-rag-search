# Phase 1.5 — Data Cleaning & Validation

You are an expert Python developer specializing in data preprocessing for NLP pipelines. I'm building a RAG pipeline for my thesis on lyrics-based semantic search.

I have a JSON file (lyrics.json) where each entry contains: title, artist, release_date, album, lyrics, genius_id. The lyrics were scraped from Genius via lyricsgenius and contain noise.

Help me write a Python script to clean and validate this data. Specifically:

1. Remove Genius-specific artifacts from lyrics:
   - Embedded contributor/translation text (e.g., "123 ContributorsTranslations...")
   - Advertisement or promotional text injected by the scraper
   - Empty or placeholder entries
2. Normalize section headers (e.g., [Verse 1], [Chorus], [Bridge]) — keep them as-is but ensure consistent formatting, since they will be useful as structural markers during chunking
3. Validate each entry:
   - Flag entries where lyrics are suspiciously short (< 100 characters) or empty
   - Flag entries where the returned artist/title from Genius doesn't closely match the input (possible wrong song match)
4. Output:
   - A cleaned JSON file (cleaned_lyrics.json) with the same schema
   - A validation report (validation_report.csv) listing flagged entries with the reason for flagging, so I can manually review them

If you make assumptions about common Genius scraping artifacts, list them. Use only standard library + rapidfuzz for fuzzy string matching.
