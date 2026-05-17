# Phase 1 — Data Ingestion

You are an expert Python developer specializing in data engineering. I'm building a RAG pipeline for my thesis on lyrics-based semantic search.

Help me write a Python script to ingest lyrics data for my project. The data source will be the Genius API (I have an API key). The script should:

1. Accept a list of song titles + artists as input (from a CSV file with columns: title, artist)
2. For each song, fetch from Genius API:
   - Full lyrics
   - Artist name
   - Release date
   - Primary album name
3. Handle rate limiting gracefully with exponential backoff
4. Save results to a structured JSON file where each entry has: title, artist, release_date, album, lyrics, genius_id
5. Log failed fetches to a separate file so I can retry them

If you're unsure about any Genius API endpoint behavior, say so. Use lyricsgenius as the Python client library.
