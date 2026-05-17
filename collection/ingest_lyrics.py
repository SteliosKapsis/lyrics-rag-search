"""
Lyrics ingestion script for RAG pipeline.
Fetches lyrics + metadata from Genius API via lyricsgenius.

Usage (from project root):
    .venv/Scripts/python collection/ingest_lyrics.py --input charts.csv --token YOUR_GENIUS_API_TOKEN

Or with .env loaded:
    .venv/Scripts/python collection/ingest_lyrics.py --input charts.csv

The input CSV must have columns: song, artist (matching charts.csv format).
Also accepts: title, artist as column names.

Outputs go to data/raw/lyrics.json and data/failed/failed_fetches.csv by default.
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import lyricsgenius

# Load .env from project root so GENIUS_API_TOKEN is available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_songs(csv_path: str) -> list[dict]:
    """Load song list from CSV. Supports 'song'/'title' and 'artist' columns."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        # Resolve the title column name
        if "song" in columns:
            title_col = "song"
        elif "title" in columns:
            title_col = "title"
        else:
            raise ValueError(f"CSV must have a 'song' or 'title' column. Found: {columns}")

        if "artist" not in columns:
            raise ValueError(f"CSV must have an 'artist' column. Found: {columns}")

        seen = set()
        for row in reader:
            title = row[title_col].strip()
            artist = row["artist"].strip()
            key = (title.lower(), artist.lower())
            if key not in seen:
                seen.add(key)
                songs.append({"title": title, "artist": artist})

    log.info("Loaded %d unique songs from %s", len(songs), csv_path)
    return songs


def load_already_fetched(output_path: str) -> set[tuple[str, str]]:
    """Load keys of songs already in the output file to enable resuming."""
    if not os.path.exists(output_path):
        return set()
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
    return {(entry["title"].lower(), entry["artist"].lower()) for entry in data}


def _primary_artist(artist: str) -> str:
    """Extract the primary artist from a Billboard-style artist string.

    Examples:
        'The Kid LAROI & Justin Bieber' -> 'The Kid LAROI'
        'Lil Nas X & Jack Harlow'       -> 'Lil Nas X'
        'Drake Featuring Future'         -> 'Drake'
        'Adele'                          -> 'Adele'
    """
    import re
    return re.split(r"\s+(?:&|Featuring|Feat\.?|feat\.?|And|and|,|X\s)", artist)[0].strip()


def fetch_song(
    genius: lyricsgenius.Genius,
    title: str,
    artist: str,
    max_retries: int = 5,
) -> dict | None:
    """
    Fetch a single song's lyrics and metadata from Genius with exponential backoff.

    Returns a dict with: title, artist, release_date, album, lyrics, genius_id
    or None if the song couldn't be found/fetched.
    """
    primary = _primary_artist(artist)

    for attempt in range(max_retries):
        try:
            song = genius.search_song(title, primary)

            if song is None:
                log.warning("No result for '%s' by '%s'", title, artist)
                return None

            # Extract metadata from the song dict.
            # lyricsgenius returns album as a dict and release_date as a string.
            song_data = song.to_dict()

            album_name = None
            if song.album:
                album_name = song.album if isinstance(song.album, str) else song.album.get("name")

            return {
                "title": song.title,
                "artist": song.artist,
                "release_date": song_data.get("release_date"),  # e.g. "2021-10-15"
                "album": album_name,
                "lyrics": song.lyrics,
                "genius_id": song_data.get("id"),
            }

        except (Exception, AssertionError) as e:
            # Detect 429 rate limit — no point retrying, quota is exhausted
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                log.error(
                    "Rate limit hit for '%s' by '%s' — stopping retries. "
                    "Wait for quota reset before restarting.", title, artist
                )
                return None

            wait = min(2**attempt * 2, 120)
            log.warning(
                "Attempt %d/%d failed for '%s' by '%s': %s. Retrying in %ds...",
                attempt + 1, max_retries, title, artist, e, wait,
            )
            time.sleep(wait)

    return None


def save_results(results: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_failures(failures: list[dict], failures_path: str) -> None:
    with open(failures_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "artist", "error"])
        writer.writeheader()
        writer.writerows(failures)


def main():
    parser = argparse.ArgumentParser(description="Ingest lyrics from Genius API")
    parser.add_argument("--input", required=True, help="Path to input CSV (columns: song/title, artist)")
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "raw" / "lyrics.json"
    default_failures = project_root / "data" / "failed" / "failed_fetches.csv"

    parser.add_argument("--output", default=str(default_output), help="Path to output JSON file")
    parser.add_argument("--failures", default=str(default_failures), help="Path to failed-fetches log CSV")
    parser.add_argument("--token", default=None, help="Genius API token (or set GENIUS_API_TOKEN env var)")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds to wait between requests (default: 1.5)")
    args = parser.parse_args()

    token = args.token or os.environ.get("GENIUS_API_TOKEN")
    if not token:
        parser.error("Provide a Genius API token via --token or GENIUS_API_TOKEN env var")

    songs = load_songs(args.input)

    # Resume support: skip songs already in the output file
    already_fetched = load_already_fetched(args.output)
    if already_fetched:
        log.info("Found %d songs already fetched — skipping those", len(already_fetched))

    # Load existing results so we append to them
    results = []
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            results = json.load(f)

    genius = lyricsgenius.Genius(token, remove_section_headers=False)
    logging.getLogger("lyricsgenius").setLevel(logging.WARNING)
    genius.excluded_terms = [
        "(Remix)", "(Live)",
        "Übersetzung", "Übersetzungen",      # German translations
        "Traduzione", "Traduzioni",            # Italian translations
        "Traduction", "Traductions",           # French translations
        "Traducción", "Traducciones",          # Spanish translations
        "Перевод",                             # Russian translations
        "Çeviri",                              # Turkish translations
        "翻訳",                                # Japanese translations
        "번역",                                # Korean translations
    ]

    failures = []
    total = len(songs)

    for i, song in enumerate(songs, 1):
        key = (song["title"].lower(), song["artist"].lower())
        if key in already_fetched:
            continue

        log.info("[%d/%d] Fetching: '%s' by '%s'", i, total, song["title"], song["artist"])

        result = fetch_song(genius, song["title"], song["artist"])
        if result:
            results.append(result)
            # Incremental save — don't lose progress on crash
            if len(results) % 10 == 0:
                save_results(results, args.output)
                log.info("Checkpoint: saved %d results", len(results))
        else:
            failures.append({
                "title": song["title"],
                "artist": song["artist"],
                "error": "not_found_or_max_retries",
            })

        time.sleep(args.delay)

    # Final save
    save_results(results, args.output)
    log.info("Done. Saved %d songs to %s", len(results), args.output)

    if failures:
        save_failures(failures, args.failures)
        log.warning("%d songs failed. Logged to %s", len(failures), args.failures)


if __name__ == "__main__":
    main()
