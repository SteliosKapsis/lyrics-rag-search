"""One-off filter for non-song contamination in data/raw/cleaned_lyrics.json.

Genius hosts far more than song lyrics -- literary excerpts, movie/TV
scripts, Supreme Court opinions, political speeches, "Listening Log"
blog posts, list-style meta pages, etc. The ingestion step's fuzzy
search against Genius occasionally matched a Billboard chart title to
one of these completely unrelated pages instead of the actual song.

The lyrics-length distribution is bimodal and makes this easy to isolate:
95% of entries sit under ~4,500 characters (median 1,912), consistent
with real transcribed song lyrics, then there is a sharp discontinuity --
a long tail from ~8,000 up to 825,000 characters that is, on inspection,
entirely non-song content (novel chapters, court opinions, film scripts).
There is also a handful of degenerate near-empty entries (1-11 characters)
from deleted/removed Genius pages.

Policy (deliberately conservative -- verified against real short songs
like "PPAP" at 202 chars and "Animals" by Martin Garrix at 84 chars,
both of which survive):
  - drop entries with len(lyrics) > MAX_LYRICS_CHARS (8,000)
  - drop entries with len(lyrics) < MIN_LYRICS_CHARS (15)
  - drop entries whose artist is a known placeholder for a
    deleted/removed Genius page

Usage:
    .venv\\Scripts\\python scripts\\filter_non_song_entries.py

Writes a backup of the pre-filter file to
data/raw/cleaned_lyrics.json.prefilter_bak (only if one doesn't already
exist), an audit trail of every dropped entry to
scripts/non_song_entries_removed.json, then overwrites
data/raw/cleaned_lyrics.json with the filtered list in place.

Run this AFTER scripts/dedupe_cleaned_lyrics.py (order doesn't
technically matter for correctness, but dedupe-then-filter is the
sequence this was designed and tested against).
"""
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = PROJECT_ROOT / "data" / "raw" / "cleaned_lyrics.json"
BACKUP_PATH = CLEANED_PATH.with_suffix(".json.prefilter_bak")
AUDIT_PATH = PROJECT_ROOT / "scripts" / "non_song_entries_removed.json"

MAX_LYRICS_CHARS = 8000
MIN_LYRICS_CHARS = 15
JUNK_ARTISTS = {
    "Deleted Artist",
    "Marked for Deletion (DO NOT USE, USE THE FORUMS INSTEAD)",
}


def reason_for(entry: dict) -> str | None:
    artist = entry.get("artist", "")
    lyrics = entry.get("lyrics", "")
    n = len(lyrics)
    if artist in JUNK_ARTISTS:
        return "junk_artist"
    if n > MAX_LYRICS_CHARS:
        return "too_long"
    if n < MIN_LYRICS_CHARS:
        return "too_short"
    return None


def main() -> None:
    if not CLEANED_PATH.exists():
        log.error("Not found: %s -- nothing to filter.", CLEANED_PATH)
        sys.exit(1)

    with open(CLEANED_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    log.info("Loaded %d entries from %s", len(entries), CLEANED_PATH)

    kept = []
    dropped = []
    for entry in entries:
        reason = reason_for(entry)
        if reason is None:
            kept.append(entry)
        else:
            dropped.append({
                "title": entry.get("title", ""),
                "artist": entry.get("artist", ""),
                "lyrics_length": len(entry.get("lyrics", "")),
                "reason": reason,
            })

    log.info("Kept %d entries, dropped %d", len(kept), len(dropped))
    by_reason: dict[str, int] = {}
    for d in dropped:
        by_reason[d["reason"]] = by_reason.get(d["reason"], 0) + 1
    for reason, count in sorted(by_reason.items()):
        log.info("  dropped as %s: %d", reason, count)

    if not dropped:
        log.info("Nothing to drop -- file already clean, no changes written.")
        return

    if BACKUP_PATH.exists():
        log.info("Backup already exists at %s -- not overwriting it.", BACKUP_PATH)
    else:
        with open(BACKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        log.info("Backed up pre-filter state (%d entries) to %s", len(entries), BACKUP_PATH)

    AUDIT_PATH.parent.mkdir(exist_ok=True)
    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(dropped, key=lambda d: -d["lyrics_length"]), f, ensure_ascii=False, indent=2)
    log.info("Wrote audit trail of dropped entries to %s", AUDIT_PATH)

    with open(CLEANED_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d filtered entries to %s", len(kept), CLEANED_PATH)
    log.info("Done. Re-run pipeline/chunking.py next.")


if __name__ == "__main__":
    main()
