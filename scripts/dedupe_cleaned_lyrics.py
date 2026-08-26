"""One-off dedup pass over data/raw/cleaned_lyrics.json.

cleaned_lyrics.json currently has 19,320 entries but only 14,997 unique
(title, artist) pairs -- 4,245 songs are present more than once (mostly
Billboard chart entries that charted across multiple weeks/years and were
scraped/kept more than once). Left as-is, pipeline/chunking.py's
process_songs() does not deduplicate, so every duplicate gets chunked and
indexed again, giving duplicated songs a disproportionate share of FAISS/
BM25 candidate slots relative to non-duplicated songs -- a real, if subtle,
bias in retrieval evaluation, confirmed to affect 14 of the 75 golden-set
answerable queries.

Policy: keep the first occurrence per (title, artist) in file order, drop
the rest. Deterministic, no judgment call about which copy is "better" --
the two occurrences of any given duplicate are effectively identical
(same lyrics, cleaned the same way) since they came from the same Genius
lookup for the same song.

Usage:
    .venv\\Scripts\\python scripts\\dedupe_cleaned_lyrics.py

Writes a timestamped-free backup of the original file to
data/raw/cleaned_lyrics.json.bak (only if one doesn't already exist, so
re-running this script is safe and won't clobber the real original with
an already-deduped copy), then overwrites data/raw/cleaned_lyrics.json
with the deduplicated list in place -- so every downstream command
(pipeline/chunking.py with no extra flags) keeps working unchanged.
"""
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = PROJECT_ROOT / "data" / "raw" / "cleaned_lyrics.json"
BACKUP_PATH = CLEANED_PATH.with_suffix(".json.bak")


def main() -> None:
    if not CLEANED_PATH.exists():
        log.error("Not found: %s -- nothing to dedupe.", CLEANED_PATH)
        sys.exit(1)

    with open(CLEANED_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    log.info("Loaded %d entries from %s", len(entries), CLEANED_PATH)

    seen = set()
    deduped = []
    dropped = 0
    for entry in entries:
        key = (entry.get("title"), entry.get("artist"))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(entry)

    log.info(
        "Deduplicated: %d unique (title, artist) pairs kept, %d duplicate entries dropped",
        len(deduped), dropped,
    )

    if dropped == 0:
        log.info("No duplicates found -- file already clean, nothing to write.")
        return

    if BACKUP_PATH.exists():
        log.info("Backup already exists at %s -- not overwriting it.", BACKUP_PATH)
    else:
        with open(BACKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        log.info("Backed up original (%d entries) to %s", len(entries), BACKUP_PATH)

    with open(CLEANED_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d deduplicated entries to %s", len(deduped), CLEANED_PATH)
    log.info("Done. Re-run pipeline/chunking.py next -- it will now produce fewer, non-duplicated chunks.")


if __name__ == "__main__":
    main()
