"""
Semi-automated chunk-ID ground-truth backfill for notebooks/evaluation.ipynb's
TEST_SET (see Documentation/phase_6_evaluation.md's golden-set versioning note
and Instructions/AUDIT_RESULTS.md file 06/01 findings).

For each answerable TEST_SET entry, finds candidate chunks belonging to the
expected song in data/processed/metadata.json, then fuzzy-matches the query
text against each candidate chunk's text to propose which chunk_index the
query is most likely grounded in.

This does NOT fully automate ground-truth creation. For "direct" (verbatim
lyric) queries the proposed match is usually reliable. For "semantic",
"partial", and "ambiguous" queries — which are deliberately paraphrases, not
lyric excerpts — this can only narrow the field, not definitively identify
the one correct chunk; those need a human (or a careful read of the top
candidates) to confirm. Read-only against metadata.json; writes a review
file, never touches the notebook directly.

Usage (from project root):
    .venv/Scripts/python scripts/backfill_chunk_ids.py
"""

import json
import re
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "chunk_id_backfill_review.json"

# Confidence cutoff below which a proposed match is flagged "low_confidence"
# for mandatory human review rather than auto-accepted.
CONFIDENCE_CUTOFF = 80


def normalize(s: str) -> str:
    """Identical to notebooks/evaluation.ipynb cell 7's normalize() — keep in sync."""
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def load_test_set() -> list[dict]:
    """
    Load TEST_SET by executing notebooks/evaluation.ipynb's cell 5 source in an
    isolated namespace — avoids hand-retyping 75 entries and keeps this script
    always in sync with whatever the notebook actually defines.
    """
    nb = json.loads((PROJECT_ROOT / "notebooks" / "evaluation.ipynb").read_text(encoding="utf-8"))
    cell5_source = "".join(nb["cells"][5]["source"])
    assert "TEST_SET = [" in cell5_source, "notebooks/evaluation.ipynb cell 5 is no longer the TEST_SET cell"
    ns: dict = {"defaultdict": __import__("collections").defaultdict}
    exec(compile(cell5_source, "cell5", "exec"), ns)
    return ns["TEST_SET"]


def main():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(metadata)} chunks from {METADATA_PATH}")

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} TEST_SET entries from notebooks/evaluation.ipynb\n")

    # Index metadata by normalized (title, artist) for fast candidate lookup
    by_song: dict[tuple[str, str], list[dict]] = {}
    for m in metadata:
        key = (normalize(m["title"]), normalize(m["artist"]))
        by_song.setdefault(key, []).append(m)

    results = []
    low_confidence_count = 0
    no_candidates_count = 0

    for entry in test_set:
        norm_title = normalize(entry["expected_title"])
        norm_artist = normalize(entry["expected_artist"])
        candidates = by_song.get((norm_title, norm_artist), [])

        if not candidates:
            # Fall back to substring match (mirrors find_rank's partial-match logic)
            for (t, a), chunks in by_song.items():
                shorter = min(norm_title, t, key=len)
                if len(shorter) >= 8 and (norm_title in t or t in norm_title) and (norm_artist in a or a in norm_artist):
                    candidates = chunks
                    break

        if not candidates:
            no_candidates_count += 1
            results.append({
                "query": entry["query"], "expected_title": entry["expected_title"],
                "expected_artist": entry["expected_artist"], "type": entry["type"],
                "matched_chunk_index": None, "match_score": 0.0, "status": "no_candidates_found",
                "chunk_text_preview": None,
            })
            print(f"[NO CANDIDATES] {entry['expected_title']!r} by {entry['expected_artist']!r} "
                  f"not found in metadata.json — query: {entry['query'][:60]!r}")
            continue

        norm_query = normalize(entry["query"])
        best = max(candidates, key=lambda c: fuzz.partial_ratio(norm_query, normalize(c["text"])))
        score = fuzz.partial_ratio(norm_query, normalize(best["text"]))
        status = "ok" if score >= CONFIDENCE_CUTOFF else "low_confidence"
        if status == "low_confidence":
            low_confidence_count += 1

        results.append({
            "query": entry["query"], "expected_title": entry["expected_title"],
            "expected_artist": entry["expected_artist"], "type": entry["type"],
            "matched_chunk_index": best["chunk_index"], "match_score": round(score, 1), "status": status,
            "chunk_text_preview": best["text"][:200],
        })

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"Wrote {len(results)} proposed annotations to {OUTPUT_PATH}")
    print(f"  ok (>= {CONFIDENCE_CUTOFF} score): {sum(1 for r in results if r['status'] == 'ok')}")
    print(f"  low_confidence (< {CONFIDENCE_CUTOFF}): {low_confidence_count}")
    print(f"  no_candidates_found: {no_candidates_count}")
    print("\nBy type:")
    for qtype in sorted(set(r["type"] for r in results)):
        type_results = [r for r in results if r["type"] == qtype]
        ok = sum(1 for r in type_results if r["status"] == "ok")
        print(f"  {qtype}: {ok}/{len(type_results)} auto-confident")
    print(f"\nReview {OUTPUT_PATH.name} before trusting any 'low_confidence' or 'no_candidates_found' entries.")


if __name__ == "__main__":
    main()
