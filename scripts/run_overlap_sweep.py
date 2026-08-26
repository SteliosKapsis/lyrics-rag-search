"""Automates the overlap-lines ablation: chunking.py + embedding.py for
overlap=1 and overlap=2 (chunk size held at the default 400), across
both local embedding models (MiniLM, BGE) so notebooks/evaluation.ipynb
Section 4's "Chunking Parameter Sweep" cell (already built, previously
unpopulated for overlap>0) auto-detects the results with no notebook
changes needed.

Skips overlap=0 -- that's just the existing default faiss.index /
faiss_bge.index, already built by the regular embedding.py run.

Naming convention (matches notebooks/evaluation.ipynb cell 25 exactly):
    MiniLM: faiss_c400_o{overlap}.index / metadata_c400_o{overlap}.json / bm25_c400_o{overlap}.pkl
    BGE:    faiss_c400_o{overlap}_bge.index / metadata_c400_o{overlap}_bge.json / bm25_c400_o{overlap}_bge.pkl

Usage:
    .venv\\Scripts\\python scripts\\run_overlap_sweep.py

Idempotent / resumable at the step level: if an index file already
exists on disk, that chunking+embedding step is skipped rather than
redone. Each individual embedding.py call is itself checkpointed and
resumable if interrupted mid-run (see pipeline/embedding.py).

Run this AFTER the main 5-variant embedding.py sequence (or independently
-- it doesn't touch faiss.index/faiss_bge.index, only the new _c400_o{N}
files), since it reads the same data/processed/chunks.jsonl-derived
inputs and is unrelated to the OpenAI variant.
"""
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PYTHON = sys.executable  # the interpreter running this script -- reused for subprocess calls

OVERLAPS = [1, 2]
MAX_CHUNK_SIZE = 400  # the existing default; this sweep varies overlap only

MODELS = [
    {"label": "MiniLM (default)", "flags": [], "suffix": ""},
    {"label": "BGE", "flags": ["--model", "BAAI/bge-small-en-v1.5"], "suffix": "_bge"},
]


def run(cmd: list[str], step_name: str) -> None:
    log.info("--- %s ---", step_name)
    log.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        log.error("FAILED (exit code %d): %s", result.returncode, step_name)
        sys.exit(result.returncode)


def main() -> None:
    for overlap in OVERLAPS:
        chunks_path = PROCESSED_DIR / f"chunks_c{MAX_CHUNK_SIZE}_o{overlap}.jsonl"

        if chunks_path.exists():
            log.info("Already exists, skipping chunking: %s", chunks_path)
        else:
            run(
                [
                    PYTHON, str(PROJECT_ROOT / "pipeline" / "chunking.py"),
                    "--max-chunk-size", str(MAX_CHUNK_SIZE),
                    "--overlap-lines", str(overlap),
                    "--output", str(chunks_path),
                ],
                f"chunking (overlap={overlap})",
            )

        for model in MODELS:
            index_path = PROCESSED_DIR / f"faiss_c{MAX_CHUNK_SIZE}_o{overlap}{model['suffix']}.index"
            metadata_path = PROCESSED_DIR / f"metadata_c{MAX_CHUNK_SIZE}_o{overlap}{model['suffix']}.json"
            bm25_path = PROCESSED_DIR / f"bm25_c{MAX_CHUNK_SIZE}_o{overlap}{model['suffix']}.pkl"

            if index_path.exists():
                log.info("Already exists, skipping embedding: %s", index_path)
                continue

            run(
                [
                    PYTHON, str(PROJECT_ROOT / "pipeline" / "embedding.py"),
                    "--input", str(chunks_path),
                    "--index-output", str(index_path),
                    "--metadata-output", str(metadata_path),
                    "--bm25-output", str(bm25_path),
                    *model["flags"],
                ],
                f"embedding (overlap={overlap}, model={model['label']})",
            )

    log.info("")
    log.info("Overlap sweep complete. %d overlap values x %d models = %d new index variants.",
              len(OVERLAPS), len(MODELS), len(OVERLAPS) * len(MODELS))
    log.info("notebooks/evaluation.ipynb Section 4 (Chunking Parameter Sweep) will auto-detect these.")


if __name__ == "__main__":
    main()
