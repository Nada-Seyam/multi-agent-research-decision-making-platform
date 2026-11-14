"""Download pinned Sentence Transformers models into deployable directories."""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--embedding-destination", type=Path, required=True)
    parser.add_argument("--reranker-model", required=True)
    parser.add_argument("--reranker-revision", required=True)
    parser.add_argument("--reranker-destination", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    args.embedding_destination.parent.mkdir(parents=True, exist_ok=True)
    args.reranker_destination.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    embedding = SentenceTransformer(
        args.embedding_model,
        revision=args.embedding_revision,
        cache_folder=str(args.cache_dir),
        device="cpu",
    )
    embedding.save(str(args.embedding_destination))
    del embedding
    gc.collect()

    reranker = CrossEncoder(
        args.reranker_model,
        revision=args.reranker_revision,
        cache_dir=str(args.cache_dir),
        device="cpu",
    )
    reranker.save(str(args.reranker_destination))
    del reranker
    gc.collect()
    shutil.rmtree(args.cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
