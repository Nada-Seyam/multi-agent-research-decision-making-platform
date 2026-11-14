"""Download the pinned analytics NLI model into a deployable directory."""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    args.destination.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=str(args.cache_dir),
        use_fast=False,
    )
    tokenizer.save_pretrained(args.destination)
    del tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=str(args.cache_dir),
    )
    model.save_pretrained(args.destination, safe_serialization=True)
    del model
    gc.collect()

    # The destination contains the runtime copy; do not duplicate the download
    # cache inside the final image layer.
    shutil.rmtree(args.cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
