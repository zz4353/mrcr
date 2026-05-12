"""Print the first raw sample from data/mrcr/jsonl/train.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TRAIN_JSONL = Path("data") / "mrcr" / "jsonl" / "train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the first raw MRCR train JSONL row.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TRAIN_JSONL,
        help=f"Path to train.jsonl. Default: {DEFAULT_TRAIN_JSONL}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.path.open("r", encoding="utf-8") as file:
        first_line = file.readline()

    if not first_line:
        raise ValueError(f"No rows found in {args.path}")

    print(first_line, end="")


if __name__ == "__main__":
    main()
