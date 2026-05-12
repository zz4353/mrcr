"""Compute basic statistics for the MRCR JSONL dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_TRAIN_JSONL = Path("data") / "mrcr" / "jsonl" / "train.jsonl"
PERCENTILES = (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show statistics for MRCR JSONL data.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TRAIN_JSONL,
        help=f"Path to a MRCR JSONL file. Default: {DEFAULT_TRAIN_JSONL}",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only read the first N rows. Default: read all rows.",
    )
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="Also compute token counts with tiktoken o200k_base. This is slower.",
    )
    return parser.parse_args()


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0

    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * q)
    return sorted_values[index]


def format_distribution(counter: Counter[int]) -> str:
    total = sum(counter.values())
    lines = []
    for key in sorted(counter):
        count = counter[key]
        ratio = count / total if total else 0
        lines.append(f"  {key}: {count} ({ratio:.2%})")
    return "\n".join(lines)


def format_numeric_stats(name: str, values: list[int]) -> str:
    if not values:
        return f"{name}: no values"

    lines = [
        f"{name}:",
        f"  count: {len(values)}",
        f"  min: {min(values)}",
        f"  mean: {mean(values):.2f}",
        f"  max: {max(values)}",
    ]

    for q in PERCENTILES:
        label = f"p{int(q * 100)}"
        lines.append(f"  {label}: {percentile(values, q)}")

    return "\n".join(lines)


def iter_rows(path: Path, max_rows: int | None) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if max_rows is not None and index >= max_rows:
                break
            if line.strip():
                yield json.loads(line)


def count_tokens(row: dict, encoder) -> int:
    if "messages" in row:
        prompt_text = "".join(message.get("content", "") for message in row["messages"])
    else:
        prompt_text = row.get("prompt", "")

    return len(encoder.encode(prompt_text)) + len(encoder.encode(row.get("answer", "")))


def main() -> None:
    args = parse_args()

    encoder = None
    if args.tokens:
        try:
            import tiktoken
        except ImportError as exc:
            raise SystemExit("Install tiktoken first, or run without --tokens.") from exc
        encoder = tiktoken.get_encoding("o200k_base")

    n_needles = Counter()
    n_chars = []
    total_messages = []
    desired_msg_index = []
    target_relative_position = []
    token_counts = []
    row_count = 0

    for row in iter_rows(args.path, args.max_rows):
        row_count += 1

        n_needles[row["n_needles"]] += 1
        n_chars.append(row["n_chars"])
        total_messages.append(row["total_messages"])
        desired_msg_index.append(row["desired_msg_index"])

        if row["total_messages"] > 1:
            target_relative_position.append(row["desired_msg_index"] / (row["total_messages"] - 1))

        if encoder is not None:
            token_counts.append(count_tokens(row, encoder))

    print(f"File: {args.path}")
    print(f"Rows: {row_count}")
    print()

    print("n_needles distribution:")
    print(format_distribution(n_needles))
    print()

    print(format_numeric_stats("context length in characters (n_chars)", n_chars))
    print()

    print(format_numeric_stats("total_messages", total_messages))
    print()

    print(format_numeric_stats("desired_msg_index", desired_msg_index))
    print()

    if target_relative_position:
        target_percent = [round(value * 100) for value in target_relative_position]
        print(format_numeric_stats("target relative position percent", target_percent))
        print()

    if token_counts:
        print(format_numeric_stats("context length in tokens (o200k_base)", token_counts))


if __name__ == "__main__":
    main()
