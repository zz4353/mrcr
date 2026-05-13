"""Extract small stratified val/test subsets from MRCR JSONL.

Default output:
  data/mrcr/mini/val.jsonl
  data/mrcr/mini/test.jsonl

Strata:
  - n_needles: 2, 4, 8
  - token bins up to 128K: 4K-8K, 8K-16K, 16K-32K, 32K-64K, 64K-128K
  - target position: early, middle, late

Default quotas:
  - val: every (n_needles, token_bin) gets 1 sample
  - test: n_needles 2/4 get 2 samples, n_needles 8 gets 1 sample

So the intended sizes are val=15 and test=75 when all cells have enough rows.
Val covers every n_needles/token_bin pair and balances target positions overall.
Test covers every n_needles/token_bin/target_position cell.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data") / "mrcr" / "jsonl" / "train.jsonl"
DEFAULT_OUTPUT_DIR = Path("data") / "mrcr" / "mini"
TOKEN_BINS = (
    (4_096, 8_192),
    (8_192, 16_384),
    (16_384, 32_768),
    (32_768, 65_536),
    (65_536, 131_072),
)
TARGET_POSITION_BINS = ("early", "middle", "late")


@dataclass(frozen=True)
class Candidate:
    row_index: int
    row: dict[str, Any]
    n_tokens: int
    token_bin: str
    target_position_bin: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mini MRCR val/test JSONL files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input MRCR JSONL path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking. Default: 42",
    )
    parser.add_argument(
        "--test-quota-2-4",
        type=int,
        default=2,
        help="Test samples per cell for n_needles=2 and 4. Default: 2",
    )
    parser.add_argument(
        "--test-quota-8",
        type=int,
        default=1,
        help="Test samples per cell for n_needles=8. Default: 1",
    )
    parser.add_argument(
        "--val-quota",
        type=int,
        default=1,
        help="Val samples per (n_needles, token_bin). Default: 1",
    )
    return parser.parse_args()


def load_encoder():
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("Install tiktoken first: pip install tiktoken") from exc

    return tiktoken.get_encoding("o200k_base")


def count_tokens(row: dict[str, Any], encoder) -> int:
    prompt_text = "".join(message.get("content", "") for message in row["messages"])
    return len(encoder.encode(prompt_text)) + len(encoder.encode(row.get("answer", "")))


def token_bin_for(n_tokens: int) -> str | None:
    for lower, upper in TOKEN_BINS:
        if lower < n_tokens <= upper:
            return f"{lower // 1024}k-{upper // 1024}k"
    return None


def target_position_bin(row: dict[str, Any]) -> str:
    total_messages = row["total_messages"]
    if total_messages <= 1:
        return "early"

    relative_position = row["desired_msg_index"] / (total_messages - 1)
    if relative_position < 1 / 3:
        return "early"
    if relative_position < 2 / 3:
        return "middle"
    return "late"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for row_index, line in enumerate(file):
            if line.strip():
                yield row_index, json.loads(line)


def collect_candidates(path: Path) -> dict[tuple[int, str, str], list[Candidate]]:
    encoder = load_encoder()
    cells: dict[tuple[int, str, str], list[Candidate]] = {}

    for row_index, row in iter_jsonl(path):
        if row["n_needles"] not in (2, 4, 8):
            continue

        n_tokens = count_tokens(row, encoder)
        token_bin = token_bin_for(n_tokens)
        if token_bin is None:
            continue

        position_bin = target_position_bin(row)
        key = (row["n_needles"], token_bin, position_bin)
        cells.setdefault(key, []).append(
            Candidate(
                row_index=row_index,
                row=row,
                n_tokens=n_tokens,
                token_bin=token_bin,
                target_position_bin=position_bin,
            )
        )

    return cells


def test_quota_for(n_needles: int, args: argparse.Namespace) -> int:
    if n_needles == 8:
        return args.test_quota_8
    return args.test_quota_2_4


def select_candidates(
    cells: dict[tuple[int, str, str], list[Candidate]], args: argparse.Namespace
) -> tuple[list[Candidate], list[Candidate], list[str]]:
    rng = random.Random(args.seed)
    val: list[Candidate] = []
    test: list[Candidate] = []
    used_row_indices: set[int] = set()
    warnings: list[str] = []

    token_bin_labels = [f"{lower // 1024}k-{upper // 1024}k" for lower, upper in TOKEN_BINS]

    for needle_index, n_needles in enumerate((2, 4, 8)):
        for token_index, token_bin_label in enumerate(token_bin_labels):
            desired_position = TARGET_POSITION_BINS[
                (needle_index + token_index) % len(TARGET_POSITION_BINS)
            ]
            candidates = list(cells.get((n_needles, token_bin_label, desired_position), []))

            val_quota = args.val_quota

            if len(candidates) < val_quota:
                fallback_candidates = []
                for position_bin in TARGET_POSITION_BINS:
                    if position_bin != desired_position:
                        fallback_candidates.extend(cells.get((n_needles, token_bin_label, position_bin), []))

                warnings.append(
                    f"val {(n_needles, token_bin_label, desired_position)}: wanted "
                    f"{val_quota}, found {len(candidates)}; falling back to other positions"
                )
                candidates.extend(fallback_candidates)

            candidates = [item for item in candidates if item.row_index not in used_row_indices]

            selected = rng.sample(candidates, k=min(val_quota, len(candidates)))
            val.extend(selected)
            used_row_indices.update(item.row_index for item in selected)

    for n_needles in (2, 4, 8):
        for token_bin_label in token_bin_labels:
            for position_bin in TARGET_POSITION_BINS:
                key = (n_needles, token_bin_label, position_bin)
                candidates = cells.get(key, [])
                candidates = [item for item in candidates if item.row_index not in used_row_indices]

                test_quota = test_quota_for(n_needles, args)

                if len(candidates) < test_quota:
                    warnings.append(
                        f"test {key}: wanted {test_quota}, found {len(candidates)}; using what is available"
                    )

                selected = rng.sample(candidates, k=min(test_quota, len(candidates)))
                test.extend(selected)
                used_row_indices.update(item.row_index for item in selected)

    overlap = {item.row_index for item in val} & {item.row_index for item in test}
    if overlap:
        raise RuntimeError(f"Val/test overlap found: {sorted(overlap)[:10]}")

    return val, test, warnings


def write_jsonl(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for candidate in candidates:
            row = dict(candidate.row)
            row["mrcr_source_row_index"] = candidate.row_index
            row["n_tokens_o200k"] = candidate.n_tokens
            row["token_bin_o200k"] = candidate.token_bin
            row["target_position_bin"] = candidate.target_position_bin
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(name: str, candidates: list[Candidate]) -> None:
    print(f"{name}: {len(candidates)} samples")
    counts: dict[tuple[int, str, str], int] = {}
    for candidate in candidates:
        key = (
            candidate.row["n_needles"],
            candidate.token_bin,
            candidate.target_position_bin,
        )
        counts[key] = counts.get(key, 0) + 1

    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


def main() -> None:
    args = parse_args()
    cells = collect_candidates(args.input)
    val, test, warnings = select_candidates(cells, args)

    val_path = args.output_dir / "val.jsonl"
    test_path = args.output_dir / "test.jsonl"
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    print(f"Wrote {val_path}")
    summarize("val", val)
    print()
    print(f"Wrote {test_path}")
    summarize("test", test)

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  {warning}")


if __name__ == "__main__":
    main()
