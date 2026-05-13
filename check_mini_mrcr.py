"""Validate the extracted MRCR mini val/test files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


VAL_PATH = Path("data") / "mrcr" / "mini" / "val.jsonl"
TEST_PATH = Path("data") / "mrcr" / "mini" / "test.jsonl"

TOKEN_BINS = {
    "4k-8k": (4_096, 8_192),
    "8k-16k": (8_192, 16_384),
    "16k-32k": (16_384, 32_768),
    "32k-64k": (32_768, 65_536),
    "64k-128k": (65_536, 131_072),
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_token_bin(row: dict) -> None:
    label = row["token_bin_o200k"]
    lower, upper = TOKEN_BINS[label]
    n_tokens = row["n_tokens_o200k"]
    if not lower < n_tokens <= upper:
        raise AssertionError(
            f"row {row['mrcr_source_row_index']} has {n_tokens} tokens, "
            f"outside bin {label} ({lower}, {upper}]"
        )


def summarize(name: str, rows: list[dict]) -> None:
    print(f"{name}: {len(rows)} rows")

    print("  n_needles:", dict(sorted(Counter(row["n_needles"] for row in rows).items())))
    print("  token_bin:", dict(sorted(Counter(row["token_bin_o200k"] for row in rows).items())))
    print(
        "  target_position:",
        dict(sorted(Counter(row["target_position_bin"] for row in rows).items())),
    )

    cell_counts = Counter(
        (row["n_needles"], row["token_bin_o200k"], row["target_position_bin"])
        for row in rows
    )
    print(f"  cells: {len(cell_counts)}")
    print(f"  max_tokens: {max(row['n_tokens_o200k'] for row in rows)}")


def main() -> None:
    val_rows = read_jsonl(VAL_PATH)
    test_rows = read_jsonl(TEST_PATH)

    for row in val_rows + test_rows:
        assert_token_bin(row)

    val_ids = {row["mrcr_source_row_index"] for row in val_rows}
    test_ids = {row["mrcr_source_row_index"] for row in test_rows}
    overlap = val_ids & test_ids
    if overlap:
        raise AssertionError(f"val/test overlap: {sorted(overlap)[:10]}")

    if len(val_ids) != len(val_rows):
        raise AssertionError("duplicate rows inside val")
    if len(test_ids) != len(test_rows):
        raise AssertionError("duplicate rows inside test")

    if len(val_rows) != 15:
        raise AssertionError(f"expected 15 val rows, got {len(val_rows)}")
    if len(test_rows) != 75:
        raise AssertionError(f"expected 75 test rows, got {len(test_rows)}")

    summarize("val", val_rows)
    summarize("test", test_rows)
    print("OK")


if __name__ == "__main__":
    main()
