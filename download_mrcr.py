"""Download the full OpenAI MRCR benchmark into this repository.

The script saves two forms by default:
1. Hugging Face `save_to_disk` format under `data/mrcr/hf_dataset`.
2. JSONL files under `data/mrcr/jsonl`, with `prompt` expanded into `messages`.

Usage:
    python download_mrcr.py
    python download_mrcr.py --output-dir data/mrcr --no-jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


DEFAULT_REPO_ID = "openai/mrcr"
DEFAULT_OUTPUT_DIR = Path("data") / "mrcr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the full OpenAI MRCR dataset.")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store all downloaded/exported data. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Only save the Hugging Face dataset; skip expanded JSONL export.",
    )
    return parser.parse_args()


def parse_prompt_messages(prompt: str) -> list[dict[str, Any]]:
    """MRCR stores prompt as a JSON-encoded list of chat messages."""
    messages = json.loads(prompt)
    if not isinstance(messages, list):
        raise ValueError("Expected prompt to decode to a list of messages.")
    return messages


def export_split_to_jsonl(dataset: Any, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for row in dataset:
            obj = dict(row)
            obj["messages"] = parse_prompt_messages(obj["prompt"])
            del obj["prompt"]
            file.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    hf_dataset_dir = output_dir / "hf_dataset"
    jsonl_dir = output_dir / "jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {args.repo_id}")
    dataset = load_dataset(args.repo_id)

    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    print(f"Saving Hugging Face dataset to: {hf_dataset_dir}")
    dataset.save_to_disk(str(hf_dataset_dir))

    if not args.no_jsonl:
        print(f"Exporting expanded JSONL files to: {jsonl_dir}")
        for split_name, split_dataset in dataset.items():
            output_path = jsonl_dir / f"{split_name}.jsonl"
            count = export_split_to_jsonl(split_dataset, output_path)
            print(f"  {split_name}: {count} rows -> {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
