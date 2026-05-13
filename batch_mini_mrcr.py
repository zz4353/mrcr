"""Submit, retrieve, and download MRCR mini Batch API jobs.

Examples:
    python batch_mini_mrcr.py submit --split val --mode text --model gpt-5
    python batch_mini_mrcr.py retrieve --split val --mode text --model gpt-5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from mrcr_image_history import ConversationImageRenderer, build_image_history_messages


DEFAULT_DATA_DIR = Path("data") / "mini"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_ENV_FILE = Path(".env")
ENDPOINT = "/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRCR mini OpenAI Batch API helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "submit", "retrieve"):
        subparser = subparsers.add_parser(command)
        add_dataset_args(subparser)
        if command in ("create", "submit"):
            subparser.add_argument("--max-completion-tokens", type=int, default=None)

    return parser.parse_args()


def add_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    add_api_args(parser)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mode", choices=("text", "image-history"), default="text")
    parser.add_argument("--recent-turns", type=int, default=3)
    parser.add_argument("--model", default="gpt-5")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def require_api_key(env_file: Path) -> None:
    load_env_file(env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY or add it to .env before using the Batch API.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)


def stem_for(args: argparse.Namespace) -> str:
    return f"mrcr_{args.split}_{args.mode}_{safe_model_name(args.model)}"


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    stem = stem_for(args)
    return {
        "batch_input": args.results_dir / f"{stem}_batch_input.jsonl",
        "manifest": args.results_dir / f"{stem}_manifest.jsonl",
        "batch": args.results_dir / f"{stem}_batch.json",
        "batch_output": args.results_dir / f"{stem}_batch_output.jsonl",
        "batch_errors": args.results_dir / f"{stem}_batch_errors.jsonl",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_messages(
    row: dict[str, Any],
    *,
    mode: str,
    recent_turns: int,
    renderer: ConversationImageRenderer | None,
) -> list[dict[str, Any]]:
    if mode == "text":
        return [dict(message) for message in row["messages"]]

    return build_image_history_messages(
        row["messages"],
        recent_turns=recent_turns,
        renderer=renderer,
    )


def build_batch_files(args: argparse.Namespace) -> dict[str, Path]:
    rows = read_jsonl(args.data_dir / f"{args.split}.jsonl")
    renderer = ConversationImageRenderer() if args.mode == "image-history" else None
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        source_index = row["mrcr_source_row_index"]
        custom_id = f"mrcr-{args.split}-{args.mode}-row-{source_index}"
        body: dict[str, Any] = {
            "model": args.model,
            "messages": build_messages(
                row,
                mode=args.mode,
                recent_turns=args.recent_turns,
                renderer=renderer,
            ),
        }
        if args.max_completion_tokens is not None:
            body["max_completion_tokens"] = args.max_completion_tokens

        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": ENDPOINT,
                "body": body,
            }
        )
        manifest.append(
            {
                "custom_id": custom_id,
                "split": args.split,
                "mode": args.mode,
                "model": args.model,
                "row_index_in_split": index,
                "mrcr_source_row_index": source_index,
                "n_needles": row.get("n_needles"),
                "n_tokens_o200k": row.get("n_tokens_o200k"),
                "token_bin_o200k": row.get("token_bin_o200k"),
                "target_position_bin": row.get("target_position_bin"),
                "random_string_to_prepend": row["random_string_to_prepend"],
                "answer": row["answer"],
            }
        )

    paths = output_paths(args)
    write_jsonl(paths["batch_input"], requests)
    write_jsonl(paths["manifest"], manifest)
    print(f"Wrote {len(requests)} requests: {paths['batch_input']}")
    print(f"Wrote manifest: {paths['manifest']}")
    return paths


def submit_batch(args: argparse.Namespace) -> None:
    require_api_key(args.env_file)
    paths = build_batch_files(args)
    client = OpenAI()

    with paths["batch_input"].open("rb") as file:
        uploaded_file = client.files.create(file=file, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint=ENDPOINT,
        completion_window="24h",
        metadata={
            "task": "mrcr-mini",
            "split": args.split,
            "mode": args.mode,
            "model": args.model,
        },
    )
    batch_data = batch.model_dump(mode="json")
    write_json(paths["batch"], batch_data)
    print(f"Uploaded file: {uploaded_file.id}")
    print(f"Created batch: {batch.id}")
    print(f"Wrote batch metadata: {paths['batch']}")


def download_file(client: OpenAI, file_id: str, output_path: Path) -> None:
    response = client.files.content(file_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.read())


def retrieve_batch(args: argparse.Namespace) -> None:
    require_api_key(args.env_file)
    paths = output_paths(args)
    if not paths["batch"].exists():
        raise SystemExit(f"Missing batch metadata: {paths['batch']}. Run submit first.")

    batch_id = read_json(paths["batch"])["id"]
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    write_json(paths["batch"], batch.model_dump(mode="json"))
    print(f"Batch status: {batch.status}")

    if batch.output_file_id:
        download_file(client, batch.output_file_id, paths["batch_output"])
        print(f"Wrote output: {paths['batch_output']}")
    else:
        print("No output_file_id yet.")

    if batch.error_file_id:
        download_file(client, batch.error_file_id, paths["batch_errors"])
        print(f"Wrote errors: {paths['batch_errors']}")
    else:
        print("No error_file_id.")


def main() -> None:
    args = parse_args()
    if args.command == "create":
        build_batch_files(args)
    elif args.command == "submit":
        submit_batch(args)
    elif args.command == "retrieve":
        retrieve_batch(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
