"""Submit, retrieve, and download MRCR mini Batch API jobs.

Examples:
    python batch_mini_mrcr.py submit --split val --mode text --model gpt-5.4
    python batch_mini_mrcr.py retrieve --split val --mode text --model gpt-5.4
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
MAX_REQUESTS_PER_BATCH = 15


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
    parser.add_argument("--recent-turns", type=int, default=20)
    parser.add_argument("--highres-related-top-k", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--only-parts",
        default=None,
        help="Comma-separated 1-based part numbers to create/submit again, e.g. 2,3.",
    )


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


def paths_for_stem(args: argparse.Namespace, stem: str) -> dict[str, Path]:
    return {
        "batch_input": args.results_dir / f"{stem}_batch_input.jsonl",
        "manifest": args.results_dir / f"{stem}_manifest.jsonl",
        "batch": args.results_dir / f"{stem}_batch.json",
        "batch_output": args.results_dir / f"{stem}_batch_output.jsonl",
        "batch_errors": args.results_dir / f"{stem}_batch_errors.jsonl",
    }


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    return paths_for_stem(args, stem_for(args))


def part_output_paths(args: argparse.Namespace, part_index: int) -> dict[str, Path]:
    return paths_for_stem(args, f"{stem_for(args)}_part{part_index:02d}")


def selected_parts(args: argparse.Namespace) -> set[int] | None:
    if not args.only_parts:
        return None

    parts: set[int] = set()
    for raw_part in args.only_parts.split(","):
        raw_part = raw_part.strip()
        if not raw_part:
            continue
        try:
            part = int(raw_part)
        except ValueError as exc:
            raise SystemExit(f"Invalid --only-parts value: {args.only_parts}") from exc
        if part <= 0:
            raise SystemExit("--only-parts uses 1-based positive part numbers.")
        parts.add(part)

    if not parts:
        raise SystemExit(f"Invalid --only-parts value: {args.only_parts}")
    return parts


def cleanup_previous_generated_files(args: argparse.Namespace) -> None:
    stem = stem_for(args)
    for path in args.results_dir.glob(f"{stem}_part*"):
        if path.is_file():
            path.unlink()

    paths = output_paths(args)
    for key in ("batch_input", "batch_output", "batch_errors"):
        path = paths[key]
        if path.exists():
            path.unlink()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_messages(
    row: dict[str, Any],
    *,
    mode: str,
    recent_turns: int,
    highres_related_top_k: int,
    renderer: ConversationImageRenderer | None,
) -> list[dict[str, Any]]:
    if mode == "text":
        return [dict(message) for message in row["messages"]]

    return build_image_history_messages(
        row["messages"],
        recent_turns=recent_turns,
        renderer=renderer,
        highres_related_top_k=highres_related_top_k,
    )


def build_request_rows(
    args: argparse.Namespace,
    indexed_rows: list[tuple[int, dict[str, Any]]],
    renderer: ConversationImageRenderer | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for index, row in indexed_rows:
        source_index = row["mrcr_source_row_index"]
        custom_id = f"mrcr-{args.split}-{args.mode}-row-{source_index}"
        body: dict[str, Any] = {
            "model": args.model,
            "messages": build_messages(
                row,
                mode=args.mode,
                recent_turns=args.recent_turns,
                highres_related_top_k=args.highres_related_top_k,
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

    return requests, manifest


def split_chunks(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def build_batch_files(args: argparse.Namespace) -> dict[str, Path]:
    selected = selected_parts(args)
    if selected is None:
        cleanup_previous_generated_files(args)

    rows = read_jsonl(args.data_dir / f"{args.split}.jsonl")
    indexed_rows = list(enumerate(rows))
    renderer = ConversationImageRenderer() if args.mode == "image-history" else None
    paths = output_paths(args)

    if len(indexed_rows) <= MAX_REQUESTS_PER_BATCH:
        if selected is not None:
            raise SystemExit("--only-parts can only be used when the dataset is split into multiple parts.")
        requests, manifest = build_request_rows(args, indexed_rows, renderer)
        write_jsonl(paths["batch_input"], requests)
        write_jsonl(paths["manifest"], manifest)
        print(f"Wrote {len(requests)} requests: {paths['batch_input']}")
        print(f"Wrote manifest: {paths['manifest']}")
        return paths

    all_manifest: list[dict[str, Any]] = []
    chunks = split_chunks(indexed_rows, MAX_REQUESTS_PER_BATCH)
    unknown_parts = selected - set(range(1, len(chunks) + 1)) if selected is not None else set()
    if unknown_parts:
        unknown = ", ".join(str(part) for part in sorted(unknown_parts))
        raise SystemExit(f"Unknown part(s): {unknown}. This run has {len(chunks)} parts.")

    for part_index, chunk in enumerate(chunks, 1):
        if selected is not None and part_index not in selected:
            continue

        requests, manifest = build_request_rows(args, chunk, renderer)
        part_paths = part_output_paths(args, part_index)
        write_jsonl(part_paths["batch_input"], requests)
        write_jsonl(part_paths["manifest"], manifest)
        all_manifest.extend(manifest)
        print(
            f"Wrote part {part_index}/{len(chunks)} rows "
            f"{chunk[0][0] + 1}-{chunk[-1][0] + 1}: {part_paths['batch_input']}"
        )

    if selected is None or not paths["manifest"].exists():
        full_manifest: list[dict[str, Any]] = []
        for chunk in chunks:
            _, manifest = build_request_rows(args, chunk, renderer)
            full_manifest.extend(manifest)
        write_jsonl(paths["manifest"], full_manifest)
        print(f"Wrote combined manifest: {paths['manifest']}")

    return paths


def submit_batch(args: argparse.Namespace) -> None:
    require_api_key(args.env_file)
    build_batch_files(args)
    paths = output_paths(args)
    client = OpenAI()

    selected = selected_parts(args)
    part_inputs = sorted(args.results_dir.glob(f"{stem_for(args)}_part*_batch_input.jsonl"))
    if selected is not None:
        part_inputs = [
            path
            for path in part_inputs
            if int(path.name.split("_part", 1)[1].split("_", 1)[0]) in selected
        ]

    if not part_inputs:
        part_inputs = [paths["batch_input"]]

    existing_metadata = read_json(paths["batch"]) if paths["batch"].exists() else {}
    existing_parts_by_number: dict[int, dict[str, Any]] = {}
    if existing_metadata.get("multi_batch"):
        existing_parts_by_number = {
            int(part["part"]): part for part in existing_metadata.get("parts", [])
        }

    batch_parts: list[dict[str, Any]] = []
    total_parts = len(split_chunks(read_jsonl(args.data_dir / f"{args.split}.jsonl"), MAX_REQUESTS_PER_BATCH))

    for batch_input_path in part_inputs:
        if "_part" in batch_input_path.name:
            part_number = int(batch_input_path.name.split("_part", 1)[1].split("_", 1)[0])
            part_paths = part_output_paths(args, part_number)
        else:
            part_number = 1
            part_paths = paths

        with batch_input_path.open("rb") as file:
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
                "part": str(part_number),
                "parts": str(total_parts),
            },
        )
        batch_data = batch.model_dump(mode="json")
        write_json(part_paths["batch"], batch_data)
        batch_parts.append(
            {
                "part": part_number,
                "batch_path": str(part_paths["batch"]),
                "batch_input_path": str(batch_input_path),
                "batch_id": batch.id,
                "uploaded_file_id": uploaded_file.id,
            }
        )
        print(f"Part {part_number}/{total_parts} uploaded file: {uploaded_file.id}")
        print(f"Part {part_number}/{total_parts} created batch: {batch.id}")

    if part_inputs and "_part" in part_inputs[0].name:
        for part in batch_parts:
            existing_parts_by_number[int(part["part"])] = part
        combined_parts = [
            existing_parts_by_number[part_number]
            for part_number in sorted(existing_parts_by_number)
        ]
        write_json(
            paths["batch"],
            {
                "multi_batch": True,
                "parts": combined_parts,
                "max_requests_per_batch": MAX_REQUESTS_PER_BATCH,
            },
        )
        print(f"Wrote combined batch metadata: {paths['batch']}")
    else:
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

    batch_metadata = read_json(paths["batch"])
    client = OpenAI()

    if not batch_metadata.get("multi_batch"):
        batch_id = batch_metadata["id"]
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
        return

    combined_output_rows: list[dict[str, Any]] = []
    combined_error_rows: list[dict[str, Any]] = []
    updated_parts: list[dict[str, Any]] = []

    for part in batch_metadata["parts"]:
        part_number = int(part["part"])
        part_paths = part_output_paths(args, part_number)
        batch = client.batches.retrieve(part["batch_id"])
        write_json(part_paths["batch"], batch.model_dump(mode="json"))
        updated_part = dict(part)
        updated_part["status"] = batch.status
        updated_parts.append(updated_part)
        print(f"Part {part_number}/{len(batch_metadata['parts'])} status: {batch.status}")

        if batch.output_file_id:
            download_file(client, batch.output_file_id, part_paths["batch_output"])
            combined_output_rows.extend(read_jsonl(part_paths["batch_output"]))
            print(f"Wrote output: {part_paths['batch_output']}")
        else:
            print(f"Part {part_number} has no output_file_id yet.")

        if batch.error_file_id:
            download_file(client, batch.error_file_id, part_paths["batch_errors"])
            combined_error_rows.extend(read_jsonl(part_paths["batch_errors"]))
            print(f"Wrote errors: {part_paths['batch_errors']}")

    batch_metadata["parts"] = updated_parts
    write_json(paths["batch"], batch_metadata)

    if combined_output_rows:
        write_jsonl(paths["batch_output"], combined_output_rows)
        print(f"Wrote combined output: {paths['batch_output']}")
    if combined_error_rows:
        write_jsonl(paths["batch_errors"], combined_error_rows)
        print(f"Wrote combined errors: {paths['batch_errors']}")


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
