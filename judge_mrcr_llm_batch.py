r"""LLM-judge MRCR batch outputs independently.

Examples:
    python judge_mrcr_llm_batch.py submit --manifest results\mrcr_val_text_gpt-5.4_manifest.jsonl --batch-output results\mrcr_val_text_gpt-5.4_batch_output.jsonl
    python judge_mrcr_llm_batch.py retrieve --manifest results\mrcr_val_text_gpt-5.4_manifest.jsonl --batch-output results\mrcr_val_text_gpt-5.4_batch_output.jsonl
    python judge_mrcr_llm_batch.py report --manifest results\mrcr_val_text_gpt-5.4_manifest.jsonl --batch-output results\mrcr_val_text_gpt-5.4_batch_output.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_ENV_FILE = Path(".env")
DEFAULT_JUDGE_MODEL = "gpt-5.4"
ENDPOINT = "/v1/chat/completions"

SYSTEM_PROMPT = """You are a fair judge for a long-context memory benchmark.

Judge whether model_response remembered the same prior assistant answer as expected_answer.

This is not an exact transcription test. Accept paraphrases and wording changes if the response is clearly the same item and keeps the main topic, purpose, structure, and key details.

Mark correct=true for:
- Same item with similar meaning, even if wording, formatting, emojis, markdown, or closing lines differ.
- Same article/email/post/poem/scene with minor omissions or small rewritten parts.

Mark correct=false for:
- wrong_needle: a different prior item, even if the topic is similar.
- partial: only a small part is remembered, or major sections are missing.
- extra_content: new content changes the purpose or identity of the item.
- paraphrase_too_loose: rewritten so heavily that it no longer feels like the same item.
- missing_prefix: required prefix is missing or not at the beginning.
- refusal, empty, or other clear failure.

When uncertain between correct and false, prefer correct if the same remembered item is recognizable.

Return only JSON:
{
  "correct": boolean,
  "confidence": number from 0 to 1,
  "error_type": "correct" | "wrong_needle" | "partial" | "extra_content" | "paraphrase_too_loose" | "missing_prefix" | "refusal" | "empty" | "other",
  "reason": short string
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM judge MRCR batch outputs with Batch API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("create", "submit", "retrieve", "report"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--batch-output", type=Path, required=True)
        subparser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
        subparser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)

    return parser.parse_args()


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def safe_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)


def stem_from_batch_output(path: Path) -> str:
    name = path.name
    if name.endswith("_batch_output.jsonl"):
        return name[: -len("_batch_output.jsonl")]
    return path.stem


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    stem = stem_from_batch_output(args.batch_output)
    base = args.batch_output.parent / f"{stem}_llm_judge_{safe_model_name(args.judge_model)}"
    return {
        "batch_input": Path(f"{base}_batch_input.jsonl"),
        "batch": Path(f"{base}_batch.json"),
        "batch_output": Path(f"{base}_batch_output.jsonl"),
        "batch_errors": Path(f"{base}_batch_errors.jsonl"),
        "report": Path(f"{base}_report.json"),
    }


def extract_response_text(batch_row: dict[str, Any] | None) -> str:
    if not batch_row:
        return ""
    response = batch_row.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def load_final_user_messages(split: str | None) -> dict[int, str]:
    if not split:
        return {}

    data_path = Path("data") / "mini" / f"{split}.jsonl"
    if not data_path.exists():
        return {}

    mapping: dict[int, str] = {}
    with data_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            source_row = json.loads(line)
            messages = source_row.get("messages") or []
            final_user = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    final_user = str(message.get("content", ""))
                    break
            source_index = source_row.get("mrcr_source_row_index")
            if isinstance(source_index, int):
                mapping[source_index] = final_user
    return mapping


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_rows = read_jsonl(args.manifest)
    batch_rows = {row["custom_id"]: row for row in read_jsonl(args.batch_output)}
    split = manifest_rows[0].get("split") if manifest_rows else None
    final_user_messages = load_final_user_messages(split)
    records: list[dict[str, Any]] = []

    for row in manifest_rows:
        response_text = extract_response_text(batch_rows.get(row["custom_id"]))
        records.append(
            {
                "custom_id": row["custom_id"],
                "split": row.get("split"),
                "mode": row.get("mode"),
                "model": row.get("model"),
                "row_index_in_split": row.get("row_index_in_split"),
                "mrcr_source_row_index": row["mrcr_source_row_index"],
                "n_needles": row.get("n_needles"),
                "n_tokens_o200k": row.get("n_tokens_o200k"),
                "token_bin_o200k": row.get("token_bin_o200k"),
                "target_position_bin": row.get("target_position_bin"),
                "random_string_to_prepend": row.get("random_string_to_prepend", ""),
                "final_user_message": final_user_messages.get(row["mrcr_source_row_index"], ""),
                "expected_answer": row.get("answer", ""),
                "model_response": response_text,
                "prefix_ok": response_text.startswith(row.get("random_string_to_prepend", "")),
            }
        )

    return records


def build_user_prompt(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "final_user_message": row.get("final_user_message", ""),
            "expected_answer": row.get("expected_answer", ""),
            "model_response": row.get("model_response", ""),
        },
        ensure_ascii=False,
    )


def create_batch_file(args: argparse.Namespace) -> dict[str, Path]:
    records = build_records(args)
    requests: list[dict[str, Any]] = []

    for row in records:
        requests.append(
            {
                "custom_id": f"judge-{row['custom_id']}",
                "method": "POST",
                "url": ENDPOINT,
                "body": {
                    "model": args.judge_model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(row)},
                    ],
                },
            }
        )

    paths = output_paths(args)
    write_jsonl(paths["batch_input"], requests)
    print(f"Wrote {len(requests)} judge requests: {paths['batch_input']}")
    return paths


def submit_batch(args: argparse.Namespace) -> None:
    require_api_key(args.env_file)
    paths = create_batch_file(args)
    client = OpenAI()

    with paths["batch_input"].open("rb") as file:
        uploaded_file = client.files.create(file=file, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint=ENDPOINT,
        completion_window="24h",
        metadata={"task": "mrcr-llm-judge", "judge_model": args.judge_model},
    )
    write_json(paths["batch"], batch.model_dump(mode="json"))
    print(f"Created judge batch: {batch.id}")
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


def extract_judge_json(batch_row: dict[str, Any] | None) -> dict[str, Any]:
    if not batch_row:
        return {"correct": False, "confidence": 0, "error_type": "empty", "reason": "Missing judge output"}

    response = batch_row.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return {"correct": False, "confidence": 0, "error_type": "empty", "reason": "No judge output"}

    content = (choices[0].get("message") or {}).get("content") or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "correct": False,
            "confidence": 0,
            "error_type": "other",
            "reason": f"Invalid judge JSON: {content[:200]}",
        }


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [bool(row["llm_correct"]) for row in details]
    summary: dict[str, Any] = {
        "n": len(details),
        "llm_accuracy": sum(correct) / len(correct) if correct else 0.0,
        "llm_correct": sum(correct),
        "llm_incorrect": len(correct) - sum(correct),
    }

    error_types: dict[str, int] = defaultdict(int)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        error_types[str(row.get("llm_error_type"))] += 1
        for field in ("n_needles", "token_bin_o200k", "target_position_bin"):
            value = row.get(field)
            if value is not None:
                groups[(field, str(value))].append(row)

    summary["error_types"] = dict(sorted(error_types.items()))
    summary["by_group"] = {}
    for (field, value), rows in sorted(groups.items()):
        group_correct = [bool(row["llm_correct"]) for row in rows]
        summary["by_group"][f"{field}={value}"] = {
            "n": len(rows),
            "llm_accuracy": sum(group_correct) / len(group_correct),
            "llm_correct": sum(group_correct),
        }
    return summary


def build_report(args: argparse.Namespace) -> None:
    paths = output_paths(args)
    if not paths["batch_output"].exists():
        raise SystemExit(f"Missing judge output: {paths['batch_output']}. Run retrieve first.")

    records = {f"judge-{row['custom_id']}": row for row in build_records(args)}
    judge_rows = {row["custom_id"]: row for row in read_jsonl(paths["batch_output"])}
    details: list[dict[str, Any]] = []

    for judge_custom_id, original in sorted(
        records.items(),
        key=lambda item: item[1]["row_index_in_split"],
    ):
        judge = extract_judge_json(judge_rows.get(judge_custom_id))
        details.append(
            {
                "custom_id": original["custom_id"],
                "mrcr_source_row_index": original["mrcr_source_row_index"],
                "row_index_in_split": original["row_index_in_split"],
                "split": original["split"],
                "mode": original["mode"],
                "model": original["model"],
                "n_needles": original.get("n_needles"),
                "n_tokens_o200k": original.get("n_tokens_o200k"),
                "token_bin_o200k": original.get("token_bin_o200k"),
                "target_position_bin": original.get("target_position_bin"),
                "prefix_ok": original["prefix_ok"],
                "llm_correct": bool(judge.get("correct")),
                "llm_confidence": judge.get("confidence"),
                "llm_error_type": judge.get("error_type"),
                "llm_reason": judge.get("reason"),
            }
        )

    report = {
        "manifest": str(args.manifest),
        "batch_output": str(args.batch_output),
        "judge_model": args.judge_model,
        "summary": summarize(details),
        "details": details,
    }
    write_json(paths["report"], report)
    print(f"Wrote judge report: {paths['report']}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "create":
        create_batch_file(args)
    elif args.command == "submit":
        submit_batch(args)
    elif args.command == "retrieve":
        retrieve_batch(args)
    elif args.command == "report":
        build_report(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
