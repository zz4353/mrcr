"""Grade MRCR mini results downloaded from the OpenAI Batch API.

Example:
    python grade_mrcr_batch.py ^
      --manifest results/mrcr_val_text_gpt-5_manifest.jsonl ^
      --batch-output results/mrcr_val_text_gpt-5_batch_output.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade MRCR Batch API output.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.99,
        help="Score at or above this threshold is counted as correct. Default: 0.99",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_response(answer: str, random_string_to_prepend: str, response: str) -> float:
    if not response.startswith(random_string_to_prepend):
        return 0.0

    sampled_answer = response.removeprefix(random_string_to_prepend)
    expected_answer = answer.removeprefix(random_string_to_prepend)
    return SequenceMatcher(None, sampled_answer, expected_answer).ratio()


def extract_response_text(batch_row: dict[str, Any]) -> str:
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


def default_output_path(batch_output_path: Path) -> Path:
    name = batch_output_path.name
    if name.endswith("_batch_output.jsonl"):
        stem = name[: -len("_batch_output.jsonl")]
    else:
        stem = batch_output_path.stem
    return DEFAULT_RESULTS_DIR / f"{stem}_graded.json"


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in details]
    correct = [row["correct"] for row in details]
    summary: dict[str, Any] = {
        "n": len(details),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "n_correct": sum(correct),
        "n_incorrect": len(correct) - sum(correct),
    }

    by_group: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        for field in ("n_needles", "token_bin_o200k", "target_position_bin"):
            value = row.get(field)
            if value is not None:
                groups[(field, str(value))].append(row)

    for (field, value), rows in sorted(groups.items()):
        group_scores = [row["score"] for row in rows]
        group_correct = [row["correct"] for row in rows]
        by_group[f"{field}={value}"] = {
            "n": len(rows),
            "mean_score": sum(group_scores) / len(group_scores),
            "accuracy": sum(group_correct) / len(group_correct),
            "n_correct": sum(group_correct),
        }

    summary["by_group"] = by_group
    return summary


def main() -> None:
    args = parse_args()
    manifest = {row["custom_id"]: row for row in read_jsonl(args.manifest)}
    batch_rows = {row["custom_id"]: row for row in read_jsonl(args.batch_output)}

    details: list[dict[str, Any]] = []
    missing_outputs = sorted(set(manifest) - set(batch_rows))
    unexpected_outputs = sorted(set(batch_rows) - set(manifest))

    for custom_id, expected in sorted(manifest.items(), key=lambda item: item[1]["row_index_in_split"]):
        batch_row = batch_rows.get(custom_id)
        response_text = extract_response_text(batch_row) if batch_row else ""
        error = None if batch_row is None else batch_row.get("error")
        status_code = None
        finish_reason = None
        usage = None

        if batch_row:
            response = batch_row.get("response") or {}
            status_code = response.get("status_code")
            body = response.get("body") or {}
            choices = body.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason")
            usage = body.get("usage")

        prefix_ok = response_text.startswith(expected["random_string_to_prepend"])
        score = score_response(
            expected["answer"],
            expected["random_string_to_prepend"],
            response_text,
        )
        correct = score >= args.correct_threshold

        details.append(
            {
                "custom_id": custom_id,
                "mrcr_source_row_index": expected["mrcr_source_row_index"],
                "row_index_in_split": expected["row_index_in_split"],
                "split": expected["split"],
                "mode": expected["mode"],
                "model": expected["model"],
                "n_needles": expected.get("n_needles"),
                "n_tokens_o200k": expected.get("n_tokens_o200k"),
                "token_bin_o200k": expected.get("token_bin_o200k"),
                "target_position_bin": expected.get("target_position_bin"),
                "status_code": status_code,
                "finish_reason": finish_reason,
                "prefix_ok": prefix_ok,
                "score": score,
                "correct": correct,
                "error": error,
                "usage": usage,
                "expected_answer": expected["answer"],
                "model_response": response_text,
            }
        )

    result = {
        "manifest": str(args.manifest),
        "batch_output": str(args.batch_output),
        "correct_threshold": args.correct_threshold,
        "summary": summarize(details),
        "missing_outputs": missing_outputs,
        "unexpected_outputs": unexpected_outputs,
        "details": details,
    }

    output_path = args.output or default_output_path(args.batch_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote graded results: {output_path}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
