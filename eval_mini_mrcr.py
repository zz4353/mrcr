"""Run MRCR mini evaluation with optional image-history conversion.

Examples:
    python eval_mini_mrcr.py --split val --model gpt-4.1
    python eval_mini_mrcr.py --split test --mode image-history --model gpt-4.1
    python eval_mini_mrcr.py --grade-only runs/mrcr_val_text.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import OpenAI

from mrcr_image_history import ConversationImageRenderer, build_image_history_messages


DEFAULT_DATA_DIR = Path("data") / "mini"
DEFAULT_OUTPUT_DIR = Path("runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an OpenAI-compatible model on MRCR mini.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("text", "image-history"), default="text")
    parser.add_argument("--recent-turns", type=int, default=3)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument(
        "--grade-only",
        type=Path,
        default=None,
        help="Skip model calls and summarize an existing JSONL output file.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_response(answer: str, random_string_to_prepend: str, response: str) -> float:
    """MRCR scoring: require the prefix, then compare generated and gold answer bodies."""
    if not response.startswith(random_string_to_prepend):
        return 0.0

    sampled_answer = response.removeprefix(random_string_to_prepend)
    expected_answer = answer.removeprefix(random_string_to_prepend)
    return SequenceMatcher(None, sampled_answer, expected_answer).ratio()


def default_output_path(args: argparse.Namespace) -> Path:
    model_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in args.model)
    return DEFAULT_OUTPUT_DIR / f"mrcr_{args.split}_{args.mode}_{model_name}.jsonl"


def completed_source_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    for row in read_jsonl(path):
        source_index = row.get("mrcr_source_row_index")
        if isinstance(source_index, int):
            done.add(source_index)
    return done


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


def call_model(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int | None,
) -> str:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens

    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content
    return content or ""


def summarize(path: Path) -> None:
    if not path.exists():
        print(f"No file found: {path}")
        return

    rows = read_jsonl(path)
    if not rows:
        print(f"No rows found in {path}")
        return

    scores = [float(row["score"]) for row in rows]
    print(f"file: {path}")
    print(f"rows: {len(rows)}")
    print(f"mean_score: {sum(scores) / len(scores):.4f}")

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        for field in ("n_needles", "token_bin_o200k", "target_position_bin"):
            value = row.get(field)
            if value is not None:
                groups[(field, str(value))].append(float(row["score"]))

    for (field, value), group_scores in sorted(groups.items()):
        mean_score = sum(group_scores) / len(group_scores)
        print(f"{field}={value}: n={len(group_scores)} mean={mean_score:.4f}")


def run_eval(args: argparse.Namespace) -> Path:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running model evaluation.")

    data_path = args.data_dir / f"{args.split}.jsonl"
    rows = read_jsonl(data_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = args.output or default_output_path(args)
    done = completed_source_indices(output_path) if args.resume else set()
    renderer = ConversationImageRenderer() if args.mode == "image-history" else None
    client = OpenAI()

    for offset, row in enumerate(rows, 1):
        source_index = row["mrcr_source_row_index"]
        if source_index in done:
            continue

        messages = build_messages(
            row,
            mode=args.mode,
            recent_turns=args.recent_turns,
            renderer=renderer,
        )
        response = call_model(
            client,
            model=args.model,
            messages=messages,
            max_completion_tokens=args.max_completion_tokens,
        )
        score = score_response(row["answer"], row["random_string_to_prepend"], response)

        result = {
            "split": args.split,
            "mode": args.mode,
            "model": args.model,
            "mrcr_source_row_index": source_index,
            "n_needles": row.get("n_needles"),
            "n_tokens_o200k": row.get("n_tokens_o200k"),
            "token_bin_o200k": row.get("token_bin_o200k"),
            "target_position_bin": row.get("target_position_bin"),
            "random_string_to_prepend": row["random_string_to_prepend"],
            "answer": row["answer"],
            "response": response,
            "score": score,
        }
        append_jsonl(output_path, result)
        print(f"[{offset}/{len(rows)}] row={source_index} score={score:.4f}")

    return output_path


def main() -> None:
    args = parse_args()
    if args.grade_only is not None:
        summarize(args.grade_only)
        return

    output_path = run_eval(args)
    print()
    summarize(output_path)


if __name__ == "__main__":
    main()
