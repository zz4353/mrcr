"""BM25-based search for related conversation turns.

This builds a small in-memory BM25 index from the supplied conversation on each
call. It does not persist conversation text or write an index to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Plus
except ImportError as exc:  # pragma: no cover - depends on local environment
    BM25Plus = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .search import (
    ConversationTurn,
    RelatedTurn,
    content_to_text,
    join_role_text,
    normalize_text,
    skip_recent,
    split_into_turns,
    tokenize,
)


def search_related_turns(
    conversation: list[dict[str, Any]],
    user_input: str,
    *,
    top_k: int = 5,
    exclude_last_user_turn: bool = True,
    skip_recent_turns: int = 0,
    user_weight: float = 1.0,
    assistant_weight: float = 0.35,
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 1.0,
) -> list[RelatedTurn]:
    """Return related turns using BM25Plus from rank-bm25."""

    if BM25Plus is None:
        raise RuntimeError("Missing dependency: install rank-bm25 to use search2.py") from _IMPORT_ERROR

    messages = [message for message in conversation if isinstance(message, dict)]
    if exclude_last_user_turn and messages and messages[-1].get("role") == "user":
        messages = messages[:-1]

    query_tokens = tokenize(normalize_text(user_input))
    if not query_tokens:
        return []

    candidate_turns = skip_recent(split_into_turns(messages), skip_recent_turns)
    if not candidate_turns:
        return []

    user_docs = [turn_tokens(turn, role="user") for turn in candidate_turns]
    assistant_docs = [turn_tokens(turn, role_not="user") for turn in candidate_turns]

    user_scores = bm25_scores(user_docs, query_tokens, k1=k1, b=b, delta=delta)
    assistant_scores = bm25_scores(assistant_docs, query_tokens, k1=k1, b=b, delta=delta)

    scored: list[RelatedTurn] = []
    for index, turn in enumerate(candidate_turns):
        user_score = float(user_scores[index])
        assistant_score = float(assistant_scores[index])
        score = user_weight * user_score + assistant_weight * assistant_score
        if score <= 0:
            continue

        reasons = []
        if user_score > 0:
            reasons.append(f"user_bm25:{user_score:.3f}")
        if assistant_score > 0:
            reasons.append(f"assistant_bm25:{assistant_score:.3f}")

        scored.append(RelatedTurn(score=score, turn=turn, reasons=tuple(reasons)))

    scored.sort(key=lambda item: (-item.score, item.turn.turn_index))
    return scored[:top_k]


def turn_tokens(
    turn: ConversationTurn,
    *,
    role: str | None = None,
    role_not: str | None = None,
) -> list[str]:
    return tokenize(normalize_text(join_role_text(turn.messages, role=role, role_not=role_not)))


def bm25_scores(
    tokenized_docs: list[list[str]],
    query_tokens: list[str],
    *,
    k1: float,
    b: float,
    delta: float,
):
    if not tokenized_docs or not any(tokenized_docs):
        return [0.0 for _ in tokenized_docs]
    bm25 = BM25Plus(tokenized_docs, k1=k1, b=b, delta=delta)
    return bm25.get_scores(query_tokens)


def _preview(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _load_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/mini/val.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-recent-turns", type=int, default=0)
    parser.add_argument("--user-weight", type=float, default=1.0)
    parser.add_argument("--assistant-weight", type=float, default=0.35)
    args = parser.parse_args()

    row = _load_first_jsonl(args.jsonl)
    messages = row["messages"]
    active_user = content_to_text(messages[-1].get("content", ""))
    related = search_related_turns(
        messages,
        active_user,
        top_k=args.top_k,
        skip_recent_turns=args.skip_recent_turns,
        user_weight=args.user_weight,
        assistant_weight=args.assistant_weight,
    )

    print(f"user_input: {active_user}")
    print(f"expected desired_msg_index: {row.get('desired_msg_index')}")
    print(f"answer_prefix: {_preview(row.get('answer', ''), 160)}")
    print()

    for rank, item in enumerate(related, 1):
        user_text = join_role_text(item.turn.messages, role="user")
        assistant_text = join_role_text(item.turn.messages, role_not="user")
        print(
            f"#{rank} score={item.score:.3f} "
            f"turn={item.turn.turn_index} "
            f"messages={item.turn.start_message_index + 1}-{item.turn.end_message_index + 1} "
            f"reasons={', '.join(item.reasons)}"
        )
        print(f"user: {_preview(user_text)}")
        print(f"assistant: {_preview(assistant_text, 320)}")
        print()


if __name__ == "__main__":
    main()
