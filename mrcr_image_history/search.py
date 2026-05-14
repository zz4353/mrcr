"""Fast lexical search for related conversation turns.

This module intentionally does not persist an index. It derives lightweight
signals from the supplied conversation on each call and returns scored turns.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"(?u)[^\W_]+(?:[_\-.][^\W_]+)*")


@dataclass(frozen=True)
class ConversationTurn:
    """A user-started turn and its following assistant/tool messages."""

    turn_index: int
    start_message_index: int
    end_message_index: int
    messages: list[dict[str, Any]]


@dataclass(frozen=True)
class RelatedTurn:
    """A scored related turn returned by search_related_turns."""

    score: float
    turn: ConversationTurn
    reasons: tuple[str, ...]


def split_into_turns(messages: list[dict[str, Any]]) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    current: list[dict[str, Any]] = []
    start_index = 0
    turn_index = 0

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        if message.get("role") == "user":
            if current:
                turns.append(
                    ConversationTurn(
                        turn_index=turn_index,
                        start_message_index=start_index,
                        end_message_index=message_index - 1,
                        messages=current,
                    )
                )
            turn_index += 1
            start_index = message_index
            current = [message]
        elif current:
            current.append(message)

    if current:
        turns.append(
            ConversationTurn(
                turn_index=turn_index,
                start_message_index=start_index,
                end_message_index=len(messages) - 1,
                messages=current,
            )
        )

    return turns


def search_related_turns(
    conversation: list[dict[str, Any]],
    user_input: str,
    *,
    top_k: int = 5,
    exclude_last_user_turn: bool = True,
    skip_recent_turns: int = 0,
) -> list[RelatedTurn]:
    """Return turns related to user_input using cheap lexical scoring."""

    messages = [message for message in conversation if isinstance(message, dict)]
    if exclude_last_user_turn and messages and messages[-1].get("role") == "user":
        messages = messages[:-1]

    query = normalize_text(user_input)
    query_tokens = tokenize(query)
    query_bigrams = bigrams(query_tokens)

    candidate_turns = skip_recent(split_into_turns(messages), skip_recent_turns)
    token_weights = inverse_document_frequency(candidate_turns)

    scored: list[RelatedTurn] = []
    for turn in candidate_turns:
        user_text = normalize_text(join_role_text(turn.messages, role="user"))
        other_text = normalize_text(join_role_text(turn.messages, role_not="user"))
        user_tokens = tokenize(user_text)
        other_tokens = tokenize(other_text)

        score = 0.0
        reasons: list[str] = []

        if query and query in user_text:
            score += 8.0
            reasons.append("exact_query_in_user")

        if query and query in other_text:
            score += 2.5
            reasons.append("exact_query_in_assistant")

        user_overlap = weighted_token_overlap_score(query_tokens, user_tokens, token_weights)
        if user_overlap:
            score += 4.0 * user_overlap
            reasons.append(f"user_overlap:{user_overlap:.2f}")

        phrase_overlap = bigram_overlap_score(query_bigrams, user_tokens)
        if phrase_overlap:
            score += 2.0 * phrase_overlap
            reasons.append(f"bigram_overlap:{phrase_overlap:.2f}")

        other_overlap = weighted_token_overlap_score(query_tokens, other_tokens, token_weights)
        if other_overlap:
            score += 1.2 * other_overlap
            reasons.append(f"assistant_overlap:{other_overlap:.2f}")

        if score > 0:
            scored.append(RelatedTurn(score=score, turn=turn, reasons=tuple(reasons)))

    scored.sort(key=lambda item: (-item.score, item.turn.turn_index))
    return scored[:top_k]


def skip_recent(turns: list[ConversationTurn], recent_turns: int) -> list[ConversationTurn]:
    if recent_turns <= 0:
        return turns
    if recent_turns >= len(turns):
        return []
    return turns[:-recent_turns]


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def inverse_document_frequency(turns: list[ConversationTurn]) -> dict[str, float]:
    document_frequency: dict[str, int] = {}
    for turn in turns:
        text = normalize_text(join_role_text(turn.messages))
        for token in set(tokenize(text)):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    document_count = max(1, len(turns))
    return {
        token: 1.0 + math.log((document_count + 1) / (count + 1))
        for token, count in document_frequency.items()
    }


def weighted_token_overlap_score(
    query_tokens: list[str],
    candidate_tokens: list[str],
    token_weights: dict[str, float],
) -> float:
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    candidate_set = set(candidate_tokens)
    denominator = sum(token_weights.get(token, 1.0) for token in query_set)
    if denominator <= 0:
        return 0.0
    numerator = sum(token_weights.get(token, 1.0) for token in query_set & candidate_set)
    candidate_denominator = sum(token_weights.get(token, 1.0) for token in candidate_set)
    if candidate_denominator <= 0:
        return 0.0
    recall = numerator / denominator
    precision = numerator / candidate_denominator
    return math.sqrt(recall * precision)


def bigram_overlap_score(
    query_bigrams: set[tuple[str, str]],
    candidate_tokens: list[str],
) -> float:
    if not query_bigrams:
        return 0.0
    candidate_bigrams = bigrams(candidate_tokens)
    if not candidate_bigrams:
        return 0.0
    overlap = len(query_bigrams & candidate_bigrams)
    return overlap / math.sqrt(len(query_bigrams) * len(candidate_bigrams))


def join_role_text(
    messages: list[dict[str, Any]],
    *,
    role: str | None = None,
    role_not: str | None = None,
) -> str:
    parts: list[str] = []
    for message in messages:
        message_role = message.get("role")
        if role is not None and message_role != role:
            continue
        if role_not is not None and message_role == role_not:
            continue
        parts.append(content_to_text(message.get("content", "")))
    return "\n".join(parts)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _preview(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
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
    parser.add_argument(
        "--skip-recent-turns",
        type=int,
        default=0,
        help="Exclude this many most recent completed turns from search candidates.",
    )
    args = parser.parse_args()

    row = _load_first_jsonl(args.jsonl)
    messages = row["messages"]
    active_user = content_to_text(messages[-1].get("content", ""))
    related = search_related_turns(
        messages,
        active_user,
        top_k=args.top_k,
        skip_recent_turns=args.skip_recent_turns,
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
