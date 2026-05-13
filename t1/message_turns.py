"""Helpers for reasoning about blocked conversation turns."""

from __future__ import annotations

from typing import Any


def is_blocked_message(message: Any) -> bool:
    return isinstance(message, dict) and message.get("blocked") is True


def indexed_messages_by_turn(messages: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    """Return messages with their array index and 1-based user turn index."""

    rows: list[tuple[int, int, dict[str, Any]]] = []
    current_turn = 0
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            current_turn += 1
        rows.append((idx, current_turn, message))
    return rows


def blocked_turn_indices(messages: list[dict[str, Any]]) -> set[int]:
    return {
        turn_index
        for _, turn_index, message in indexed_messages_by_turn(messages)
        if turn_index > 0 and is_blocked_message(message)
    }


def unblocked_turn_messages(messages: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    blocked_turns = blocked_turn_indices(messages)
    return [
        (idx, turn_index, message)
        for idx, turn_index, message in indexed_messages_by_turn(messages)
        if turn_index == 0 or turn_index not in blocked_turns
    ]
