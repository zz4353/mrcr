"""Convert older chat messages into image blocks while keeping recent turns as text."""

from __future__ import annotations

from typing import Any

from .renderer import ConversationImageRenderer


RECENT_TEXT_TURNS = 3
IMAGE_HISTORY_PROMPT = (
    "The user's earlier conversation history is important context for this chat, but it could not be sent directly as text. "
    "It has therefore been captured in the images below, shown in chronological order. "
    "Each red number marks a chat message, and the numbering continues across images. "
    "Read these numbered images as the older part of the same conversation, then continue from the recent text messages that follow."
)


def split_recent_turns(
    messages: list[dict[str, Any]],
    *,
    recent_turns: int = RECENT_TEXT_TURNS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into old history and the latest N user turns.

    A turn starts at a user message and continues until the next user message.
    This mirrors `t1/message_loader.py`.
    """
    if recent_turns <= 0:
        return messages, []

    indexed: list[tuple[int, int, dict[str, Any]]] = []
    current_turn = 0
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            current_turn += 1
        indexed.append((index, current_turn, message))

    if current_turn <= recent_turns:
        return [], messages

    first_recent_turn = current_turn - recent_turns + 1
    old_messages = [
        message
        for _, turn_index, message in indexed
        if turn_index == 0 or turn_index < first_recent_turn
    ]
    recent_messages = [
        message
        for _, turn_index, message in indexed
        if turn_index >= first_recent_turn
    ]
    return old_messages, recent_messages


def exclude_active_user_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the last user message, matching the original app runtime behavior."""
    if messages and messages[-1].get("role") == "user":
        return messages[:-1]
    return messages


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _as_text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _as_image_block(data_url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url}}


def _message_with_content(message: dict[str, Any], content: Any) -> dict[str, Any]:
    updated = dict(message)
    updated["content"] = content
    return updated


def build_image_history_messages(
    messages: list[dict[str, Any]],
    *,
    recent_turns: int = RECENT_TEXT_TURNS,
    renderer: ConversationImageRenderer | None = None,
    exclude_active_user: bool = False,
) -> list[dict[str, Any]]:
    """Return messages where old history is replaced by image blocks.

    The returned shape is OpenAI/chat-style dictionaries. Older messages are rendered
    into data-url images and placed at the front of the first recent user message,
    matching the merge behavior in `t1/message_loader.py`.

    For MRCR eval, keep `exclude_active_user=False` so the final query remains present.
    Set it to True only when another runtime appends the active user message separately.
    """
    source_messages = [msg for msg in messages if isinstance(msg, dict)]
    if exclude_active_user:
        source_messages = exclude_active_user_turn(source_messages)

    old_messages, recent_messages = split_recent_turns(
        source_messages,
        recent_turns=recent_turns,
    )
    if not old_messages:
        return [dict(message) for message in recent_messages]

    image_renderer = renderer or ConversationImageRenderer()
    old_blocks: list[dict[str, Any]] = [_as_text_block(IMAGE_HISTORY_PROMPT)]
    old_blocks.extend(_as_image_block(data_url) for data_url in image_renderer.render_data_urls(old_messages))

    if not recent_messages:
        return [{"role": "user", "content": old_blocks}]

    first_recent = dict(recent_messages[0])
    first_content = first_recent.get("content", "")
    if isinstance(first_content, list):
        merged_content = old_blocks + list(first_content)
    else:
        merged_content = old_blocks + [_as_text_block(_content_to_text(first_content))]

    converted = [_message_with_content(first_recent, merged_content)]
    converted.extend(dict(message) for message in recent_messages[1:])
    return converted
