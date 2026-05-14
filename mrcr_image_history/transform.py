"""Convert older chat messages into image blocks while keeping recent turns as text."""

from __future__ import annotations

import re
from typing import Any

from .renderer import ConversationImageRenderer, ConversationImageRendererConfig

try:
    from rank_bm25 import BM25Plus
except ImportError as exc:  # pragma: no cover - depends on local environment
    BM25Plus = None  # type: ignore[assignment]
    _BM25_IMPORT_ERROR = exc
else:
    _BM25_IMPORT_ERROR = None


RECENT_TEXT_TURNS = 20
HIGHRES_RELATED_TOP_K = 3
HIGHRES_IMAGE_CONFIG = ConversationImageRendererConfig(
    width=1024,
    max_height=1024,
    min_height=128,
    font_size=20,
    margin=4,
    box_padding=4,
)
TOKEN_RE = re.compile(r"(?u)[^\W_]+(?:[_\-.][^\W_]+)*")
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
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _as_text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _as_image_block(data_url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url}}


def _message_with_content(message: dict[str, Any], content: Any) -> dict[str, Any]:
    updated = dict(message)
    updated["content"] = content
    return updated


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _content_to_text(message.get("content", ""))
    return ""


def _normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _turn_ranges(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        if start_index is not None:
            ranges.append((start_index, index - 1))
        start_index = index

    if start_index is not None:
        ranges.append((start_index, len(messages) - 1))

    return ranges


def _join_message_text(
    messages: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    *,
    role: str | None = None,
    role_not: str | None = None,
) -> str:
    parts: list[str] = []
    for message in messages[start_index : end_index + 1]:
        message_role = message.get("role")
        if role is not None and message_role != role:
            continue
        if role_not is not None and message_role == role_not:
            continue
        parts.append(_content_to_text(message.get("content", "")))
    return "\n".join(parts)


def _bm25_scores(
    tokenized_docs: list[list[str]],
    query_tokens: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 1.0,
) -> list[float]:
    if BM25Plus is None:
        raise RuntimeError("Missing dependency: install rank-bm25 to use highres related search") from _BM25_IMPORT_ERROR
    if not tokenized_docs or not any(tokenized_docs):
        return [0.0 for _ in tokenized_docs]
    return [float(score) for score in BM25Plus(tokenized_docs, k1=k1, b=b, delta=delta).get_scores(query_tokens)]


def _related_message_numbers(
    messages: list[dict[str, Any]],
    user_input: str,
    *,
    top_k: int,
    user_weight: float = 1.0,
    assistant_weight: float = 0.35,
) -> set[int]:
    if top_k <= 0 or not user_input:
        return set()

    query_tokens = _tokenize(_normalize_text(user_input))
    if not query_tokens:
        return set()

    ranges = _turn_ranges(messages)
    if not ranges:
        return set()

    user_docs = [
        _tokenize(_normalize_text(_join_message_text(messages, start, end, role="user")))
        for start, end in ranges
    ]
    assistant_docs = [
        _tokenize(_normalize_text(_join_message_text(messages, start, end, role_not="user")))
        for start, end in ranges
    ]

    user_scores = _bm25_scores(user_docs, query_tokens)
    assistant_scores = _bm25_scores(assistant_docs, query_tokens)
    ranked = sorted(
        (
            (user_weight * user_scores[index] + assistant_weight * assistant_scores[index], start, end)
            for index, (start, end) in enumerate(ranges)
        ),
        key=lambda item: (-item[0], item[1]),
    )

    numbers: set[int] = set()
    for score, start, end in ranked[:top_k]:
        if score <= 0:
            continue
        numbers.update(range(start + 1, end + 2))
    return numbers


def build_image_history_messages(
    messages: list[dict[str, Any]],
    *,
    recent_turns: int = RECENT_TEXT_TURNS,
    renderer: ConversationImageRenderer | None = None,
    exclude_active_user: bool = False,
    highres_related_top_k: int = HIGHRES_RELATED_TOP_K,
    highres_config: ConversationImageRendererConfig | None = None,
) -> list[dict[str, Any]]:
    """Return messages where old history is replaced by image blocks.

    The returned shape is OpenAI/chat-style dictionaries. Older messages are rendered
    into data-url images and placed at the front of the first recent user message,
    matching the merge behavior in `t1/message_loader.py`.

    For MRCR eval, keep `exclude_active_user=False` so the final query remains present.
    Set it to True only when another runtime appends the active user message separately.
    """
    original_messages = [msg for msg in messages if isinstance(msg, dict)]
    active_user_input = _last_user_text(original_messages)
    source_messages = original_messages
    if exclude_active_user:
        source_messages = exclude_active_user_turn(source_messages)

    old_messages, recent_messages = split_recent_turns(
        source_messages,
        recent_turns=recent_turns,
    )
    if not old_messages:
        return [dict(message) for message in recent_messages]

    selected_highres_config = highres_config or HIGHRES_IMAGE_CONFIG
    if renderer is None:
        image_renderer = ConversationImageRenderer(
            highres_config=selected_highres_config if highres_related_top_k > 0 else None,
        )
    elif highres_related_top_k > 0 and renderer.highres_config is None:
        image_renderer = ConversationImageRenderer(
            renderer.config,
            highres_config=selected_highres_config,
        )
    else:
        image_renderer = renderer

    highres_message_numbers = _related_message_numbers(
        old_messages,
        active_user_input,
        top_k=highres_related_top_k,
    )
    old_blocks: list[dict[str, Any]] = [_as_text_block(IMAGE_HISTORY_PROMPT)]
    old_blocks.extend(
        _as_image_block(data_url)
        for data_url in image_renderer.render_data_urls(
            old_messages,
            highres_message_numbers=highres_message_numbers,
        )
    )

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
