"""Utilities for converting older MRCR chat history into compact images."""

from .renderer import (
    ConversationImageRenderer,
    ConversationImageRendererConfig,
    RenderedConversationImage,
)
from .transform import (
    IMAGE_HISTORY_PROMPT,
    RECENT_TEXT_TURNS,
    build_image_history_messages,
    split_recent_turns,
)

__all__ = [
    "ConversationImageRenderer",
    "ConversationImageRendererConfig",
    "ConversationTurn",
    "RenderedConversationImage",
    "RelatedTurn",
    "IMAGE_HISTORY_PROMPT",
    "RECENT_TEXT_TURNS",
    "build_image_history_messages",
    "search_related_turns",
    "split_recent_turns",
    "split_into_turns",
]


def __getattr__(name: str):
    if name in {"ConversationTurn", "RelatedTurn", "search_related_turns", "split_into_turns"}:
        from importlib import import_module

        search = import_module(".search", __name__)
        return getattr(search, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
