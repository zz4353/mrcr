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
    "RenderedConversationImage",
    "IMAGE_HISTORY_PROMPT",
    "RECENT_TEXT_TURNS",
    "build_image_history_messages",
    "split_recent_turns",
]
