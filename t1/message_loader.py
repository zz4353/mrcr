"""Message loader service for loading chat history from conversations table."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .conversation_image_renderer import ConversationImageRenderer

if TYPE_CHECKING:
    from ..infra.db import Database
    from ..infra.settings import Settings
    from .attachments import AttachmentService

logger = logging.getLogger(__name__)

RECENT_TEXT_TURNS = 3
IMAGE_HISTORY_PROMPT = (
    "Older conversation history is provided as images below. "
    "Read them as prior chat context, then continue from the recent text messages."
)


def _deserialize_message(data: dict[str, Any]) -> BaseMessage:
    """Deserialize simplified JSON format back to LangChain message.
    
    Args:
        data: Simplified message dict
        
    Returns:
        BaseMessage object
    """
    role = data.get("role")
    content = data.get("content", "")
    
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        tool_calls = data.get("tool_calls")
        return AIMessage(content=content, tool_calls=tool_calls or [])
    elif role == "tool":
        tool_call_id = data.get("tool_call_id", "")
        if not tool_call_id:
            logger.warning("ToolMessage missing tool_call_id, using empty string")
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=data.get("tool_name")
        )
    else:
        logger.warning(f"Unknown message role '{role}', falling back to HumanMessage")
        return HumanMessage(content=content)


def _split_recent_turns(
    messages: list[dict[str, Any]],
    *,
    recent_turns: int = RECENT_TEXT_TURNS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into old history and the latest N user turns.

    A turn starts at a user message and continues until the next user message.
    Keeping whole turns avoids separating assistant tool calls from their tool
    results in the recent text window.
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


def _exclude_active_user_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude the just-persisted user message that runtime appends separately."""
    if messages and messages[-1].get("role") == "user":
        return messages[:-1]
    return messages


def _role_label(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "message").lower()
    return {"user": "User", "assistant": "Assistant", "tool": "Tool result"}.get(role, role)


def _attachment_refs(message: dict[str, Any]) -> list[dict[str, Any]]:
    attachments = message.get("attachments")
    return [item for item in attachments if isinstance(item, dict)] if isinstance(attachments, list) else []


class MessageLoader:
    """Service for loading chat history from conversations table."""
    
    def __init__(
        self,
        db: "Database",
        attachments: "AttachmentService",
        settings: "Settings",
    ):
        """Initialize MessageLoader.
        
        Args:
            db: Database instance for querying conversations
        """
        self.db = db
        self.attachments = attachments
        self.settings = settings
        self.image_renderer = ConversationImageRenderer()
    
    async def load_history(
        self,
        conversation_id: str,
        user_id: str
    ) -> list[BaseMessage]:
        """Load chat history and convert to LangGraph messages.
        
        Args:
            conversation_id: UUID of the conversation
            user_id: User ID for validation (required)
            
        Returns:
            List of BaseMessage objects (HumanMessage, AIMessage, ToolMessage)
            in insertion order. Returns empty list if conversation not found
            or deserialization fails.
        """
        # Query conversations table with user_id validation
        sql = "SELECT messages FROM conversations WHERE id = %s AND user_id = %s"
        params: tuple[Any, ...] = (conversation_id, user_id)
        
        row = await self.db.fetch_one(sql, params)
        
        if not row:
            logger.warning(f"Conversation {conversation_id} not found for user {user_id}")
            return []
        
        # Get messages from JSONB column
        messages_data = row.get("messages") or []
        
        if not messages_data:
            # Empty conversation - return empty list
            return []
        
        messages = _exclude_active_user_turn(
            [msg for msg in messages_data if isinstance(msg, dict)]
        )

        # Deserialize using custom format
        try:
            old_messages, recent_messages = _split_recent_turns(messages)
            history: list[BaseMessage] = []
            image_count = 0
            file_count = 0

            if old_messages:
                old_blocks, image_count, file_count = await self._build_old_history_blocks(
                    old_messages, conversation_id, user_id
                )
            else:
                old_blocks = []

            if old_blocks and recent_messages:
                # Merge old history blocks into the first recent HumanMessage to avoid
                # consecutive HumanMessages (which some providers reject).
                first_msg, image_count, file_count = await self._deserialize_message(
                    recent_messages[0], conversation_id, user_id,
                    image_count=image_count, file_count=file_count,
                )
                first_content = first_msg.content
                if isinstance(first_content, str):
                    merged: list[dict[str, Any]] = old_blocks + [{"type": "text", "text": first_content}]
                elif isinstance(first_content, list):
                    merged = old_blocks + list(first_content)
                else:
                    merged = old_blocks + [{"type": "text", "text": str(first_content)}]
                history.append(HumanMessage(content=merged))
                remaining = recent_messages[1:]
            elif old_blocks:
                history.append(HumanMessage(content=old_blocks))
                remaining = []
            else:
                remaining = recent_messages

            for msg in remaining:
                msg_result, image_count, file_count = await self._deserialize_message(
                    msg, conversation_id, user_id,
                    image_count=image_count, file_count=file_count,
                )
                history.append(msg_result)
            return history
        except Exception as exc:
            logger.error(f"Failed to deserialize messages for conversation {conversation_id}", exc_info=exc)
            return []

    async def _build_old_history_blocks(
        self,
        messages: list[dict[str, Any]],
        conversation_id: str,
        user_id: str,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Render old messages into image content blocks.

        Returns (content_blocks, final_image_count, final_file_count).
        content_blocks is empty when there is nothing to show.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": IMAGE_HISTORY_PROMPT}]
        text_buffer: list[dict[str, Any]] = []
        image_count = 0
        file_count = 0

        async def flush_text_buffer() -> None:
            nonlocal text_buffer
            if not text_buffer:
                return
            try:
                image_data_urls = await asyncio.to_thread(
                    self.image_renderer.render_data_urls, text_buffer
                )
            except Exception:
                logger.exception(
                    "Failed to render an old history text segment for conversation %s",
                    conversation_id,
                )
                image_data_urls = []
            for data_url in image_data_urls:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            text_buffer = []

        for message in messages:
            refs = _attachment_refs(message)
            if not refs:
                text_buffer.append(message)
                continue

            await flush_text_buffer()
            content.append(
                {
                    "type": "text",
                    "text": f"{_role_label(message)}: {str(message.get('content') or '').strip()}",
                }
            )
            blocks, image_count, file_count = await self._attachment_blocks(
                refs, conversation_id, user_id,
                image_count=image_count, file_count=file_count,
            )
            content.extend(blocks)

        await flush_text_buffer()
        if len(content) <= 1:
            return [], image_count, file_count
        return content, image_count, file_count

    async def _deserialize_message(
        self,
        data: dict[str, Any],
        conversation_id: str,
        user_id: str,
        *,
        image_count: int = 0,
        file_count: int = 0,
    ) -> tuple[BaseMessage, int, int]:
        role = data.get("role")
        content = data.get("content", "")
        refs = _attachment_refs(data)
        if not refs:
            return _deserialize_message(data), image_count, file_count

        blocks: list[dict[str, Any]] = [{"type": "text", "text": str(content)}]
        attachment_blocks, image_count, file_count = await self._attachment_blocks(
            refs, conversation_id, user_id,
            image_count=image_count, file_count=file_count,
        )
        blocks.extend(attachment_blocks)

        if role == "user":
            return HumanMessage(content=blocks), image_count, file_count
        if role == "assistant":
            return AIMessage(content=_blocks_to_text(blocks)), image_count, file_count
        if role == "tool":
            return ToolMessage(
                content=_blocks_to_text(blocks),
                tool_call_id=data.get("tool_call_id", ""),
                name=data.get("tool_name"),
            ), image_count, file_count
        return HumanMessage(content=blocks), image_count, file_count

    async def _attachment_blocks(
        self,
        refs: list[dict[str, Any]],
        conversation_id: str,
        user_id: str,
        *,
        image_count: int,
        file_count: int,
    ) -> tuple[list[dict[str, Any]], int, int]:
        blocks: list[dict[str, Any]] = []
        current_image_count = image_count
        current_file_count = file_count
        for ref in refs:
            attachment_id = ref.get("id")
            if not attachment_id:
                continue
            try:
                [row] = await self.attachments.get_many_for_conversation(
                    attachment_ids=[str(attachment_id)],
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
            except Exception:
                logger.exception("Failed to load attachment %s for conversation %s", attachment_id, conversation_id)
                blocks.append({"type": "text", "text": f"Attachment unavailable: {attachment_id}"})
                continue

            label = self.attachments.file_context_text(row)
            if row.get("kind") == "image":
                if current_image_count >= self.settings.attachment_max_history_images:
                    blocks.append({"type": "text", "text": f"Image attachment omitted from context: {label}"})
                    continue
                blocks.append({"type": "text", "text": label})
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self.attachments.image_data_url(row)},
                    }
                )
                current_image_count += 1
            else:
                if current_file_count >= self.settings.attachment_max_history_files:
                    blocks.append({"type": "text", "text": f"File attachment omitted from context: {label}"})
                    continue
                file_id = await self.attachments.ensure_provider_file_ref(row, self.settings.llm_provider)
                if file_id:
                    blocks.append({"type": "text", "text": label})
                    blocks.append(self.attachments.provider_file_block(row, self.settings.llm_provider, file_id))
                else:
                    blocks.append({"type": "text", "text": label})
                current_file_count += 1
        return blocks, current_image_count, current_file_count


def _blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for block in blocks:
        if block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
        elif block.get("type") == "image_url":
            texts.append("[image attachment]")
    return "\n".join(text for text in texts if text).strip()
