"""Render conversation history into compact base64 images."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


ROLE_LABELS = {
    "user": "User",
    "human": "User",
    "assistant": "Assistant",
    "ai": "Assistant",
    "tool": "Tool result",
    "system": "System",
}


@dataclass(frozen=True)
class ConversationImageRendererConfig:
    width: int = 512
    max_height: int = 512
    min_height: int = 64
    font_size: int = 10
    margin: int = 2
    box_padding: int = 2
    line_spacing: int = 0
    background: str = "white"
    image_format: str = "PNG"


@dataclass(frozen=True)
class RenderedConversationImage:
    index: int
    bytes: bytes
    mime_type: str

    @property
    def base64(self) -> str:
        return base64.b64encode(self.bytes).decode("ascii")

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64}"


class ConversationImageRenderer:
    """Render conversation history into compact readable images."""

    def __init__(self, config: ConversationImageRendererConfig | None = None):
        self.config = config or ConversationImageRendererConfig()
        self._validate_config()
        self.font = self._load_font(self.config.font_size)
        self.char_width, self.line_height = self._font_metrics(self.font)

    def render_pages(
        self,
        conversation: list[Any],
        output_dir: str | Path = "conversation_pages",
        *,
        prefix: str = "conversation",
        clear_output: bool = True,
    ) -> list[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if clear_output:
            for old_image in output_dir.glob(f"{prefix}_*.png"):
                old_image.unlink()

        images = self.render_images(conversation)
        digits = len(str(max(1, len(images))))
        saved_paths: list[Path] = []

        for image in images:
            output_path = output_dir / f"{prefix}_{image.index:0{digits}d}.png"
            output_path.write_bytes(image.bytes)
            saved_paths.append(output_path)

        return saved_paths

    def render_base64_pages(self, conversation: list[Any]) -> list[str]:
        return [image.base64 for image in self.render_images(conversation)]

    def render_data_urls(self, conversation: list[Any]) -> list[str]:
        return [image.data_url for image in self.render_images(conversation)]

    def render_images(self, conversation: list[Any]) -> list[RenderedConversationImage]:
        pages = self._paginate(conversation)
        rendered: list[RenderedConversationImage] = []

        for page_index, page in enumerate(pages, 1):
            image = self._draw_page(page)
            buffer = BytesIO()
            image.save(buffer, format=self.config.image_format)
            rendered.append(
                RenderedConversationImage(
                    index=page_index,
                    bytes=buffer.getvalue(),
                    mime_type=self._mime_type(),
                )
            )

        return rendered

    def _validate_config(self) -> None:
        if self.config.width <= 0 or self.config.max_height <= 0:
            raise ValueError("width and max_height must be positive")
        if self.config.min_height <= 0:
            raise ValueError("min_height must be positive")
        if self.config.min_height > self.config.max_height:
            raise ValueError("min_height cannot be greater than max_height")

    def _paginate(self, conversation: list[Any]) -> list[list[tuple[int, list[str], int]]]:
        probe = Image.new("RGB", (self.config.width, 1), color=self.config.background)
        probe_draw = ImageDraw.Draw(probe)

        available_width = (
            self.config.width
            - (2 * self.config.margin)
            - (2 * self.config.box_padding)
            - 2
        )
        chars_per_line = max(1, available_width // self.char_width)
        max_text_height = self.config.max_height - (2 * self.config.margin)
        max_lines_per_box = max(
            1,
            (max_text_height - (2 * self.config.box_padding)) // self.line_height,
        )

        pages: list[list[tuple[int, list[str], int]]] = []
        current_page: list[tuple[int, list[str], int]] = []
        current_y = self.config.margin

        def add_page() -> None:
            nonlocal current_page, current_y
            if current_page:
                pages.append(current_page)
            current_page = []
            current_y = self.config.margin

        for number, item in enumerate(conversation, 1):
            text = self._conversation_item_to_text(item)
            badge_gap = 2
            badge_width = self._badge_width(probe_draw, number)
            max_text_width = max(1, available_width - badge_width - badge_gap)
            wrapped_lines = self._wrap_numbered_text(
                text,
                chars_per_line=chars_per_line,
                max_text_width=max_text_width,
                draw=probe_draw,
            )

            for line_chunk in self._chunks(wrapped_lines, max_lines_per_box):
                text_height = len(line_chunk) * self.line_height + self.config.box_padding * 2
                box_height = text_height

                if current_page and current_y + box_height > self.config.max_height - self.config.margin:
                    add_page()

                if box_height > max_text_height:
                    safe_line_count = max(
                        1,
                        math.floor(
                            (max_text_height - (2 * self.config.box_padding))
                            / self.line_height
                        ),
                    )
                    for smaller_chunk in self._chunks(line_chunk, safe_line_count):
                        text_height = (
                            len(smaller_chunk) * self.line_height + self.config.box_padding * 2
                        )
                        box_height = text_height
                        if (
                            current_page
                            and current_y + box_height > self.config.max_height - self.config.margin
                        ):
                            add_page()
                        current_page.append((number, smaller_chunk, current_y))
                        current_y += box_height + self.config.line_spacing
                    continue

                current_page.append((number, line_chunk, current_y))
                current_y += box_height + self.config.line_spacing

        add_page()
        return pages

    def _draw_page(self, page: list[tuple[int, list[str], int]]) -> Image.Image:
        content_bottom = self.config.margin
        for _, lines, y in page:
            text_height = len(lines) * self.line_height + self.config.box_padding * 2
            content_bottom = max(content_bottom, y + text_height)

        height = min(
            self.config.max_height,
            max(self.config.min_height, content_bottom + self.config.margin),
        )
        image = Image.new("RGB", (self.config.width, height), color=self.config.background)
        draw = ImageDraw.Draw(image)

        for number, lines, y in page:
            self._draw_box(draw, number=number, lines=lines, y=y)

        return image

    def _draw_box(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        number: int,
        lines: list[str],
        y: int,
    ) -> int:
        text_height = len(lines) * self.line_height + self.config.box_padding * 2
        box_height = text_height

        box_x1 = self.config.margin
        box_y1 = y
        box_x2 = self.config.width - self.config.margin
        box_y2 = y + box_height
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline="red", width=1)

        number_text = str(number)
        number_bbox = draw.textbbox((0, 0), number_text, font=self.font)
        number_width = number_bbox[2] - number_bbox[0]
        number_height = number_bbox[3] - number_bbox[1]
        badge_x = box_x1 + self.config.box_padding
        badge_y = box_y1 + self.config.box_padding
        badge_width = self._badge_width(draw, number)
        badge_height = self.line_height

        draw.rectangle(
            [badge_x, badge_y, badge_x + badge_width, badge_y + badge_height - 1],
            fill="red",
            outline="red",
        )
        draw.text(
            (
                badge_x + (badge_width - number_width) // 2,
                badge_y + (badge_height - number_height) // 2 - 1,
            ),
            number_text,
            fill="white",
            font=self.font,
        )

        text_x = badge_x + badge_width + 2
        text_y = box_y1 + self.config.box_padding
        for line in lines:
            draw.text((text_x, text_y), line, fill="black", font=self.font)
            text_y += self.line_height

        return box_y2

    def _mime_type(self) -> str:
        image_format = self.config.image_format.upper()
        if image_format == "JPG":
            image_format = "JPEG"
        return f"image/{image_format.lower()}"

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        font_names = (
            "consola.ttf",
            "cour.ttf",
            "DejaVuSansMono.ttf",
            "arial.ttf",
        )
        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, size)
            except OSError:
                pass
        return ImageFont.load_default()

    def _font_metrics(self, font: ImageFont.ImageFont) -> tuple[int, int]:
        probe = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), "M", font=font)
        char_width = max(1, bbox[2] - bbox[0])
        line_height = max(1, bbox[3] - bbox[1] + 3)
        return char_width, line_height

    def _text_width(self, draw: ImageDraw.ImageDraw, text: str) -> int:
        if not text:
            return 0
        bbox = draw.textbbox((0, 0), text, font=self.font)
        return bbox[2] - bbox[0]

    def _badge_width(self, draw: ImageDraw.ImageDraw, number: int) -> int:
        number_text = str(number)
        number_width = self._text_width(draw, number_text)
        return max(self.line_height, number_width + 5)

    def _compact_json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _content_to_text(self, content: Any) -> str:
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
                else:
                    parts.append(self._compact_json(block))
            return "\n".join(parts)
        return self._compact_json(content)

    def _format_tool_call(self, tool_call: Any) -> str:
        if not isinstance(tool_call, dict):
            return self._compact_json(tool_call)

        name = (
            tool_call.get("name")
            or tool_call.get("tool_name")
            or tool_call.get("function", {}).get("name")
        )
        args = tool_call.get("args")
        if args is None and isinstance(tool_call.get("function"), dict):
            args = tool_call["function"].get("arguments")

        parts = []
        if name:
            parts.append(str(name))
        if args not in (None, "", {}):
            parts.append(self._content_to_text(args))
        return "(" + ", ".join(parts) + ")" if parts else self._compact_json(tool_call)

    def _format_message_dict(self, item: dict[str, Any]) -> str:
        role = str(item.get("role") or item.get("speaker") or item.get("name") or "message")
        label = ROLE_LABELS.get(role.lower(), role)
        content = self._content_to_text(item.get("content", item.get("text", item.get("message", ""))))

        lines = [f"{label}: {content}".rstrip()]

        tool_calls = item.get("tool_calls") or []
        if tool_calls:
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            for tool_call in tool_calls:
                lines.append(f"ToolCall: {self._format_tool_call(tool_call)}")

        return "\n".join(lines)

    def _conversation_item_to_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            if any(key in item for key in ("role", "content", "tool_calls", "tool_call_id", "tool_name")):
                return self._format_message_dict(item)
            return self._compact_json(item)

        if isinstance(item, (list, tuple)) and len(item) == 2:
            return f"{item[0]}: {item[1]}"

        return str(item)

    def _wrap_text_by_pixels(
        self,
        text: str,
        max_width: int,
        *,
        draw: ImageDraw.ImageDraw,
        tabsize: int = 4,
    ) -> list[str]:
        lines: list[str] = []
        for raw_line in text.expandtabs(tabsize).splitlines() or [""]:
            lines.extend(self._wrap_line_by_pixels(raw_line, max_width, draw=draw))
        return lines

    def _wrap_line_by_pixels(
        self,
        line: str,
        max_width: int,
        *,
        draw: ImageDraw.ImageDraw,
    ) -> list[str]:
        if not line:
            return [""]

        max_width = max(1, max_width)
        wrapped: list[str] = []
        remaining = line

        while remaining:
            if self._text_width(draw, remaining) <= max_width:
                wrapped.append(remaining)
                break

            low = 1
            high = len(remaining)
            fit = 1
            while low <= high:
                mid = (low + high) // 2
                if self._text_width(draw, remaining[:mid]) <= max_width:
                    fit = mid
                    low = mid + 1
                else:
                    high = mid - 1

            prefix = remaining[:fit]
            break_at = prefix.rfind(" ")
            if break_at > 0 and prefix[:break_at].strip():
                wrapped.append(prefix[:break_at].rstrip())
                remaining = remaining[break_at + 1 :].lstrip()
            else:
                wrapped.append(prefix.rstrip() or prefix)
                remaining = remaining[fit:].lstrip()

        return wrapped or [""]

    def _looks_structured_or_code(self, lines: list[str]) -> bool:
        for line in lines:
            if not line.strip():
                continue
            if line[:1].isspace():
                return True
            stripped = line.strip()
            if stripped in {"{", "}", "};"}:
                return True
            if any(marker in stripped for marker in ("{", "}", "\\", "[Unit]", "[Service]", "[Install]")):
                return True
        return False

    def _compact_short_multiline_text(self, text: str, chars_per_line: int) -> str:
        raw_lines = text.expandtabs(4).splitlines()
        non_empty = [line.strip() for line in raw_lines if line.strip()]

        if len(non_empty) <= 1:
            return text
        if self._looks_structured_or_code(raw_lines):
            return "\n".join(line.rstrip() for line in raw_lines if line.strip())

        longest_line = max(len(line) for line in non_empty)
        joined = " | ".join(non_empty)
        compact_limit = chars_per_line * 3

        if longest_line <= chars_per_line and len(joined) <= compact_limit:
            return joined
        return text

    def _wrap_numbered_text(
        self,
        text: str,
        *,
        chars_per_line: int,
        max_text_width: int,
        draw: ImageDraw.ImageDraw,
    ) -> list[str]:
        compacted_text = self._compact_short_multiline_text(text, chars_per_line)
        return self._wrap_text_by_pixels(compacted_text, max_text_width, draw=draw)

    def _chunks(self, items: list[str], chunk_size: int) -> Iterable[list[str]]:
        for index in range(0, len(items), chunk_size):
            yield items[index : index + chunk_size]
