"""Attachment metadata and storage orchestration."""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Iterable

from PIL import Image, ImageOps
from psycopg.types.json import Jsonb

from ..infra.db import Database
from ..infra.settings import Settings
from ..infra.llm import _normalize_base_url
from .attachment_storage import LocalAttachmentStorage

logger = logging.getLogger(__name__)


IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
PDF_MIME_TYPE = "application/pdf"
TEXT_MIME_TYPES = {"text/plain", "text/markdown", "text/csv"}


@dataclass(frozen=True)
class AttachmentUploadResult:
    id: str
    conversation_id: str
    filename: str
    mime_type: str
    size_bytes: int
    kind: str
    status: str


def _safe_user_id(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id).strip("._-")
    return safe[:80] or "user"


def _safe_filename(filename: str | None) -> str:
    name = (filename or "upload").replace("\\", "/").split("/")[-1].strip()
    return name or "upload"


def _infer_mime_type(filename: str, content_type: str | None) -> str:
    if content_type and content_type != "application/octet-stream":
        return content_type.split(";")[0].strip().lower()
    guessed, _ = mimetypes.guess_type(filename)
    return (guessed or "application/octet-stream").lower()


def _kind_for_mime(mime_type: str) -> str:
    if mime_type in IMAGE_MIME_TYPES:
        return "image"
    if mime_type == PDF_MIME_TYPE:
        return "pdf"
    if mime_type in TEXT_MIME_TYPES:
        return "text"
    if mime_type in {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }:
        return "document"
    if mime_type in {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }:
        return "spreadsheet"
    return "unknown"


def attachment_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "filename": str(row.get("filename") or "upload"),
        "mime_type": str(row.get("mime_type") or "application/octet-stream"),
        "kind": str(row.get("kind") or "unknown"),
    }


class AttachmentService:
    def __init__(self, db: Database, settings: Settings, storage: LocalAttachmentStorage):
        self.db = db
        self.settings = settings
        self.storage = storage

    async def upload(
        self,
        *,
        conversation_id: str,
        user_id: str,
        upload_file,
    ) -> AttachmentUploadResult:
        await self._require_conversation(conversation_id, user_id)

        filename = _safe_filename(getattr(upload_file, "filename", None))
        mime_type = _infer_mime_type(filename, getattr(upload_file, "content_type", None))
        if mime_type not in self.settings.attachment_allowed_mime_types:
            raise ValueError(f"Unsupported file type: {mime_type}")

        kind = _kind_for_mime(mime_type)
        max_mb = self.settings.attachment_max_image_mb if kind == "image" else self.settings.attachment_max_file_mb
        max_bytes = max_mb * 1024 * 1024
        attachment_id = str(uuid.uuid4())
        storage_key = self._storage_key(user_id, conversation_id, attachment_id, "original")

        size_bytes, sha256 = await self.storage.write_stream(
            storage_key,
            upload_file,
            max_bytes=max_bytes,
        )

        metadata: dict[str, Any] = {}
        try:
            if kind == "image":
                try:
                    metadata.update(self._validate_and_thumbnail_image(storage_key, mime_type))
                except Exception as exc:
                    raise ValueError("Invalid image file.") from exc

            await self.db.execute(
                """
                INSERT INTO conversation_attachments (
                    id, conversation_id, user_id, filename, mime_type, size_bytes,
                    sha256, kind, storage_backend, storage_key, status, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'uploaded', %s)
                """,
                (
                    attachment_id,
                    conversation_id,
                    user_id,
                    filename,
                    mime_type,
                    size_bytes,
                    sha256,
                    kind,
                    self.settings.attachment_storage_backend,
                    storage_key,
                    Jsonb(metadata),
                ),
            )
        except Exception:
            self.storage.delete(storage_key)
            thumbnail_key = metadata.get("thumbnail_key")
            if isinstance(thumbnail_key, str):
                self.storage.delete(thumbnail_key)
            raise

        return AttachmentUploadResult(
            id=attachment_id,
            conversation_id=conversation_id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            kind=kind,
            status="uploaded",
        )

    async def list_for_conversation(self, conversation_id: str, user_id: str) -> list[dict[str, Any]]:
        await self._require_conversation(conversation_id, user_id)
        rows = await self.db.fetch_all(
            """
            SELECT id, conversation_id, user_id, turn_index, filename, mime_type,
                   size_bytes, sha256, kind, storage_backend, storage_key,
                   status, metadata, created_at, updated_at
            FROM conversation_attachments
            WHERE conversation_id = %s AND user_id = %s AND status <> 'deleted'
            ORDER BY created_at ASC, id ASC
            """,
            (conversation_id, user_id),
        )
        return [self._normalize_row(row) for row in rows]

    async def get_for_user(self, attachment_id: str, user_id: str) -> dict[str, Any]:
        row = await self.db.fetch_one(
            """
            SELECT id, conversation_id, user_id, turn_index, filename, mime_type,
                   size_bytes, sha256, kind, storage_backend, storage_key,
                   status, metadata, created_at, updated_at
            FROM conversation_attachments
            WHERE id = %s AND user_id = %s AND status <> 'deleted'
            """,
            (attachment_id, user_id),
        )
        if not row:
            raise ValueError("Attachment not found.")
        return self._normalize_row(row)

    async def get_many_for_conversation(
        self,
        *,
        attachment_ids: Iterable[str],
        conversation_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for attachment_id in attachment_ids:
            row = await self.db.fetch_one(
                """
                SELECT id, conversation_id, user_id, turn_index, filename, mime_type,
                       size_bytes, sha256, kind, storage_backend, storage_key,
                       status, metadata, created_at, updated_at
                FROM conversation_attachments
                WHERE id = %s
                  AND conversation_id = %s
                  AND user_id = %s
                  AND status <> 'deleted'
                """,
                (attachment_id, conversation_id, user_id),
            )
            if not row:
                raise ValueError("Attachment not found for this conversation.")
            rows.append(self._normalize_row(row))
        return rows

    async def ensure_provider_file_ref(self, row: dict[str, Any], provider: str) -> str | None:
        if not self._provider_supports_attachment(row, provider):
            return None
        try:
            if provider == "openai_compatible":
                return await self._ensure_openai_file_ref(row)
            if provider == "anthropic":
                return await self._ensure_anthropic_file_ref(row)
        except Exception:
            logger.exception(
                "Provider file upload failed for attachment %s via provider %s; falling back to metadata",
                row.get("id"),
                provider,
            )
        return None

    async def attach_to_turn(
        self,
        *,
        cursor,
        conversation_id: str,
        user_id: str,
        attachment_ids: list[str],
        turn_index: int,
    ) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for attachment_id in attachment_ids:
            await cursor.execute(
                """
                SELECT id, filename, mime_type, kind, turn_index, status
                FROM conversation_attachments
                WHERE id = %s
                  AND conversation_id = %s
                  AND user_id = %s
                  AND status <> 'deleted'
                FOR UPDATE
                """,
                (attachment_id, conversation_id, user_id),
            )
            row = await cursor.fetchone()
            if not row:
                raise ValueError("Attachment not found for this conversation.")
            existing_turn = row.get("turn_index")
            if existing_turn is not None and int(existing_turn) != turn_index:
                raise ValueError("Attachment is already attached to a different turn.")

            await cursor.execute(
                """
                UPDATE conversation_attachments
                SET turn_index = %s,
                    status = 'attached',
                    updated_at = NOW()
                WHERE id = %s
                """,
                (turn_index, attachment_id),
            )
            refs.append(attachment_ref(row))
        return refs

    def content_path(self, row: dict[str, Any], *, thumbnail: bool = False):
        storage_key = str(row["storage_key"])
        if thumbnail:
            metadata = row.get("metadata") or {}
            if isinstance(metadata, dict) and isinstance(metadata.get("thumbnail_key"), str):
                storage_key = metadata["thumbnail_key"]
        return self.storage.path_for(storage_key)

    def image_data_url(self, row: dict[str, Any]) -> str:
        if row.get("kind") != "image":
            raise ValueError("Attachment is not an image.")
        data = self.storage.read_bytes(str(row["storage_key"]))
        mime_type = str(row.get("mime_type") or "image/jpeg")
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def file_context_text(self, row: dict[str, Any]) -> str:
        filename = row.get("filename") or "upload"
        mime_type = row.get("mime_type") or "application/octet-stream"
        size_bytes = row.get("size_bytes") or 0
        label = "Uploaded image" if row.get("kind") == "image" else "Uploaded file"
        return f"{label}: {filename} ({mime_type}, {size_bytes} bytes)."

    def provider_file_block(self, row: dict[str, Any], provider: str, file_id: str) -> dict[str, Any]:
        mime_type = str(row.get("mime_type") or "application/octet-stream")
        filename = str(row.get("filename") or "upload")
        kind = row.get("kind")
        if provider == "anthropic" and kind == "image":
            return {
                "type": "image",
                "file_id": file_id,
            }
        return {
            "type": "file",
            "file_id": file_id,
            "mime_type": mime_type,
            "filename": filename,
        }

    @staticmethod
    def _provider_supports_attachment(row: dict[str, Any], provider: str) -> bool:
        mime_type = str(row.get("mime_type") or "")
        kind = row.get("kind")
        if provider == "openai_compatible":
            return mime_type == PDF_MIME_TYPE
        if provider == "anthropic":
            return kind == "image" or mime_type in {PDF_MIME_TYPE, "text/plain"}
        return False

    async def _require_conversation(self, conversation_id: str, user_id: str) -> None:
        row = await self.db.fetch_one(
            """
            SELECT id
            FROM conversations
            WHERE id = %s AND user_id = %s AND is_active = TRUE
            """,
            (conversation_id, user_id),
        )
        if not row:
            raise ValueError("Conversation not found.")

    def _storage_key(self, user_id: str, conversation_id: str, attachment_id: str, name: str) -> str:
        return (
            f"users/{_safe_user_id(user_id)}/conversations/{conversation_id}"
            f"/attachments/{attachment_id}/{name}"
        )

    def _validate_and_thumbnail_image(self, storage_key: str, mime_type: str) -> dict[str, Any]:
        data = self.storage.read_bytes(storage_key)
        with Image.open(BytesIO(data)) as image:
            image.verify()

        with Image.open(BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((512, 512))
            buffer = BytesIO()
            thumbnail.save(buffer, format="WEBP", quality=82)

        thumbnail_key = storage_key.rsplit("/", 1)[0] + "/thumbnail.webp"
        self.storage.write_bytes(thumbnail_key, buffer.getvalue())
        return {
            "width": width,
            "height": height,
            "thumbnail_key": thumbnail_key,
            "thumbnail_mime_type": "image/webp",
            "verified_mime_type": mime_type,
        }

    async def _ensure_openai_file_ref(self, row: dict[str, Any]) -> str | None:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        refs = metadata.get("provider_refs") if isinstance(metadata.get("provider_refs"), dict) else {}
        openai_ref = refs.get("openai") if isinstance(refs.get("openai"), dict) else {}
        file_id = openai_ref.get("file_id")
        if isinstance(file_id, str) and file_id:
            return file_id
        if not self.settings.openai_api_key:
            return None

        from openai import AsyncOpenAI

        client_kwargs = {"api_key": self.settings.openai_api_key}
        # docker-compose may pass OPENAI_BASE_URL as an empty env var. The
        # OpenAI SDK reads that env var and treats it as an override, so pass
        # the official default explicitly when no configured base URL exists.
        client_kwargs["base_url"] = (
            _normalize_base_url(self.settings.openai_base_url)
            or "https://api.openai.com/v1"
        )
        client = AsyncOpenAI(**client_kwargs)
        data = self.storage.read_bytes(str(row["storage_key"]))
        uploaded = await client.files.create(
            file=(str(row.get("filename") or "upload"), data, str(row.get("mime_type") or "application/octet-stream")),
            purpose="user_data",
        )
        file_id = str(uploaded.id)
        await self._store_provider_ref(
            str(row["id"]),
            provider_key="openai",
            provider_ref={
                "file_id": file_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "purpose": "user_data",
            },
        )
        if not isinstance(row.get("metadata"), dict):
            row["metadata"] = {}
        row["metadata"].setdefault("provider_refs", {})["openai"] = {"file_id": file_id}
        return file_id

    async def _ensure_anthropic_file_ref(self, row: dict[str, Any]) -> str | None:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        refs = metadata.get("provider_refs") if isinstance(metadata.get("provider_refs"), dict) else {}
        anthropic_ref = refs.get("anthropic") if isinstance(refs.get("anthropic"), dict) else {}
        file_id = anthropic_ref.get("file_id")
        if isinstance(file_id, str) and file_id:
            return file_id
        if not self.settings.anthropic_api_key:
            return None

        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        data = self.storage.read_bytes(str(row["storage_key"]))
        uploaded = await client.beta.files.upload(
            file=(str(row.get("filename") or "upload"), data, str(row.get("mime_type") or "application/octet-stream")),
            betas=["files-api-2025-04-14"],
        )
        file_id = str(uploaded.id)
        await self._store_provider_ref(
            str(row["id"]),
            provider_key="anthropic",
            provider_ref={
                "file_id": file_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "beta": "files-api-2025-04-14",
            },
        )
        if not isinstance(row.get("metadata"), dict):
            row["metadata"] = {}
        row["metadata"].setdefault("provider_refs", {})["anthropic"] = {"file_id": file_id}
        return file_id

    async def _store_provider_ref(
        self,
        attachment_id: str,
        *,
        provider_key: str,
        provider_ref: dict[str, Any],
    ) -> None:
        row = await self.db.fetch_one(
            "SELECT metadata FROM conversation_attachments WHERE id = %s",
            (attachment_id,),
        )
        metadata = row.get("metadata") if row and isinstance(row.get("metadata"), dict) else {}
        provider_refs = metadata.get("provider_refs") if isinstance(metadata.get("provider_refs"), dict) else {}
        provider_refs[provider_key] = provider_ref
        metadata["provider_refs"] = provider_refs
        await self.db.execute(
            """
            UPDATE conversation_attachments
            SET metadata = %s,
                updated_at = NOW()
            WHERE id = %s
            """,
            (Jsonb(metadata), attachment_id),
        )

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        for key in ("id", "conversation_id"):
            if normalized.get(key) is not None:
                normalized[key] = str(normalized[key])
        return normalized
