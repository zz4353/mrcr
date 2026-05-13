"""Attachment storage backends."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO


class LocalAttachmentStorage:
    """Filesystem-backed attachment storage.

    Storage keys are relative POSIX-style paths. They are resolved under
    `root` and never allowed to escape it.
    """

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, storage_key: str) -> Path:
        relative = Path(storage_key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid storage key.")
        resolved = (self.root / relative).resolve()
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError("Storage key escapes root.")
        return resolved

    async def write_stream(
        self,
        storage_key: str,
        source,
        *,
        max_bytes: int,
        chunk_size: int = 1024 * 1024,
    ) -> tuple[int, str]:
        path = self.path_for(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")

        hasher = hashlib.sha256()
        size = 0
        try:
            with temp_path.open("wb") as target:
                while True:
                    chunk = await source.read(chunk_size)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("File exceeds the configured size limit.")
                    hasher.update(chunk)
                    target.write(chunk)
            temp_path.replace(path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            raise

        return size, hasher.hexdigest()

    def write_bytes(self, storage_key: str, data: bytes) -> None:
        path = self.path_for(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            temp_path.write_bytes(data)
            temp_path.replace(path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            raise

    def read_bytes(self, storage_key: str) -> bytes:
        return self.path_for(storage_key).read_bytes()

    def open(self, storage_key: str) -> BinaryIO:
        return self.path_for(storage_key).open("rb")

    def exists(self, storage_key: str) -> bool:
        return self.path_for(storage_key).exists()

    def delete(self, storage_key: str) -> None:
        self.path_for(storage_key).unlink(missing_ok=True)
