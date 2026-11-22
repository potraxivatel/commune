"""
Filesystem tools for Commune AI.
Provides ReadFileComponent, WriteFileComponent, and ListDirComponent.
"""

from __future__ import annotations

import fnmatch
import os
import time
from pathlib import Path
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageMetadata, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest


class ReadFileComponent(Component):
    """Reads content from a local file with optional line slicing."""

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name="read_file",
            kind=ComponentKind.CODE,
            description="Reads content from a local file with optional line offset and limit",
            version="0.2.0",
            accepts=["text", "json"],
            produces=["text", "code"],
            tags=["filesystem", "tool", "read"],
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        start_time = time.time()
        task_id = envelope.task_id

        # Parse parameters
        path_str = ""
        offset = 0
        limit: int | None = None

        if isinstance(envelope.content, dict):
            path_str = str(envelope.content.get("path", ""))
            offset = int(envelope.content.get("offset", 0))
            limit = envelope.content.get("limit")
            if limit is not None:
                limit = int(limit)
        elif isinstance(envelope.content, str):
            path_str = envelope.content.strip()

        if not path_str:
            return Envelope.error(task_id, "read_file", "No file path provided in envelope content")

        file_path = Path(path_str).resolve()
        if not file_path.exists():
            return Envelope.error(task_id, "read_file", f"File does not exist: {file_path}")
        if not file_path.is_file():
            return Envelope.error(task_id, "read_file", f"Path is not a regular file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            if offset < 0:
                offset = 0
            if offset >= total_lines and total_lines > 0:
                selected_lines: list[str] = []
            else:
                end = (offset + limit) if limit is not None else total_lines
                selected_lines = lines[offset:end]

            content_text = "".join(selected_lines)
            duration_ms = (time.time() - start_time) * 1000.0

            return Envelope(
                task_id=task_id,
                sender="read_file",
                status=MessageStatus.OK,
                content_type=ContentType.CODE if file_path.suffix in {".py", ".sh", ".json", ".md", ".toml"} else ContentType.TEXT,
                content=content_text,
                metadata=MessageMetadata(
                    token_count=len(content_text.split()),
                    latency_ms=duration_ms,
                ),
            )
        except Exception as e:
            return Envelope.error(task_id, "read_file", f"Failed to read {file_path}: {e}")


class WriteFileComponent(Component):
    """Safely writes content to a file, creating parent directories if needed."""

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name="write_file",
            kind=ComponentKind.CODE,
            description="Writes content to a local file, creating parent directories as necessary",
            version="0.2.0",
            accepts=["text", "json"],
            produces=["json"],
            tags=["filesystem", "tool", "write"],
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        start_time = time.time()
        task_id = envelope.task_id

        if not isinstance(envelope.content, dict):
            return Envelope.error(task_id, "write_file", "Envelope content must be a dict with 'path' and 'content'")

        path_str = str(envelope.content.get("path", "")).strip()
        data = envelope.content.get("content", "")
        append = bool(envelope.content.get("append", False))

        if not path_str:
            return Envelope.error(task_id, "write_file", "No 'path' specified in payload")

        file_path = Path(path_str).resolve()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(data)

            bytes_written = len(data.encode("utf-8"))
            duration_ms = (time.time() - start_time) * 1000.0

            return Envelope(
                task_id=task_id,
                sender="write_file",
                status=MessageStatus.OK,
                content_type=ContentType.JSON,
                content={
                    "path": str(file_path),
                    "bytes_written": bytes_written,
                    "append": append,
                    "success": True,
                },
                metadata=MessageMetadata(latency_ms=duration_ms),
            )
        except Exception as e:
            return Envelope.error(task_id, "write_file", f"Failed to write to {file_path}: {e}")


class ListDirComponent(Component):
    """Lists files and folders with pattern matching and depth control."""

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name="list_dir",
            kind=ComponentKind.CODE,
            description="Lists directory entries with optional glob filtering",
            version="0.2.0",
            accepts=["text", "json"],
            produces=["json"],
            tags=["filesystem", "tool", "list"],
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        start_time = time.time()
        task_id = envelope.task_id

        path_str = "."
        pattern = "*"
        recursive = False
        max_depth = 2

        if isinstance(envelope.content, dict):
            path_str = str(envelope.content.get("path", "."))
            pattern = str(envelope.content.get("pattern", "*"))
            recursive = bool(envelope.content.get("recursive", False))
            max_depth = int(envelope.content.get("max_depth", 2))
        elif isinstance(envelope.content, str) and envelope.content.strip():
            path_str = envelope.content.strip()

        dir_path = Path(path_str).resolve()
        if not dir_path.exists():
            return Envelope.error(task_id, "list_dir", f"Directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            return Envelope.error(task_id, "list_dir", f"Path is not a directory: {dir_path}")

        try:
            results: list[dict[str, Any]] = []

            if not recursive:
                for entry in sorted(dir_path.iterdir()):
                    if fnmatch.fnmatch(entry.name, pattern):
                        results.append({
                            "name": entry.name,
                            "path": str(entry),
                            "is_dir": entry.is_dir(),
                            "size": entry.stat().st_size if entry.is_file() else 0,
                        })
            else:
                base_depth = len(dir_path.parts)
                for root, dirs, files in os.walk(dir_path):
                    current_depth = len(Path(root).parts) - base_depth
                    if current_depth > max_depth:
                        continue
                    # Filter out hidden or build directories
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                    for f in files:
                        if fnmatch.fnmatch(f, pattern):
                            full_p = Path(root) / f
                            rel_p = full_p.relative_to(dir_path)
                            results.append({
                                "name": str(rel_p),
                                "path": str(full_p),
                                "is_dir": False,
                                "size": full_p.stat().st_size if full_p.exists() else 0,
                            })

            duration_ms = (time.time() - start_time) * 1000.0
            return Envelope(
                task_id=task_id,
                sender="list_dir",
                status=MessageStatus.OK,
                content_type=ContentType.JSON,
                content={
                    "directory": str(dir_path),
                    "total_entries": len(results),
                    "entries": results,
                },
                metadata=MessageMetadata(latency_ms=duration_ms),
            )
        except Exception as e:
            return Envelope.error(task_id, "list_dir", f"Failed to list {dir_path}: {e}")
