"""
Codebase Investigation & Staged Aggregation Pipeline for Commune AI.

Executes the 3-stage hypothesis:
  Stage 1 (Tools): Gather raw information via shell/filesystem components.
  Stage 2 (Compressor): Compress raw traces and outputs via specialized micro-model.
  Stage 3 (Memory / Synthesis): Extract structured records and update MemoryStore.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageMetadata, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest
from engine.memory.records import MemoryCategory, MemoryUpdate
from engine.memory.store import MemoryStore
from engine.tools.compressor import CompressorComponent
from engine.tools.filesystem import ListDirComponent, ReadFileComponent
from engine.tools.terminal import RunCommandComponent


class CodebasePipeline(Component):
    """
    Orchestrates a 3-stage codebase analysis workflow:
    1. Inspection (Command execution or file reading).
    2. Compression (Stage 2 Compressor reduces context footprint).
    3. Memory Persistence (Updates project/knowledge store).
    """

    def __init__(
        self,
        name: str = "codebase_pipeline",
        memory_store: MemoryStore | None = None,
        compressor: CompressorComponent | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._store = memory_store or MemoryStore(Path.cwd() / "memory")
        self._compressor = compressor or CompressorComponent()
        self._cmd_tool = RunCommandComponent()
        self._read_tool = ReadFileComponent()
        self._list_tool = ListDirComponent()

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.HYBRID,
            description="3-stage workflow: tool execution -> context compression -> memory integration",
            version="0.2.0",
            accepts=["text", "json"],
            produces=["json", "text"],
            tags=["workflow", "pipeline", "staged-aggregation"],
        )

    async def execute_tool_and_compress(
        self,
        tool_name: str,
        params: dict[str, Any] | str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Executes a deterministic tool and pipes output through Stage 2 Compressor.
        Returns a dictionary containing raw envelope, compressed envelope, and timing.
        """
        tid = task_id or f"task_{uuid.uuid4().hex[:8]}"

        # Stage 1: Tool Execution
        t0 = time.time()
        in_env = Envelope(
            task_id=tid,
            sender="pipeline",
            content_type=ContentType.JSON if isinstance(params, dict) else ContentType.TEXT,
            content=params,
        )

        if tool_name == "run_command":
            raw_env = await self._cmd_tool.process(in_env)
        elif tool_name == "read_file":
            raw_env = await self._read_tool.process(in_env)
        elif tool_name == "list_dir":
            raw_env = await self._list_tool.process(in_env)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

        stage1_time = (time.time() - t0) * 1000.0

        # Stage 2: Staged Compression
        t1 = time.time()
        comp_env = await self._compressor.process(raw_env)
        stage2_time = (time.time() - t1) * 1000.0

        # Stage 3: Auto-persist summary into memory if there's actionable signal
        raw_content_str = str(raw_env.content)
        comp_content_str = str(comp_env.content)
        orig_chars = len(raw_content_str)
        comp_chars = len(comp_content_str)
        ratio = round(orig_chars / max(comp_chars, 1), 2)

        # Update knowledge memory store
        update = MemoryUpdate(
            key=f"tool_run_{tool_name}_{int(time.time())}",
            category=MemoryCategory.KNOWLEDGE,
            content={
                "tool": tool_name,
                "summary": comp_content_str,
                "compression_ratio": ratio,
            },
            source=f"pipeline:{tool_name}",
            component="commune_pipeline",
            tags=["tool_run", tool_name],
        )
        self._store.add(update)
        self._store.save()

        return {
            "task_id": tid,
            "tool": tool_name,
            "stage1_raw": raw_env,
            "stage2_compressed": comp_env,
            "metrics": {
                "stage1_ms": stage1_time,
                "stage2_ms": stage2_time,
                "original_chars": orig_chars,
                "compressed_chars": comp_chars,
                "compression_ratio": ratio,
            },
        }

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        task_id = envelope.task_id
        content = envelope.content

        tool_name = "run_command"
        params: Any = content

        if isinstance(content, dict):
            tool_name = content.get("tool", "run_command")
            params = content.get("params", content)

        res = await self.execute_tool_and_compress(tool_name, params, task_id=task_id)

        return Envelope(
            task_id=task_id,
            sender=self._name,
            status=MessageStatus.OK,
            content_type=ContentType.JSON,
            content=res,
            metadata=MessageMetadata(
                latency_ms=res["metrics"]["stage1_ms"] + res["metrics"]["stage2_ms"],
                confidence=res["metrics"]["compression_ratio"],
            ),
        )
