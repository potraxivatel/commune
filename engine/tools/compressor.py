"""
Stage 2 Context Compressor tool for Commune AI.
Provides CompressorComponent for reducing noise in logs, compiler output, and large diffs.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageMetadata, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest
from engine.inference.llama_backend import LlamaComponent

SYSTEM_COMPRESSOR_PROMPT = (
    "You are a Stage 2 Context Compressor. Given raw terminal output, stack traces, "
    "or compiler dumps, extract only the essential facts, failing lines, errors, "
    "and test outcomes into concise bullet points. Eliminate all repetitive boilerplate."
)


class CompressorComponent(Component):
    """
    Compresses raw tool output (e.g. from run_command or read_file)
    using the fine-tuned commune_compressor GGUF model or a deterministic fallback.
    """

    def __init__(
        self,
        name: str = "stage2_compressor",
        model_path: str | Path | None = None,
        min_chars_to_compress: int = 250,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
    ) -> None:
        super().__init__()
        self._name = name
        self._min_chars = min_chars_to_compress

        if model_path is None:
            # Default to local custom model if exists
            default_p = Path(__file__).resolve().parent.parent.parent / "llms" / "custom" / "commune_compressor.Q4_K_M.gguf"
            if default_p.exists():
                model_path = default_p

        self._model_path = Path(model_path) if model_path else None
        self._llama: LlamaComponent | None = None
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.HYBRID,
            description="Compresses verbose tool outputs, stack traces, and logs down to core signals",
            version="0.2.0",
            accepts=["text", "json", "code"],
            produces=["text"],
            tags=["compression", "context", "stage2"],
        )

    async def load(self) -> bool:
        """Attempt to load the underlying GGUF model."""
        if self._llama is not None:
            return True
        if self._model_path and self._model_path.exists():
            try:
                self._llama = LlamaComponent(
                    name=f"{self._name}_llama",
                    model_path=self._model_path,
                    system_prompt=SYSTEM_COMPRESSOR_PROMPT,
                    n_ctx=self._n_ctx,
                    n_gpu_layers=self._n_gpu_layers,
                    temperature=0.2,
                    max_tokens=512,
                )
                await self._llama.load()
                return True
            except Exception:
                self._llama = None
                return False
        return False

    async def unload(self) -> None:
        if self._llama is not None:
            await self._llama.unload()
            self._llama = None

    def _deterministic_compress(self, raw_text: str) -> str:
        """Heuristic fallback compressor when LLM model is not active."""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return ""

        extracted: list[str] = []
        in_traceback = False

        for line in lines:
            # Capture error headers & exceptions
            if "Traceback (most recent call last):" in line:
                in_traceback = True
                extracted.append("TRACEBACK:")
                continue
            if in_traceback:
                if line.startswith("File "):
                    extracted.append(f"  {line}")
                elif any(err in line for err in ("Error:", "Exception:", "Failed:", "CRITICAL")):
                    extracted.append(f"  ERROR: {line}")
                    in_traceback = False
                continue

            # Capture pytest/test summaries
            if any(k in line.lower() for k in ("passed in", "failed in", "errors in", "warnings in", "=== short test summary")):
                extracted.append(f"TEST_SUMMARY: {line}")
                continue

            # Capture compiler / warning indicators
            if any(k in line for k in ("error:", "fatal:", "warning:", "syntax error", "FAILED")):
                extracted.append(line)

        if not extracted:
            # If no obvious patterns matched, keep head and tail lines
            if len(lines) > 10:
                extracted = lines[:4] + [f"... [{len(lines) - 8} lines omitted] ..."] + lines[-4:]
            else:
                extracted = lines

        return "\n".join(extracted)

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        start_time = time.time()
        task_id = envelope.task_id

        # Extract text from envelope content (supports raw strings or run_command dicts)
        raw_text = ""
        if isinstance(envelope.content, dict):
            stdout = envelope.content.get("stdout", "")
            stderr = envelope.content.get("stderr", "")
            exit_code = envelope.content.get("exit_code", 0)
            parts = []
            if exit_code != 0:
                parts.append(f"[Exit code: {exit_code}]")
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[STDERR]\n{stderr}")
            raw_text = "\n".join(parts)
        else:
            raw_text = str(envelope.content)

        orig_len = len(raw_text)

        # Skip compression if already compact
        if orig_len < self._min_chars:
            duration_ms = (time.time() - start_time) * 1000.0
            return Envelope(
                task_id=task_id,
                sender=self._name,
                status=MessageStatus.OK,
                content_type=ContentType.TEXT,
                content=raw_text,
                metadata=MessageMetadata(
                    token_count=len(raw_text.split()),
                    latency_ms=duration_ms,
                    confidence=1.0,
                ),
            )

        compressed_text = ""
        used_model = False

        if self._llama is not None:
            try:
                llm_env = await self._llama.process(
                    Envelope(
                        task_id=task_id,
                        sender=self._name,
                        content_type=ContentType.TEXT,
                        content=f"Compress the following raw output:\n\n{raw_text}",
                    )
                )
                if llm_env.status == MessageStatus.OK and llm_env.content:
                    compressed_text = str(llm_env.content).strip()
                    used_model = True
            except Exception:
                compressed_text = ""

        if not compressed_text:
            compressed_text = self._deterministic_compress(raw_text)

        comp_len = len(compressed_text)
        ratio = round(orig_len / max(comp_len, 1), 2)
        duration_ms = (time.time() - start_time) * 1000.0

        return Envelope(
            task_id=task_id,
            sender=self._name,
            status=MessageStatus.OK,
            content_type=ContentType.TEXT,
            content=compressed_text,
            metadata=MessageMetadata(
                token_count=len(compressed_text.split()),
                latency_ms=duration_ms,
                model_name=self._model_path.name if (used_model and self._model_path) else "heuristic_compressor",
                confidence=ratio,
            ),
        )
