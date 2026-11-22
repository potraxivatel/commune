"""
llama.cpp inference backend via llama-cpp-python.

Wraps a GGUF model file into a Pyramid Component.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageMetadata, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest

logger = logging.getLogger("pyramid.inference.llama")


class LlamaComponent(Component):
    """
    A Pyramid component backed by a local GGUF model via llama-cpp-python.

    Usage:
        comp = LlamaComponent(
            name="analyst-7b",
            model_path="/path/to/model.gguf",
            system_prompt="You are a code analyst.",
            n_ctx=4096,
            n_gpu_layers=-1,  # offload all to GPU
        )
    """

    def __init__(
        self,
        name: str,
        model_path: str | Path,
        system_prompt: str = "",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        repeat_penalty: float = 1.18,
        tags: list[str] | None = None,
        description: str = "",
    ) -> None:
        super().__init__()
        self._name = name
        self._model_path = Path(model_path)
        self._system_prompt = system_prompt
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._repeat_penalty = repeat_penalty
        self._tags = tags or []
        self._description = description
        self._llm = None

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.LLM,
            description=self._description or f"LLM component: {self._name}",
            vram_mb=0,  # will be measured after load
            tags=self._tags,
            accepts=["text", "code"],
            produces=["text", "code", "json"],
        )

    async def load(self) -> None:
        if self._llm is not None:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        try:
            from llama_cpp import Llama

            logger.info("Loading model: %s", self._model_path.name)
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            logger.info("Model loaded: %s", self._name)
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install: pip install llama-cpp-python"
            )

    async def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("Model unloaded: %s", self._name)

    async def process(
        self, envelope: Envelope, context: dict[str, Any] | None = None
    ) -> Envelope:
        if self._llm is None:
            raise RuntimeError(f"Model not loaded: {self._name}. Call load() first.")

        # Build messages
        active_system = self._system_prompt
        if context and "system_prompt" in context:
            active_system = str(context["system_prompt"])

        messages = []
        if active_system:
            messages.append({"role": "system", "content": active_system})

        # Inject context from prior steps if available
        if context and "prior_outputs" in context:
            ctx_parts = []
            for comp_name, env in context["prior_outputs"].items():
                ctx_parts.append(f"[{comp_name}]: {env.content}")
            if ctx_parts:
                messages.append({
                    "role": "system",
                    "content": "Context from prior analysis:\n" + "\n\n".join(ctx_parts),
                })

        messages.append({"role": "user", "content": str(envelope.content)})

        # Generate
        t0 = time.time()
        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            repeat_penalty=self._repeat_penalty,
        )
        latency = (time.time() - t0) * 1000

        # Extract result
        choice = response["choices"][0]
        text = choice["message"]["content"] or ""
        usage = response.get("usage", {})

        return Envelope(
            task_id=envelope.task_id,
            sender=self._name,
            status=MessageStatus.OK,
            content_type=ContentType.TEXT,
            content=text.strip(),
            metadata=MessageMetadata(
                token_count=usage.get("completion_tokens", 0),
                latency_ms=latency,
                model_name=self._model_path.name,
            ),
        )
