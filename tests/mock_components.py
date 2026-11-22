"""
Mock components for testing the engine without real LLMs.
"""

from __future__ import annotations

import time
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageMetadata, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest


class EchoComponent(Component):
    """Echoes input back. Simplest possible component for testing."""

    def __init__(self, name: str = "echo") -> None:
        super().__init__()
        self._name = name

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.CODE,
            description="Echoes input",
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        return Envelope(
            task_id=envelope.task_id,
            sender=self._name,
            status=MessageStatus.OK,
            content_type=envelope.content_type,
            content=f"[echo from {self._name}]: {envelope.content}",
            metadata=MessageMetadata(latency_ms=0.1),
        )


class UpperComponent(Component):
    """Uppercases text content. For testing chained components."""

    def __init__(self, name: str = "upper") -> None:
        super().__init__()
        self._name = name

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.CODE,
            description="Uppercases text",
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        text = str(envelope.content).upper()
        return Envelope(
            task_id=envelope.task_id,
            sender=self._name,
            status=MessageStatus.OK,
            content_type=ContentType.TEXT,
            content=text,
            metadata=MessageMetadata(latency_ms=0.1),
        )


class WordCountComponent(Component):
    """Counts words. Demonstrates a pure-code analytical component."""

    def __init__(self, name: str = "word-counter") -> None:
        super().__init__()
        self._name = name

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.CODE,
            description="Counts words in text",
            tags=["analysis"],
            produces=["json"],
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        text = str(envelope.content)
        words = text.split()
        result = {
            "word_count": len(words),
            "char_count": len(text),
            "line_count": text.count("\n") + 1,
        }
        return Envelope(
            task_id=envelope.task_id,
            sender=self._name,
            status=MessageStatus.OK,
            content_type=ContentType.JSON,
            content=result,
            metadata=MessageMetadata(latency_ms=0.1),
        )


class SlowComponent(Component):
    """Artificially slow component for timeout testing."""

    def __init__(self, name: str = "slow", delay: float = 5.0) -> None:
        super().__init__()
        self._name = name
        self._delay = delay

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.CODE,
            description=f"Waits {self._delay}s (for testing)",
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        import asyncio
        await asyncio.sleep(self._delay)
        return Envelope(
            task_id=envelope.task_id,
            sender=self._name,
            status=MessageStatus.OK,
            content_type=ContentType.TEXT,
            content=f"Finished after {self._delay}s",
            metadata=MessageMetadata(latency_ms=self._delay * 1000),
        )


class FailComponent(Component):
    """Always fails. For error handling tests."""

    def __init__(self, name: str = "fail") -> None:
        super().__init__()
        self._name = name

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.CODE,
            description="Always raises an error",
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        raise RuntimeError("Intentional failure for testing")
