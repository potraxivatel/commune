"""
Component interface — the contract every Pyramid component must satisfy.

A component is anything that can process a task: an LLM wrapper,
a code parser, a retriever, a reviewer. The Engine only sees this
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from engine.communication.envelope import Envelope


class ComponentKind(str, Enum):
    LLM = "llm"
    CODE = "code"
    HYBRID = "hybrid"


class ComponentManifest(BaseModel):
    """Declares what a component is and what it needs."""

    name: str
    kind: ComponentKind
    description: str = ""
    version: str = "0.1.0"

    # Resource requirements (advisory, used by Scheduler)
    vram_mb: int = 0
    ram_mb: int = 0
    avg_latency_ms: float = 0.0

    # What content types this component can consume and produce
    accepts: list[str] = Field(default_factory=lambda: ["text"])
    produces: list[str] = Field(default_factory=lambda: ["text"])

    # Tags for routing (e.g. ["code-analysis", "python"])
    tags: list[str] = Field(default_factory=list)


class Component(ABC):
    """Base class for all Pyramid components."""

    def __init__(self) -> None:
        self._manifest: ComponentManifest | None = None

    @abstractmethod
    def manifest(self) -> ComponentManifest:
        """Return this component's manifest."""
        ...

    @abstractmethod
    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        """
        Process an incoming envelope and return a result envelope.

        Args:
            envelope: The input message.
            context: Optional dict with memory fragments, prior step
                     outputs, or engine-injected data.

        Returns:
            A new Envelope with the component's output.
        """
        ...

    async def load(self) -> None:
        """Load resources (model weights, caches). Called before first use."""

    async def unload(self) -> None:
        """Release resources. Called when Scheduler needs to free memory."""

    @property
    def name(self) -> str:
        return self.manifest().name
