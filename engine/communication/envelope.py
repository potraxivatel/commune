"""
Communication envelope — the typed message format for all
inter-component data exchange within Pyramid.

Every message between components uses this envelope.
Free-form text is a value inside `content`, not the envelope itself.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    PARTIAL = "partial"


class ContentType(str, Enum):
    TEXT = "text"
    JSON = "json"
    CODE = "code"
    BINARY_REF = "binary_ref"


class MessageMetadata(BaseModel):
    token_count: int = 0
    latency_ms: float = 0.0
    confidence: float | None = None
    model_name: str | None = None


class Envelope(BaseModel):
    """Typed envelope for inter-component communication."""

    task_id: str
    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender: str
    status: MessageStatus = MessageStatus.OK
    content_type: ContentType = ContentType.TEXT
    content: Any = ""
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def error(cls, task_id: str, sender: str, message: str) -> Envelope:
        return cls(
            task_id=task_id,
            sender=sender,
            status=MessageStatus.ERROR,
            content_type=ContentType.TEXT,
            content=message,
        )
