"""
Swarm message and role models for Commune AI multi-agent collaboration.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class SwarmMessage(BaseModel):
    """Represents a message in the internal swarm channel or to/from the user."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    sender: str
    role_title: str
    avatar: str = "🤖"
    recipient: str = "all"  # "all", "user", or specific agent name
    content: str
    is_internal: bool = True  # True = discussion between agents, False = user facing
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
