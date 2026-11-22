"""
Memory records — the data structures for persistent memory.

Every piece of information stored in Pyramid memory is a MemoryRecord
with provenance (source, timestamp), category, and versioning.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


class MemoryCategory(str, Enum):
    PROJECT = "project"
    KNOWLEDGE = "knowledge"
    SESSION = "session"


class RecordStatus(str, Enum):
    ACTIVE = "active"
    OUTDATED = "outdated"
    DELETED = "deleted"


class Provenance(BaseModel):
    """Where a fact came from and when."""

    source: str  # e.g. "component:code-parser", "user", "file:/path"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    component: str | None = None  # which component produced this


class MemoryRecord(BaseModel):
    """One unit of persistent memory."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    category: MemoryCategory
    key: str  # lookup key, e.g. "project.structure", "fact.python.list_comprehension"
    content: str | dict | list  # the actual information
    tags: list[str] = Field(default_factory=list)
    provenance: Provenance
    status: RecordStatus = RecordStatus.ACTIVE
    version: int = 1

    @property
    def age_seconds(self) -> float:
        return time.time() - self.provenance.updated_at


class MemoryUpdate(BaseModel):
    """
    A proposed change to memory. Must be validated by MemoryStore
    before being applied.
    """

    key: str
    category: MemoryCategory
    content: str | dict | list
    source: str
    component: str | None = None
    tags: list[str] = Field(default_factory=list)
    # If set, this update targets an existing record
    target_id: str | None = None
