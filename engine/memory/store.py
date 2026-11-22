"""
Memory store — JSON-file-backed persistent memory with controlled writes.

Design:
  - Each category (project, knowledge, session) is a separate JSON file.
  - All writes go through validate → detect conflicts → apply.
  - Every record has provenance (source, timestamps).
  - Full change log for debugging.
  - Memory is independent of any LLM — pure Python, no model needed.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from engine.memory.records import (
    MemoryCategory,
    MemoryRecord,
    MemoryUpdate,
    Provenance,
    RecordStatus,
)

logger = logging.getLogger("pyramid.memory")


@dataclass
class ChangeLogEntry:
    """One entry in the change log."""

    timestamp: float
    action: str  # "add", "update", "outdated", "delete"
    record_id: str
    key: str
    source: str
    old_version: int | None = None
    new_version: int | None = None
    details: str = ""


class ConflictError(Exception):
    """Raised when a memory update conflicts with existing data."""

    def __init__(self, key: str, existing_id: str, message: str) -> None:
        self.key = key
        self.existing_id = existing_id
        super().__init__(message)


class MemoryStore:
    """
    JSON-file-backed persistent memory.

    Usage:
        store = MemoryStore("/path/to/memory")
        store.load()

        # Write
        record = store.add(MemoryUpdate(
            key="project.name",
            category=MemoryCategory.PROJECT,
            content="Pyramid",
            source="user",
        ))

        # Read
        result = store.get("project.name")
        results = store.query(category=MemoryCategory.PROJECT)
        results = store.search("pyramid")

        # Persist
        store.save()
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._records: dict[str, MemoryRecord] = {}
        self._changelog: list[ChangeLogEntry] = []

    # --- Persistence ---

    def load(self) -> int:
        """Load all records from disk. Returns count of records loaded."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        total = 0

        for category in MemoryCategory:
            path = self._category_path(category)
            if not path.exists():
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for raw in data:
                    record = MemoryRecord.model_validate(raw)
                    self._records[record.id] = record
                    total += 1
            except (json.JSONDecodeError, Exception) as exc:
                logger.error("Failed to load %s: %s", path, exc)

        logger.info("Loaded %d memory records from %s", total, self._base_dir)
        return total

    def save(self) -> None:
        """Persist all records to disk."""
        self._base_dir.mkdir(parents=True, exist_ok=True)

        for category in MemoryCategory:
            records = [
                r.model_dump() for r in self._records.values()
                if r.category == category
            ]
            path = self._category_path(category)

            # Backup before overwrite
            if path.exists():
                backup = path.with_suffix(".json.bak")
                shutil.copy2(path, backup)

            path.write_text(
                json.dumps(records, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        # Save changelog
        changelog_path = self._base_dir / "changelog.json"
        entries = [
            {
                "timestamp": e.timestamp,
                "action": e.action,
                "record_id": e.record_id,
                "key": e.key,
                "source": e.source,
                "old_version": e.old_version,
                "new_version": e.new_version,
                "details": e.details,
            }
            for e in self._changelog
        ]
        changelog_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        logger.info(
            "Saved %d records, %d changelog entries",
            len(self._records), len(self._changelog),
        )

    # --- Controlled writes ---

    def add(self, update: MemoryUpdate, force: bool = False) -> MemoryRecord:
        """
        Add or update a memory record through controlled validation.

        Steps:
          1. Validate structure
          2. Check for conflicts (existing record with same key)
          3. Create or update record
          4. Log the change

        Args:
            update: The proposed memory update.
            force: If True, overwrite existing record without conflict error.

        Returns:
            The created or updated MemoryRecord.

        Raises:
            ConflictError: If a record with the same key exists and force=False.
            ValueError: If the update is invalid.
        """
        # 1. Validate
        self._validate(update)

        # 2. Check for conflicts
        existing = self._find_by_key(update.key, update.category)

        if existing and not force and update.target_id is None:
            raise ConflictError(
                key=update.key,
                existing_id=existing.id,
                message=(
                    f"Record with key '{update.key}' already exists "
                    f"(id={existing.id}). Use force=True to overwrite, "
                    f"or set target_id to update explicitly."
                ),
            )

        # 3. Create or update
        if existing and (force or update.target_id == existing.id):
            return self._update_record(existing, update)
        elif update.target_id and update.target_id in self._records:
            return self._update_record(self._records[update.target_id], update)
        else:
            return self._create_record(update)

    def mark_outdated(self, record_id: str, source: str = "system") -> MemoryRecord | None:
        """Mark a record as outdated (soft delete)."""
        record = self._records.get(record_id)
        if record is None:
            return None

        old_status = record.status
        record.status = RecordStatus.OUTDATED
        record.provenance.updated_at = time.time()

        self._log_change(
            action="outdated",
            record=record,
            source=source,
            details=f"status: {old_status.value} → outdated",
        )

        return record

    def delete(self, record_id: str, source: str = "system") -> bool:
        """Mark a record as deleted (soft delete, keeps in store)."""
        record = self._records.get(record_id)
        if record is None:
            return False

        record.status = RecordStatus.DELETED
        record.provenance.updated_at = time.time()

        self._log_change(
            action="delete",
            record=record,
            source=source,
        )

        return True

    # --- Read operations ---

    def get(self, key: str, category: MemoryCategory | None = None) -> MemoryRecord | None:
        """Get an active record by key."""
        return self._find_by_key(key, category)

    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        """Get a record by its ID."""
        return self._records.get(record_id)

    def query(
        self,
        category: MemoryCategory | None = None,
        tags: list[str] | None = None,
        status: RecordStatus = RecordStatus.ACTIVE,
        key_prefix: str | None = None,
    ) -> list[MemoryRecord]:
        """Query records with filters."""
        results = []
        for record in self._records.values():
            if record.status != status:
                continue
            if category and record.category != category:
                continue
            if key_prefix and not record.key.startswith(key_prefix):
                continue
            if tags and not any(t in record.tags for t in tags):
                continue
            results.append(record)

        return sorted(results, key=lambda r: r.provenance.updated_at, reverse=True)

    def search(self, text: str, category: MemoryCategory | None = None) -> list[MemoryRecord]:
        """Simple text search across keys and content."""
        text_lower = text.lower()
        results = []

        for record in self._records.values():
            if record.status != RecordStatus.ACTIVE:
                continue
            if category and record.category != category:
                continue

            # Search in key
            if text_lower in record.key.lower():
                results.append(record)
                continue

            # Search in content
            content_str = str(record.content).lower()
            if text_lower in content_str:
                results.append(record)
                continue

            # Search in tags
            if any(text_lower in tag.lower() for tag in record.tags):
                results.append(record)

        return results

    def all_active(self) -> list[MemoryRecord]:
        """Get all active records."""
        return self.query(status=RecordStatus.ACTIVE)

    @property
    def count(self) -> int:
        """Count of active records."""
        return sum(1 for r in self._records.values() if r.status == RecordStatus.ACTIVE)

    @property
    def changelog(self) -> list[ChangeLogEntry]:
        return list(self._changelog)

    # --- Internals ---

    def _validate(self, update: MemoryUpdate) -> None:
        """Validate an update before applying."""
        if not update.key or not update.key.strip():
            raise ValueError("Memory key cannot be empty")
        if update.content is None:
            raise ValueError("Memory content cannot be None")
        # Key format: dot-separated, e.g. "project.structure.files"
        if not all(part.strip() for part in update.key.split(".")):
            raise ValueError(f"Invalid key format: '{update.key}'")

    def _find_by_key(
        self, key: str, category: MemoryCategory | None = None
    ) -> MemoryRecord | None:
        """Find an active record by key."""
        for record in self._records.values():
            if record.status != RecordStatus.ACTIVE:
                continue
            if record.key == key:
                if category is None or record.category == category:
                    return record
        return None

    def _create_record(self, update: MemoryUpdate) -> MemoryRecord:
        """Create a new record from an update."""
        record = MemoryRecord(
            category=update.category,
            key=update.key,
            content=update.content,
            tags=update.tags,
            provenance=Provenance(
                source=update.source,
                component=update.component,
            ),
        )
        self._records[record.id] = record

        self._log_change(action="add", record=record, source=update.source)
        logger.info("Created memory: [%s] %s", record.category.value, record.key)

        return record

    def _update_record(self, existing: MemoryRecord, update: MemoryUpdate) -> MemoryRecord:
        """Update an existing record."""
        old_version = existing.version

        existing.content = update.content
        existing.tags = update.tags if update.tags else existing.tags
        existing.provenance.updated_at = time.time()
        existing.provenance.source = update.source
        if update.component:
            existing.provenance.component = update.component
        existing.version += 1
        existing.status = RecordStatus.ACTIVE  # reactivate if was outdated

        self._log_change(
            action="update",
            record=existing,
            source=update.source,
            old_version=old_version,
            details=f"v{old_version} → v{existing.version}",
        )
        logger.info(
            "Updated memory: [%s] %s (v%d)",
            existing.category.value, existing.key, existing.version,
        )

        return existing

    def _log_change(
        self,
        action: str,
        record: MemoryRecord,
        source: str,
        old_version: int | None = None,
        details: str = "",
    ) -> None:
        """Add an entry to the changelog."""
        entry = ChangeLogEntry(
            timestamp=time.time(),
            action=action,
            record_id=record.id,
            key=record.key,
            source=source,
            old_version=old_version,
            new_version=record.version,
            details=details,
        )
        self._changelog.append(entry)

    def _category_path(self, category: MemoryCategory) -> Path:
        """Get the JSON file path for a category."""
        return self._base_dir / f"{category.value}.json"
