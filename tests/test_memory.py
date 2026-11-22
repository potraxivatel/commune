"""
Tests for Pyramid Memory System — Phase 2.

Covers:
  - Record creation and provenance
  - Controlled writes (add, update, conflict detection)
  - Read operations (get, query, search)
  - Persistence (save/load roundtrip)
  - Soft delete and outdating
  - Changelog
  - Validation
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.memory.records import MemoryCategory, MemoryRecord, MemoryUpdate, RecordStatus
from engine.memory.store import ConflictError, MemoryStore


@pytest.fixture
def tmp_memory(tmp_path):
    """Create a fresh MemoryStore in a temp directory."""
    store = MemoryStore(tmp_path / "memory")
    store.load()
    return store


# --- Record creation ---

class TestRecordCreation:
    def test_create_record(self, tmp_memory):
        record = tmp_memory.add(MemoryUpdate(
            key="project.name",
            category=MemoryCategory.PROJECT,
            content="Pyramid",
            source="user",
        ))
        assert record.key == "project.name"
        assert record.content == "Pyramid"
        assert record.version == 1
        assert record.status == RecordStatus.ACTIVE
        assert record.provenance.source == "user"

    def test_create_with_tags(self, tmp_memory):
        record = tmp_memory.add(MemoryUpdate(
            key="fact.python.lists",
            category=MemoryCategory.KNOWLEDGE,
            content="Python lists are mutable ordered sequences",
            source="component:analyzer",
            component="analyzer",
            tags=["python", "data-structures"],
        ))
        assert "python" in record.tags
        assert record.provenance.component == "analyzer"

    def test_create_with_dict_content(self, tmp_memory):
        record = tmp_memory.add(MemoryUpdate(
            key="project.structure",
            category=MemoryCategory.PROJECT,
            content={"files": ["main.py", "utils.py"], "language": "python"},
            source="component:code-parser",
        ))
        assert isinstance(record.content, dict)
        assert record.content["language"] == "python"


# --- Controlled writes ---

class TestControlledWrites:
    def test_conflict_detection(self, tmp_memory):
        tmp_memory.add(MemoryUpdate(
            key="project.name",
            category=MemoryCategory.PROJECT,
            content="Pyramid",
            source="user",
        ))

        with pytest.raises(ConflictError) as exc_info:
            tmp_memory.add(MemoryUpdate(
                key="project.name",
                category=MemoryCategory.PROJECT,
                content="Pyramid v2",
                source="user",
            ))

        assert "already exists" in str(exc_info.value)

    def test_force_overwrite(self, tmp_memory):
        r1 = tmp_memory.add(MemoryUpdate(
            key="project.name",
            category=MemoryCategory.PROJECT,
            content="Pyramid",
            source="user",
        ))
        r2 = tmp_memory.add(
            MemoryUpdate(
                key="project.name",
                category=MemoryCategory.PROJECT,
                content="Pyramid v2",
                source="user",
            ),
            force=True,
        )
        assert r2.content == "Pyramid v2"
        assert r2.version == 2
        assert r2.id == r1.id  # same record, updated

    def test_explicit_update_by_id(self, tmp_memory):
        r1 = tmp_memory.add(MemoryUpdate(
            key="project.name",
            category=MemoryCategory.PROJECT,
            content="Pyramid",
            source="user",
        ))
        r2 = tmp_memory.add(MemoryUpdate(
            key="project.name",
            category=MemoryCategory.PROJECT,
            content="Updated Pyramid",
            source="system",
            target_id=r1.id,
        ))
        assert r2.version == 2
        assert r2.content == "Updated Pyramid"
        assert r2.provenance.source == "system"

    def test_validation_empty_key(self, tmp_memory):
        with pytest.raises(ValueError, match="empty"):
            tmp_memory.add(MemoryUpdate(
                key="",
                category=MemoryCategory.PROJECT,
                content="test",
                source="user",
            ))

    def test_validation_bad_key_format(self, tmp_memory):
        with pytest.raises(ValueError, match="Invalid key"):
            tmp_memory.add(MemoryUpdate(
                key="project..name",
                category=MemoryCategory.PROJECT,
                content="test",
                source="user",
            ))


# --- Read operations ---

class TestReadOperations:
    def _populate(self, store):
        store.add(MemoryUpdate(
            key="project.name", category=MemoryCategory.PROJECT,
            content="Pyramid", source="user",
        ))
        store.add(MemoryUpdate(
            key="project.language", category=MemoryCategory.PROJECT,
            content="Python", source="user", tags=["python"],
        ))
        store.add(MemoryUpdate(
            key="fact.python.gc", category=MemoryCategory.KNOWLEDGE,
            content="Python uses reference counting with cycle detection",
            source="component:analyzer", tags=["python", "memory"],
        ))
        store.add(MemoryUpdate(
            key="session.current_task", category=MemoryCategory.SESSION,
            content="Building memory system", source="system",
        ))

    def test_get_by_key(self, tmp_memory):
        self._populate(tmp_memory)
        record = tmp_memory.get("project.name")
        assert record is not None
        assert record.content == "Pyramid"

    def test_get_nonexistent(self, tmp_memory):
        assert tmp_memory.get("nonexistent") is None

    def test_query_by_category(self, tmp_memory):
        self._populate(tmp_memory)
        results = tmp_memory.query(category=MemoryCategory.PROJECT)
        assert len(results) == 2

    def test_query_by_tags(self, tmp_memory):
        self._populate(tmp_memory)
        results = tmp_memory.query(tags=["python"])
        assert len(results) == 2  # project.language + fact.python.gc

    def test_query_by_key_prefix(self, tmp_memory):
        self._populate(tmp_memory)
        results = tmp_memory.query(key_prefix="project.")
        assert len(results) == 2

    def test_search_in_content(self, tmp_memory):
        self._populate(tmp_memory)
        results = tmp_memory.search("reference counting")
        assert len(results) == 1
        assert results[0].key == "fact.python.gc"

    def test_search_in_key(self, tmp_memory):
        self._populate(tmp_memory)
        results = tmp_memory.search("project.name")
        assert len(results) == 1

    def test_search_in_tags(self, tmp_memory):
        self._populate(tmp_memory)
        results = tmp_memory.search("memory")
        assert len(results) >= 1

    def test_count(self, tmp_memory):
        self._populate(tmp_memory)
        assert tmp_memory.count == 4


# --- Persistence ---

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        mem_dir = tmp_path / "memory"

        # Create and save
        store1 = MemoryStore(mem_dir)
        store1.load()
        store1.add(MemoryUpdate(
            key="project.name", category=MemoryCategory.PROJECT,
            content="Pyramid", source="user", tags=["core"],
        ))
        store1.add(MemoryUpdate(
            key="fact.test", category=MemoryCategory.KNOWLEDGE,
            content={"info": "test data"}, source="test",
        ))
        store1.save()

        # Load into a new store
        store2 = MemoryStore(mem_dir)
        loaded = store2.load()
        assert loaded == 2

        record = store2.get("project.name")
        assert record is not None
        assert record.content == "Pyramid"
        assert "core" in record.tags
        assert record.provenance.source == "user"

        # Dict content survives roundtrip
        fact = store2.get("fact.test")
        assert fact.content["info"] == "test data"

    def test_category_files_created(self, tmp_path):
        mem_dir = tmp_path / "memory"
        store = MemoryStore(mem_dir)
        store.load()
        store.add(MemoryUpdate(
            key="p.name", category=MemoryCategory.PROJECT,
            content="test", source="test",
        ))
        store.add(MemoryUpdate(
            key="k.fact", category=MemoryCategory.KNOWLEDGE,
            content="test", source="test",
        ))
        store.save()

        assert (mem_dir / "project.json").exists()
        assert (mem_dir / "knowledge.json").exists()
        assert (mem_dir / "changelog.json").exists()

    def test_backup_on_save(self, tmp_path):
        mem_dir = tmp_path / "memory"
        store = MemoryStore(mem_dir)
        store.load()
        store.add(MemoryUpdate(
            key="p.x", category=MemoryCategory.PROJECT,
            content="v1", source="test",
        ))
        store.save()  # first save

        store.add(
            MemoryUpdate(
                key="p.x", category=MemoryCategory.PROJECT,
                content="v2", source="test",
            ),
            force=True,
        )
        store.save()  # second save creates backup

        assert (mem_dir / "project.json.bak").exists()

    def test_survives_restart(self, tmp_path):
        """Memory must survive application restart — the core Phase 2 requirement."""
        mem_dir = tmp_path / "memory"

        # Session 1: create memory
        store = MemoryStore(mem_dir)
        store.load()
        store.add(MemoryUpdate(
            key="project.name", category=MemoryCategory.PROJECT,
            content="Pyramid", source="user",
        ))
        store.save()
        del store  # simulate shutdown

        # Session 2: memory should be there
        store2 = MemoryStore(mem_dir)
        store2.load()
        assert store2.get("project.name") is not None
        assert store2.get("project.name").content == "Pyramid"


# --- Soft delete and outdating ---

class TestLifecycle:
    def test_mark_outdated(self, tmp_memory):
        record = tmp_memory.add(MemoryUpdate(
            key="old.fact", category=MemoryCategory.KNOWLEDGE,
            content="Python 2 is current", source="old-source",
        ))
        tmp_memory.mark_outdated(record.id, source="user")

        assert record.status == RecordStatus.OUTDATED
        # Should not appear in normal queries
        assert tmp_memory.get("old.fact") is None
        assert tmp_memory.count == 0

    def test_delete_record(self, tmp_memory):
        record = tmp_memory.add(MemoryUpdate(
            key="temp.data", category=MemoryCategory.SESSION,
            content="temporary", source="system",
        ))
        assert tmp_memory.delete(record.id)
        assert tmp_memory.get("temp.data") is None

    def test_reactivate_outdated(self, tmp_memory):
        record = tmp_memory.add(MemoryUpdate(
            key="fact.x", category=MemoryCategory.KNOWLEDGE,
            content="old", source="user",
        ))
        tmp_memory.mark_outdated(record.id)

        # Update should reactivate
        tmp_memory.add(MemoryUpdate(
            key="fact.x", category=MemoryCategory.KNOWLEDGE,
            content="new and correct", source="user",
            target_id=record.id,
        ))
        assert record.status == RecordStatus.ACTIVE
        assert record.content == "new and correct"


# --- Changelog ---

class TestChangelog:
    def test_changelog_tracks_adds(self, tmp_memory):
        tmp_memory.add(MemoryUpdate(
            key="p.x", category=MemoryCategory.PROJECT,
            content="test", source="user",
        ))
        assert len(tmp_memory.changelog) == 1
        assert tmp_memory.changelog[0].action == "add"

    def test_changelog_tracks_updates(self, tmp_memory):
        tmp_memory.add(MemoryUpdate(
            key="p.x", category=MemoryCategory.PROJECT,
            content="v1", source="user",
        ))
        tmp_memory.add(
            MemoryUpdate(
                key="p.x", category=MemoryCategory.PROJECT,
                content="v2", source="user",
            ),
            force=True,
        )
        assert len(tmp_memory.changelog) == 2
        assert tmp_memory.changelog[1].action == "update"
        assert tmp_memory.changelog[1].old_version == 1
        assert tmp_memory.changelog[1].new_version == 2

    def test_changelog_saved_to_disk(self, tmp_path):
        mem_dir = tmp_path / "memory"
        store = MemoryStore(mem_dir)
        store.load()
        store.add(MemoryUpdate(
            key="p.x", category=MemoryCategory.PROJECT,
            content="test", source="user",
        ))
        store.save()

        changelog = json.loads((mem_dir / "changelog.json").read_text())
        assert len(changelog) == 1
        assert changelog[0]["action"] == "add"
