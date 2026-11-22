"""
Tests for Commune AI deterministic tools and staged compression pipeline.
"""

import asyncio
from pathlib import Path

import pytest

from engine.communication.envelope import ContentType, Envelope, MessageStatus
from engine.memory.store import MemoryStore
from engine.tools.compressor import CompressorComponent
from engine.tools.filesystem import ListDirComponent, ReadFileComponent, WriteFileComponent
from engine.tools.terminal import RunCommandComponent
from engine.workflows.codebase_pipeline import CodebasePipeline


@pytest.mark.asyncio
async def test_read_file_success(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")

    tool = ReadFileComponent()
    env = Envelope(
        task_id="t1",
        sender="test",
        content={"path": str(test_file), "offset": 1, "limit": 2},
    )
    result = await tool.process(env)

    assert result.status == MessageStatus.OK
    assert result.content == "line 2\nline 3\n"
    assert result.metadata.token_count > 0


@pytest.mark.asyncio
async def test_read_file_not_found():
    tool = ReadFileComponent()
    env = Envelope(task_id="t2", sender="test", content="/non/existent/file.xyz")
    result = await tool.process(env)

    assert result.status == MessageStatus.ERROR
    assert "File does not exist" in result.content


@pytest.mark.asyncio
async def test_write_file_nested_and_append(tmp_path: Path):
    target_file = tmp_path / "nested" / "sub" / "output.txt"
    tool = WriteFileComponent()

    # Step 1: write initial
    env1 = Envelope(
        task_id="t3",
        sender="test",
        content={"path": str(target_file), "content": "Commune AI\n"},
    )
    res1 = await tool.process(env1)
    assert res1.status == MessageStatus.OK
    assert target_file.read_text() == "Commune AI\n"

    # Step 2: append
    env2 = Envelope(
        task_id="t4",
        sender="test",
        content={"path": str(target_file), "content": "Second Line\n", "append": True},
    )
    res2 = await tool.process(env2)
    assert res2.status == MessageStatus.OK
    assert target_file.read_text() == "Commune AI\nSecond Line\n"


@pytest.mark.asyncio
async def test_list_dir(tmp_path: Path):
    (tmp_path / "file1.py").write_text("print(1)")
    (tmp_path / "file2.txt").write_text("text")
    (tmp_path / "subdir").mkdir()

    tool = ListDirComponent()
    env = Envelope(
        task_id="t5",
        sender="test",
        content={"path": str(tmp_path), "pattern": "*.py"},
    )
    res = await tool.process(env)

    assert res.status == MessageStatus.OK
    assert isinstance(res.content, dict)
    entries = res.content["entries"]
    assert len(entries) == 1
    assert entries[0]["name"] == "file1.py"


@pytest.mark.asyncio
async def test_run_command_success():
    tool = RunCommandComponent()
    env = Envelope(task_id="t6", sender="test", content="echo 'commune shell tool'")
    res = await tool.process(env)

    assert res.status == MessageStatus.OK
    assert isinstance(res.content, dict)
    assert res.content["exit_code"] == 0
    assert "commune shell tool" in res.content["stdout"]
    assert res.content["timed_out"] is False


@pytest.mark.asyncio
async def test_run_command_error_code():
    tool = RunCommandComponent()
    env = Envelope(
        task_id="t7",
        sender="test",
        content={"command": "python3 -c 'import sys; sys.stderr.write(\"custom err\"); sys.exit(7)'"},
    )
    res = await tool.process(env)

    assert res.status == MessageStatus.OK
    assert res.content["exit_code"] == 7
    assert "custom err" in res.content["stderr"]


@pytest.mark.asyncio
async def test_run_command_timeout():
    tool = RunCommandComponent()
    env = Envelope(
        task_id="t8",
        sender="test",
        content={"command": "sleep 2", "timeout": 0.2},
    )
    res = await tool.process(env)

    assert res.status == MessageStatus.ERROR
    assert res.content["timed_out"] is True
    assert "timed out" in res.content["stderr"]


@pytest.mark.asyncio
async def test_compressor_heuristic():
    comp = CompressorComponent(model_path=None, min_chars_to_compress=50)

    traceback_payload = (
        "Traceback (most recent call last):\n"
        "  File '/home/void/project/main.py', line 142, in run\n"
        "    result = execute(query)\n"
        "  File '/home/void/project/exec.py', line 55, in execute\n"
        "    raise ValueError('Invalid token count')\n"
        "ValueError: Invalid token count\n"
        "Some noisy debug log 1\n"
        "Some noisy debug log 2\n"
        "Some noisy debug log 3\n"
        "46 passed in 0.42s\n"
    )

    env = Envelope(task_id="t9", sender="test", content=traceback_payload)
    res = await comp.process(env)

    assert res.status == MessageStatus.OK
    assert isinstance(res.content, str)
    assert "TRACEBACK:" in res.content
    assert "ValueError: Invalid token count" in res.content
    assert "TEST_SUMMARY: 46 passed in 0.42s" in res.content
    # Verified compression happened
    assert len(res.content) < len(traceback_payload)


@pytest.mark.asyncio
async def test_codebase_pipeline_end_to_end(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory")
    pipeline = CodebasePipeline(memory_store=store)

    # Run command that produces output
    result = await pipeline.execute_tool_and_compress(
        tool_name="run_command",
        params="python3 -c 'print(\"Executing unit check\"); print(\"46 passed in 0.15s\")'",
        task_id="pipeline_t1",
    )

    assert "stage1_raw" in result
    assert "stage2_compressed" in result
    assert result["stage1_raw"].status == MessageStatus.OK
    assert result["stage2_compressed"].status == MessageStatus.OK

    # Check that knowledge store received memory update
    records = store.all_active()
    assert len(records) >= 1
    assert any("tool" in str(r.content) for r in records)
