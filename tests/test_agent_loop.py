"""
Tests for Commune AI Autonomous Agent Loop and tool calling.
"""

from pathlib import Path
from typing import Any

import pytest

from engine.agent.loop import AgentLoop
from engine.agent.tool_schema import component_to_tool_spec
from engine.communication.envelope import ContentType, Envelope, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest
from engine.memory.store import MemoryStore
from engine.tools.filesystem import ListDirComponent, ReadFileComponent, WriteFileComponent
from engine.tools.terminal import RunCommandComponent


class MockAgentLLM(Component):
    """Simulates an LLM that calls write_file and then returns a final response."""

    def __init__(self, target_path: str):
        super().__init__()
        self.target_path = target_path
        self.call_count = 0

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name="mock_agent_llm",
            kind=ComponentKind.LLM,
            description="Mock LLM for agent tool calling tests",
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        self.call_count += 1
        if self.call_count == 1:
            # First turn: call write_file tool
            tool_call_json = (
                '```tool\n'
                f'{{"tool": "write_file", "params": {{"path": "{self.target_path}", "content": "written by agent"}}}}\n'
                '```'
            )
            return Envelope(task_id=envelope.task_id, sender="mock_llm", content=tool_call_json)
        else:
            # Second turn: final response
            return Envelope(task_id=envelope.task_id, sender="mock_llm", content="Task completed successfully!")


def test_tool_schema_generation():
    rf = ReadFileComponent()
    spec = component_to_tool_spec(rf)

    assert spec["type"] == "function"
    assert spec["function"]["name"] == "read_file"
    assert "path" in spec["function"]["parameters"]["properties"]
    assert "path" in spec["function"]["parameters"]["required"]


@pytest.mark.asyncio
async def test_agent_loop_with_tool_calling(tmp_path: Path):
    target_file = tmp_path / "agent_created.txt"
    mock_llm = MockAgentLLM(str(target_file))

    tools = [ReadFileComponent(), WriteFileComponent(), RunCommandComponent()]
    store = MemoryStore(tmp_path / "memory")

    agent = AgentLoop(
        tools=tools,
        llm=mock_llm,
        memory_store=store,
        max_iterations=3,
    )

    result = await agent.run_task("Create an agent test file")

    assert result["status"] == "success"
    assert result["iterations"] == 1
    assert "Task completed successfully!" in result["final_answer"]
    assert target_file.exists()
    assert target_file.read_text() == "written by agent"

    # Verify memory recorded the action
    session_records = store.all_active()
    assert any("agent_task_" in r.key for r in session_records)


@pytest.mark.asyncio
async def test_agent_loop_rule_fallback(tmp_path: Path):
    test_file = tmp_path / "rule_out.txt"
    tools = [ReadFileComponent(), WriteFileComponent(), RunCommandComponent(), ListDirComponent()]

    agent = AgentLoop(tools=tools, llm=None)

    # 1. Test write
    res_write = await agent.run_task(f"создай файл {test_file} с текстом Привет Агент")
    assert res_write["status"] == "success"
    assert test_file.exists()
    assert test_file.read_text() == "Привет Агент"

    # 2. Test read
    res_read = await agent.run_task(f"прочитай файл {test_file}")
    assert res_read["status"] == "success"
    assert "Привет Агент" in res_read["final_answer"]

    # 3. Test run command
    res_cmd = await agent.run_task("запусти echo 'agent execution ok'")
    assert res_cmd["status"] == "success"
    assert "agent execution ok" in res_cmd["final_answer"]
