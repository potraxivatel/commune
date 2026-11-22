"""
Autonomous Agent Loop for Commune AI.
Coordinates LLM reasoning, structured tool invocation, staged compression, and memory.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageMetadata, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest
from engine.memory.records import MemoryCategory, MemoryUpdate
from engine.memory.store import MemoryStore
from engine.tools.compressor import CompressorComponent


class AgentTraceStep:
    def __init__(self, step_num: int, tool: str, params: Any, result_summary: str, duration_ms: float):
        self.step_num = step_num
        self.tool = tool
        self.params = params
        self.result_summary = result_summary
        self.duration_ms = duration_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_num,
            "tool": self.tool,
            "params": self.params,
            "result": self.result_summary,
            "duration_ms": self.duration_ms,
        }


SYSTEM_AGENT_PROMPT = """You are Commune AI, an autonomous local software engineering agent.
You have access to the following deterministic system tools:
- read_file(path, offset, limit)
- write_file(path, content, append)
- run_command(command, timeout)
- list_dir(path, pattern, recursive)

When you need to perform an action, output a JSON tool call block exactly like:
```tool
{"tool": "tool_name", "params": {"arg1": "value1"}}
```
When you have completed the task or have the final answer, reply directly to the user without tool blocks.
"""


class AgentLoop(Component):
    """
    Autonomous execution loop:
    1. Interprets user intent.
    2. Invokes appropriate system tools.
    3. Compresses verbose tool responses.
    4. Iterates until goal is achieved or max iterations reached.
    """

    def __init__(
        self,
        name: str = "agent_loop",
        tools: list[Component] | None = None,
        llm: Component | None = None,
        compressor: CompressorComponent | None = None,
        memory_store: MemoryStore | None = None,
        max_iterations: int = 5,
    ) -> None:
        super().__init__()
        self._name = name
        self._tools: dict[str, Component] = {t.manifest().name: t for t in (tools or [])}
        self._llm = llm
        self._compressor = compressor or CompressorComponent()
        self._memory_store = memory_store
        self._max_iterations = max_iterations

    def register_tool(self, tool: Component) -> None:
        self._tools[tool.manifest().name] = tool

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.HYBRID,
            description="Autonomous multi-step tool-calling agent loop",
            version="0.2.0",
            accepts=["text", "json"],
            produces=["text", "json"],
            tags=["agent", "autonomous", "loop"],
        )

    def _extract_tool_call(self, text: str) -> tuple[str, dict[str, Any]] | None:
        """Parses tool calls from LLM text output."""
        # 1. Check for ```tool ... ``` pattern
        match = re.search(r"```tool\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return data.get("tool", ""), data.get("params", {})
            except Exception:
                pass

        # 2. Check for raw JSON object with "tool" and "params"
        match_json = re.search(r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"params"\s*:\s*(\{.*?\})\s*\}', text, re.DOTALL)
        if match_json:
            try:
                t_name = match_json.group(1)
                t_params = json.loads(match_json.group(2))
                return t_name, t_params
            except Exception:
                pass

        return None

    def _rule_based_intent_extractor(self, user_query: str) -> tuple[str, dict[str, Any]] | None:
        """
        Lightweight fallback pattern matcher for common commands when no LLM is loaded.
        """
        q = user_query.strip()
        lower = q.lower()

        # Write file pattern: e.g. "создай файл hello.py с текстом print('hi')"
        m_write = re.search(r"(?:создай файл|запиши в файл|create file|write file)\s+([^\s]+)\s+(?:с текстом|with content)\s+(.+)", q, re.IGNORECASE)
        if m_write:
            return "write_file", {"path": m_write.group(1), "content": m_write.group(2).strip("\"'")}

        # Run command pattern: e.g. "запусти python hello.py" or "run pytest"
        m_run = re.search(r"^(?:запусти|выполни|run|exec)\s+(.+)$", q, re.IGNORECASE)
        if m_run and not lower.startswith("запомни"):
            return "run_command", {"command": m_run.group(1)}

        # Read file pattern: e.g. "прочитай файл test.txt"
        m_read = re.search(r"^(?:прочитай(?: файл)?|покажи(?: файл)?|cat|read(?: file)?)\s+([^\s]+)$", q, re.IGNORECASE)
        if m_read:
            return "read_file", {"path": m_read.group(1)}

        # List dir pattern: e.g. "покажи файлы"
        if any(k in lower for k in ("покажи файлы", "список файлов", "list files", "ls", "директори")):
            return "list_dir", {"path": "."}

        return None

    async def run_task(self, task_description: str, task_id: str | None = None) -> dict[str, Any]:
        """
        Executes a multi-step task to completion.
        """
        tid = task_id or f"agent_{uuid.uuid4().hex[:8]}"
        traces: list[AgentTraceStep] = []
        conversation: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_AGENT_PROMPT},
            {"role": "user", "content": task_description},
        ]

        total_start = time.time()
        final_answer = ""
        current_input = task_description

        for step_idx in range(1, self._max_iterations + 1):
            tool_call: tuple[str, dict[str, Any]] | None = None

            # 1. Ask LLM or check rule-based extractor
            if self._llm is not None:
                prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
                llm_res = await self._llm.process(
                    Envelope(task_id=tid, sender="agent", content=prompt_text)
                )
                model_reply = str(llm_res.content).strip()
                conversation.append({"role": "assistant", "content": model_reply})
                tool_call = self._extract_tool_call(model_reply)
                if not tool_call:
                    final_answer = model_reply
                    break
            else:
                # Fallback rule extractor for direct execution without active LLM
                tool_call = self._rule_based_intent_extractor(current_input)
                if not tool_call:
                    final_answer = f"Task completed: {task_description}"
                    break

            # 2. Execute identified Tool Call
            tool_name, tool_params = tool_call
            t_comp = self._tools.get(tool_name)

            t0 = time.time()
            if not t_comp:
                tool_err = f"Error: Tool '{tool_name}' is not registered."
                traces.append(AgentTraceStep(step_idx, tool_name, tool_params, tool_err, (time.time() - t0) * 1000))
                conversation.append({"role": "system", "content": f"[Tool Error]: {tool_err}"})
                if self._llm is None:
                    final_answer = tool_err
                    break
                continue

            tool_env = await t_comp.process(
                Envelope(task_id=tid, sender="agent", content=tool_params)
            )

            # 3. Apply Stage 2 Compressor if output is verbose
            raw_output = str(tool_env.content)
            if len(raw_output) > 250:
                comp_env = await self._compressor.process(tool_env)
                summary_output = str(comp_env.content)
            else:
                summary_output = raw_output

            step_time = (time.time() - t0) * 1000.0
            traces.append(AgentTraceStep(step_idx, tool_name, tool_params, summary_output, step_time))

            # Provide feedback to conversation
            conversation.append({
                "role": "system",
                "content": f"[Tool {tool_name} Output]:\n{summary_output}",
            })

            # In rule-based mode, single tool completes the task
            if self._llm is None:
                final_answer = f"Tool '{tool_name}' executed successfully:\n{summary_output}"
                break

        # Save action to memory store if present
        if self._memory_store is not None:
            try:
                self._memory_store.add(
                    MemoryUpdate(
                        key=f"agent_task_{int(time.time())}",
                        category=MemoryCategory.SESSION,
                        content={
                            "task": task_description,
                            "steps_count": len(traces),
                            "final_answer": final_answer[:500],
                        },
                        source="agent_loop",
                        component="agent_loop",
                        tags=["agent_action"],
                    )
                )
                self._memory_store.save()
            except Exception:
                pass

        total_duration = (time.time() - total_start) * 1000.0
        return {
            "task_id": tid,
            "status": "success" if final_answer else "exhausted",
            "iterations": len(traces),
            "final_answer": final_answer,
            "traces": [t.to_dict() for t in traces],
            "duration_ms": total_duration,
        }

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        task_id = envelope.task_id
        user_query = str(envelope.content).strip()

        result = await self.run_task(user_query, task_id=task_id)

        return Envelope(
            task_id=task_id,
            sender=self._name,
            status=MessageStatus.OK if result["status"] == "success" else MessageStatus.PARTIAL,
            content_type=ContentType.JSON,
            content=result,
            metadata=MessageMetadata(
                latency_ms=result["duration_ms"],
            ),
        )
