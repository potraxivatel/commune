"""
Pyramid Engine — the central orchestrator.

Accepts a task → routes → schedules → executes → assembles result.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageStatus
from engine.core.component import Component
from engine.router.router import Plan, PlanStep, Router
from engine.scheduler.scheduler import ExecutionResult, Scheduler

logger = logging.getLogger("pyramid")


class Engine:
    """
    Main entry point for Pyramid.

    Usage:
        engine = Engine()
        engine.register(my_component)
        engine.add_route(["code", "python"], [PlanStep("parser"), PlanStep("analyst", ["parser"])])
        result = await engine.run("Explain this Python code: ...")
    """

    def __init__(
        self,
        timeout_seconds: float = 120.0,
        max_parallel: int = 2,
    ) -> None:
        self._router = Router()
        self._timeout = timeout_seconds
        self._max_parallel = max_parallel

    def register(self, component: Component) -> None:
        """Register a component with the engine."""
        self._router.register_component(component)
        logger.info("Registered component: %s (%s)", component.name, component.manifest().kind.value)

    def add_route(self, keywords: list[str], steps: list[PlanStep]) -> None:
        """Add a routing rule."""
        self._router.add_route(keywords, steps)

    def set_default_route(self, steps: list[PlanStep]) -> None:
        """Set the default routing plan."""
        self._router.set_default(steps)

    async def run(self, task_text: str, context: dict[str, Any] | None = None) -> EngineResult:
        """
        Run a task through the full pipeline:
        route → schedule → execute → assemble.
        """
        task_id = uuid.uuid4().hex[:10]
        logger.info("=" * 60)
        logger.info("TASK %s: %s", task_id, task_text[:100])
        logger.info("=" * 60)

        # 1. Route
        plan = self._router.route(task_id, task_text)
        logger.info("Plan: %s", [s.component for s in plan.steps])

        # 2. Build initial envelope
        initial = Envelope(
            task_id=task_id,
            sender="user",
            content_type=ContentType.TEXT,
            content=task_text,
        )

        # 3. Schedule and execute
        scheduler = Scheduler(
            components=self._router.components,
            timeout_seconds=self._timeout,
            max_parallel=self._max_parallel,
        )
        exec_result = await scheduler.execute(plan, initial, context)

        # 4. Assemble
        final_output = self._assemble(exec_result)

        # 5. Log traces
        for trace in exec_result.traces:
            status_icon = "✓" if trace.status == "ok" else "✗"
            logger.info(
                "  %s %s — %s (%.0fms, %d chars)",
                status_icon,
                trace.component,
                trace.status,
                trace.latency_ms,
                trace.output_size,
            )
        logger.info(
            "Total: %.0fms, success=%s",
            exec_result.total_latency_ms,
            exec_result.success,
        )

        return EngineResult(
            task_id=task_id,
            output=final_output,
            execution=exec_result,
            plan=plan,
        )

    def _assemble(self, result: ExecutionResult) -> str:
        """
        Assemble final output from component results.
        v0.1: last-writer-wins — take the last component's output.
        """
        if not result.outputs:
            return "[No output produced]"

        # Get the last successful output
        last_output: Envelope | None = None
        for trace in result.traces:
            if trace.status == "ok" and trace.component in result.outputs:
                last_output = result.outputs[trace.component]

        if last_output is None:
            errors = [
                t.error for t in result.traces if t.error
            ]
            return f"[All components failed]\n" + "\n".join(errors)

        return str(last_output.content)


class EngineResult:
    """Wrapper around the engine's output for easy access."""

    def __init__(
        self,
        task_id: str,
        output: str,
        execution: ExecutionResult,
        plan: Plan,
    ) -> None:
        self.task_id = task_id
        self.output = output
        self.execution = execution
        self.plan = plan

    @property
    def success(self) -> bool:
        return self.execution.success

    @property
    def total_latency_ms(self) -> float:
        return self.execution.total_latency_ms

    def traces_summary(self) -> str:
        lines = []
        for t in self.execution.traces:
            lines.append(
                f"  {'✓' if t.status == 'ok' else '✗'} {t.component}: "
                f"{t.status} ({t.latency_ms:.0f}ms)"
            )
        return "\n".join(lines)
