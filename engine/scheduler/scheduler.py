"""
Scheduler — executes a Plan respecting dependencies, parallelism,
VRAM budget, and timeouts.

v0.1: simple topological execution. Parallel steps run via asyncio.
VRAM budget is advisory (logged, not enforced) in this version.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageStatus
from engine.core.component import Component
from engine.router.router import Plan

logger = logging.getLogger("pyramid.scheduler")


@dataclass
class StepTrace:
    """Execution trace for one step."""

    component: str
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "pending"
    output_size: int = 0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class ExecutionResult:
    """Result of executing a full plan."""

    task_id: str
    outputs: dict[str, Envelope] = field(default_factory=dict)
    traces: list[StepTrace] = field(default_factory=list)
    success: bool = True

    @property
    def total_latency_ms(self) -> float:
        if not self.traces:
            return 0.0
        start = min(t.start_time for t in self.traces)
        end = max(t.end_time for t in self.traces)
        return (end - start) * 1000


class Scheduler:
    """Execute a plan with dependency resolution and basic parallelism."""

    def __init__(
        self,
        components: dict[str, Component],
        timeout_seconds: float = 60.0,
        max_parallel: int = 2,
    ) -> None:
        self._components = components
        self._timeout = timeout_seconds
        self._max_parallel = max_parallel

    async def execute(
        self,
        plan: Plan,
        initial_envelope: Envelope,
        context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute all steps in the plan respecting dependencies."""
        result = ExecutionResult(task_id=plan.task_id)
        completed: dict[str, Envelope] = {}

        # Group steps by dependency depth (simple topological sort)
        layers = self._topological_layers(plan)

        for layer in layers:
            tasks = []
            for step in layer:
                component = self._components.get(step.component)
                if component is None:
                    trace = StepTrace(
                        component=step.component,
                        start_time=time.time(),
                        end_time=time.time(),
                        status="error",
                        error=f"component '{step.component}' not found",
                    )
                    result.traces.append(trace)
                    result.success = False
                    continue

                # Build input: use dependency outputs if available,
                # otherwise use the initial envelope
                if step.depends_on:
                    dep_outputs = {
                        dep: completed[dep]
                        for dep in step.depends_on
                        if dep in completed
                    }
                    # Use the last dependency's output as input
                    input_envelope = list(dep_outputs.values())[-1] if dep_outputs else initial_envelope
                    step_context = {**(context or {}), "prior_outputs": dep_outputs}
                else:
                    input_envelope = initial_envelope
                    step_context = context

                tasks.append(
                    self._run_step(component, input_envelope, step_context)
                )

            # Run layer in parallel (up to max_parallel)
            semaphore = asyncio.Semaphore(self._max_parallel)

            async def _bounded(coro):
                async with semaphore:
                    return await coro

            step_results = await asyncio.gather(
                *[_bounded(t) for t in tasks], return_exceptions=True
            )

            for i, sr in enumerate(step_results):
                if isinstance(sr, Exception):
                    name = layer[i].component if i < len(layer) else "unknown"
                    trace = StepTrace(
                        component=name,
                        start_time=time.time(),
                        end_time=time.time(),
                        status="error",
                        error=str(sr),
                    )
                    result.traces.append(trace)
                    result.success = False
                elif isinstance(sr, tuple):
                    trace, envelope = sr
                    result.traces.append(trace)
                    result.outputs[trace.component] = envelope
                    completed[trace.component] = envelope
                    if trace.status == "error":
                        result.success = False

        return result

    async def _run_step(
        self,
        component: Component,
        envelope: Envelope,
        context: dict[str, Any] | None,
    ) -> tuple[StepTrace, Envelope]:
        """Run a single component with timeout."""
        trace = StepTrace(component=component.name, start_time=time.time())

        try:
            await asyncio.wait_for(component.load(), timeout=self._timeout)
            output = await asyncio.wait_for(
                component.process(envelope, context), timeout=self._timeout
            )
            trace.end_time = time.time()
            trace.status = output.status.value
            trace.output_size = len(str(output.content))
            return trace, output

        except asyncio.TimeoutError:
            trace.end_time = time.time()
            trace.status = "timeout"
            trace.error = f"exceeded {self._timeout}s timeout"
            logger.error("component=%s timed out", component.name)
            return trace, Envelope.error(
                envelope.task_id, component.name, trace.error
            )

        except Exception as exc:
            trace.end_time = time.time()
            trace.status = "error"
            trace.error = str(exc)
            logger.exception("component=%s failed", component.name)
            return trace, Envelope.error(
                envelope.task_id, component.name, str(exc)
            )

    @staticmethod
    def _topological_layers(plan: Plan) -> list[list]:
        """Group steps into layers by dependency depth."""
        from engine.router.router import PlanStep

        step_map: dict[str, PlanStep] = {s.component: s for s in plan.steps}
        layers: list[list[PlanStep]] = []
        placed: set[str] = set()

        while len(placed) < len(plan.steps):
            layer = []
            for step in plan.steps:
                if step.component in placed:
                    continue
                deps_met = all(d in placed for d in step.depends_on)
                if deps_met:
                    layer.append(step)

            if not layer:
                # Remaining steps have unresolvable deps — push them anyway
                remaining = [
                    s for s in plan.steps if s.component not in placed
                ]
                layers.append(remaining)
                break

            layers.append(layer)
            for step in layer:
                placed.add(step.component)

        return layers
