"""
Tests for Pyramid Engine — Phase 1.

Covers:
  - Envelope creation and serialization
  - Component interface compliance
  - Router rule matching
  - Scheduler execution with dependencies
  - Engine end-to-end pipeline
  - Error handling and timeouts
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.communication.envelope import ContentType, Envelope, MessageStatus
from engine.core.component import ComponentKind
from engine.core.engine import Engine
from engine.router.router import Plan, PlanStep, Router
from engine.scheduler.scheduler import Scheduler

from tests.mock_components import (
    EchoComponent,
    FailComponent,
    SlowComponent,
    UpperComponent,
    WordCountComponent,
)


# --- Envelope ---

class TestEnvelope:
    def test_create_basic(self):
        env = Envelope(task_id="t1", sender="test", content="hello")
        assert env.task_id == "t1"
        assert env.sender == "test"
        assert env.content == "hello"
        assert env.status == MessageStatus.OK

    def test_error_factory(self):
        env = Envelope.error("t1", "comp", "something broke")
        assert env.status == MessageStatus.ERROR
        assert env.content == "something broke"

    def test_serialization_roundtrip(self):
        env = Envelope(
            task_id="t1",
            sender="test",
            content_type=ContentType.JSON,
            content={"key": "value"},
        )
        data = env.model_dump()
        restored = Envelope.model_validate(data)
        assert restored.content == {"key": "value"}
        assert restored.content_type == ContentType.JSON


# --- Components ---

class TestComponents:
    @pytest.mark.asyncio
    async def test_echo(self):
        comp = EchoComponent("my-echo")
        env = Envelope(task_id="t1", sender="user", content="test input")
        result = await comp.process(env)
        assert result.status == MessageStatus.OK
        assert "test input" in result.content
        assert result.sender == "my-echo"

    @pytest.mark.asyncio
    async def test_upper(self):
        comp = UpperComponent()
        env = Envelope(task_id="t1", sender="user", content="hello world")
        result = await comp.process(env)
        assert result.content == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_word_counter(self):
        comp = WordCountComponent()
        env = Envelope(task_id="t1", sender="user", content="one two three")
        result = await comp.process(env)
        assert result.content_type == ContentType.JSON
        assert result.content["word_count"] == 3

    def test_manifest(self):
        comp = WordCountComponent()
        m = comp.manifest()
        assert m.name == "word-counter"
        assert m.kind == ComponentKind.CODE
        assert "analysis" in m.tags


# --- Router ---

class TestRouter:
    def test_keyword_routing(self):
        router = Router()
        router.register_component(EchoComponent("echo"))
        router.register_component(WordCountComponent())

        router.add_route(
            ["count", "words"],
            [PlanStep(component="word-counter")],
        )

        plan = router.route("t1", "count the words in this text")
        assert len(plan.steps) == 1
        assert plan.steps[0].component == "word-counter"

    def test_default_route(self):
        router = Router()
        router.register_component(EchoComponent("echo"))
        router.set_default([PlanStep(component="echo")])

        plan = router.route("t1", "some random input")
        assert plan.steps[0].component == "echo"

    def test_fallback_all_components(self):
        router = Router()
        router.register_component(EchoComponent("echo"))
        router.register_component(UpperComponent())

        plan = router.route("t1", "no matching keywords")
        assert len(plan.steps) == 2


# --- Scheduler ---

class TestScheduler:
    @pytest.mark.asyncio
    async def test_single_step(self):
        comps = {"echo": EchoComponent("echo")}
        scheduler = Scheduler(comps)
        plan = Plan(task_id="t1", steps=[PlanStep(component="echo")])
        env = Envelope(task_id="t1", sender="user", content="hello")

        result = await scheduler.execute(plan, env)
        assert result.success
        assert "echo" in result.outputs
        assert len(result.traces) == 1

    @pytest.mark.asyncio
    async def test_chained_steps(self):
        comps = {
            "echo": EchoComponent("echo"),
            "upper": UpperComponent(),
        }
        scheduler = Scheduler(comps)
        plan = Plan(
            task_id="t1",
            steps=[
                PlanStep(component="echo"),
                PlanStep(component="upper", depends_on=["echo"]),
            ],
        )
        env = Envelope(task_id="t1", sender="user", content="hello")

        result = await scheduler.execute(plan, env)
        assert result.success
        # Upper should have uppercased echo's output
        upper_out = result.outputs["upper"].content
        assert upper_out == upper_out.upper()

    @pytest.mark.asyncio
    async def test_missing_component(self):
        scheduler = Scheduler({})
        plan = Plan(task_id="t1", steps=[PlanStep(component="nonexistent")])
        env = Envelope(task_id="t1", sender="user", content="hello")

        result = await scheduler.execute(plan, env)
        assert not result.success

    @pytest.mark.asyncio
    async def test_component_failure(self):
        comps = {"fail": FailComponent()}
        scheduler = Scheduler(comps)
        plan = Plan(task_id="t1", steps=[PlanStep(component="fail")])
        env = Envelope(task_id="t1", sender="user", content="hello")

        result = await scheduler.execute(plan, env)
        assert not result.success
        assert result.traces[0].error is not None

    @pytest.mark.asyncio
    async def test_timeout(self):
        comps = {"slow": SlowComponent("slow", delay=10.0)}
        scheduler = Scheduler(comps, timeout_seconds=0.5)
        plan = Plan(task_id="t1", steps=[PlanStep(component="slow")])
        env = Envelope(task_id="t1", sender="user", content="hello")

        result = await scheduler.execute(plan, env)
        assert result.traces[0].status == "timeout"

    @pytest.mark.asyncio
    async def test_parallel_independent_steps(self):
        comps = {
            "echo-a": EchoComponent("echo-a"),
            "echo-b": EchoComponent("echo-b"),
        }
        scheduler = Scheduler(comps, max_parallel=2)
        plan = Plan(
            task_id="t1",
            steps=[
                PlanStep(component="echo-a"),
                PlanStep(component="echo-b"),
            ],
        )
        env = Envelope(task_id="t1", sender="user", content="hello")

        result = await scheduler.execute(plan, env)
        assert result.success
        assert "echo-a" in result.outputs
        assert "echo-b" in result.outputs


# --- Engine end-to-end ---

class TestEngine:
    @pytest.mark.asyncio
    async def test_simple_pipeline(self):
        engine = Engine()
        engine.register(EchoComponent("echo"))
        engine.register(UpperComponent())

        engine.add_route(
            ["upper", "capitalize"],
            [
                PlanStep(component="echo"),
                PlanStep(component="upper", depends_on=["echo"]),
            ],
        )

        result = await engine.run("please upper this text")
        assert result.success
        assert result.output == result.output.upper()

    @pytest.mark.asyncio
    async def test_default_route(self):
        engine = Engine()
        engine.register(WordCountComponent())
        engine.set_default_route([PlanStep(component="word-counter")])

        result = await engine.run("one two three four five")
        assert result.success
        # word-counter returns JSON, engine assembles it as string
        assert "word_count" in result.output

    @pytest.mark.asyncio
    async def test_three_stage_pipeline(self):
        """Simulate a 3-stage pyramid: parse → analyze → review."""
        engine = Engine()

        # Stage 1: "parse" (echo simulates extracting info)
        engine.register(EchoComponent("parser"))
        # Stage 2: "analyze" (word-counter simulates analysis)
        engine.register(WordCountComponent("analyzer"))
        # Stage 3: "review" (upper simulates final formatting)
        engine.register(UpperComponent("reviewer"))

        engine.add_route(
            ["analyze", "review"],
            [
                PlanStep(component="parser"),
                PlanStep(component="analyzer", depends_on=["parser"]),
                PlanStep(component="reviewer", depends_on=["analyzer"]),
            ],
        )

        result = await engine.run("analyze and review this code")
        assert result.success
        assert len(result.execution.traces) == 3
        assert result.total_latency_ms > 0
