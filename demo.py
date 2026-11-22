#!/usr/bin/env python3
"""
Pyramid Demo — runs a 3-stage pipeline with mock components.

This demonstrates the Engine working end-to-end without any LLM.
To run: python demo.py
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.core.engine import Engine
from engine.router.router import PlanStep
from tests.mock_components import EchoComponent, UpperComponent, WordCountComponent


async def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  PYRAMID ENGINE — Demo")
    print("=" * 60)
    print()

    # Create engine
    engine = Engine()

    # Register mock components
    engine.register(EchoComponent("parser"))
    engine.register(WordCountComponent("analyzer"))
    engine.register(UpperComponent("formatter"))

    # Route: "analyze" tasks go through 3 stages
    engine.add_route(
        keywords=["analyze", "review", "check"],
        steps=[
            PlanStep(component="parser"),
            PlanStep(component="analyzer", depends_on=["parser"]),
            PlanStep(component="formatter", depends_on=["analyzer"]),
        ],
    )

    # Route: "count" tasks go to analyzer only
    engine.add_route(
        keywords=["count", "words"],
        steps=[PlanStep(component="analyzer")],
    )

    # Default: echo
    engine.set_default_route([PlanStep(component="parser")])

    # --- Run tasks ---

    print("─" * 60)
    print("Task 1: Analyze code (3-stage pipeline)")
    print("─" * 60)
    result1 = await engine.run("analyze this Python code:\ndef hello(): return 'world'")
    print(f"Output: {result1.output}")
    print(f"Traces:\n{result1.traces_summary()}")
    print()

    print("─" * 60)
    print("Task 2: Count words (single component)")
    print("─" * 60)
    result2 = await engine.run("count the words: The quick brown fox jumps over the lazy dog")
    print(f"Output: {result2.output}")
    print()

    print("─" * 60)
    print("Task 3: Generic request (default route)")
    print("─" * 60)
    result3 = await engine.run("hello pyramid!")
    print(f"Output: {result3.output}")
    print()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for i, r in enumerate([result1, result2, result3], 1):
        print(
            f"  Task {i}: {'✓' if r.success else '✗'} "
            f"({r.total_latency_ms:.1f}ms, "
            f"{len(r.execution.traces)} steps)"
        )


if __name__ == "__main__":
    asyncio.run(main())
