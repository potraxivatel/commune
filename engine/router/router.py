"""
Router — decides which components handle a task and in what order.

v0.1: rule-based routing via a decision table.
Maps task categories to component sequences with dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from engine.core.component import Component

logger = logging.getLogger("pyramid.router")


@dataclass
class PlanStep:
    """One step in an execution plan."""

    component: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """Execution plan: an ordered list of steps with dependencies."""

    task_id: str
    steps: list[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [
                {"component": s.component, "depends_on": s.depends_on}
                for s in self.steps
            ],
        }


# --- Routing rules ---

@dataclass
class Route:
    """A routing rule: if any keyword matches, use these steps."""

    keywords: list[str]
    steps: list[PlanStep]


class Router:
    """
    Rule-based router. Matches task text against keyword rules
    and returns an execution plan.
    """

    def __init__(self) -> None:
        self._routes: list[Route] = []
        self._components: dict[str, Component] = {}
        self._default_steps: list[PlanStep] = []

    def register_component(self, component: Component) -> None:
        self._components[component.name] = component

    def add_route(self, keywords: list[str], steps: list[PlanStep]) -> None:
        self._routes.append(Route(keywords=keywords, steps=steps))

    def set_default(self, steps: list[PlanStep]) -> None:
        self._default_steps = steps

    def route(self, task_id: str, task_text: str) -> Plan:
        """Match task text against rules and return a plan."""
        text_lower = task_text.lower()

        for rule in self._routes:
            if any(kw in text_lower for kw in rule.keywords):
                logger.info(
                    "task=%s routed via keywords=%s", task_id, rule.keywords
                )
                return Plan(task_id=task_id, steps=rule.steps)

        if self._default_steps:
            logger.info("task=%s routed to default", task_id)
            return Plan(task_id=task_id, steps=self._default_steps)

        # Fallback: run all registered components sequentially
        logger.info("task=%s fallback: all components sequential", task_id)
        steps = [PlanStep(component=name) for name in self._components]
        return Plan(task_id=task_id, steps=steps)

    @property
    def components(self) -> dict[str, Component]:
        return self._components
