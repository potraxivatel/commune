"""
SwarmTeam coordinator for Commune AI multi-agent collaborative workflows.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageMetadata, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest
from engine.memory.records import MemoryCategory, MemoryUpdate
from engine.memory.store import MemoryStore
from engine.swarm.message import SwarmMessage
from engine.swarm.roles import CoderMember, PlannerMember, ReviewerMember


class SwarmSessionTurn:
    """Holds the results of one round of swarm collaboration."""

    def __init__(
        self,
        user_query: str,
        internal_discussion: list[SwarmMessage],
        final_answer: SwarmMessage,
        duration_ms: float,
    ):
        self.user_query = user_query
        self.internal_discussion = internal_discussion
        self.final_answer = final_answer
        self.duration_ms = duration_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "internal_discussion": [m.model_dump() for m in self.internal_discussion],
            "final_answer": self.final_answer.model_dump(),
            "duration_ms": self.duration_ms,
        }


class SwarmTeam(Component):
    """
    Coordinates the collaborative council of agents:
    Planner ➔ Coder ➔ Reviewer ➔ Planner (Synthesizer).
    """

    def __init__(
        self,
        name: str = "swarm_team",
        memory_store: MemoryStore | None = None,
        llm: Component | None = None,
        planner: PlannerMember | None = None,
        coder: CoderMember | None = None,
        reviewer: ReviewerMember | None = None,
    ):
        super().__init__()
        self._name = name
        self._store = memory_store
        self._llm = llm
        self._planner = planner or PlannerMember(llm=llm)
        self._coder = coder or CoderMember(llm=llm)
        self._reviewer = reviewer or ReviewerMember()

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.HYBRID,
            description="Collaborative swarm team (Planner, Coder, Reviewer)",
            version="0.2.0",
            accepts=["text", "json"],
            produces=["json", "text"],
            tags=["swarm", "multi-agent", "collaboration"],
        )

    async def chat(self, user_message: str, session_id: str | None = None) -> SwarmSessionTurn:
        """
        Executes one full round of multi-agent collaboration:
        1. Internal deliberation & planning.
        2. Execution by Coder.
        3. Quality & syntax review by Reviewer.
        4. Polished user-facing answer from Planner.
        """
        start_t = time.time()
        sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
        internal_discussion: list[SwarmMessage] = []

        q_lower = user_message.lower().strip()

        # 1. Check if user wants only review/testing (Coder not needed)
        is_review_task = any(
            w in q_lower
            for w in ("проверь", "протестируй", "запусти тест", "ревью", "check", "test", "verify")
        ) and not any(
            w in q_lower
            for w in ("напиши", "создай", "сделай", "write", "create")
        )

        # 2. Check if user wants code creation/fixing
        is_coding_task = any(
            w in q_lower
            for w in (
                "напиши", "создай", "сделай", "скрипт", "файл", "функци",
                "калькулятор", "змейк", "write", "create", "fix", "исправь"
            )
        ) and not any(
            q_lower.startswith(w) for w in ("зачем", "почему", "что такое", "кто", "как работает", "why", "what is")
        )

        if is_review_task:
            # Selective: Planner -> Reviewer (Coder sleeps)
            plan_msg = await self._planner.plan_task(user_message)
            internal_discussion.append(plan_msg)

            # Try to identify file in query or fallback
            target_f = "solution.py"
            for word in user_message.split():
                if word.endswith(".py") or word.endswith(".json") or word.endswith(".sh"):
                    target_f = word.strip("`'\"")
                    break

            review_msg = await self._reviewer.verify(target_f)
            internal_discussion.append(review_msg)
            final_answer = await self._planner.summarize_to_user(user_message, internal_discussion)

        elif is_coding_task:
            # Selective: Planner -> Coder
            plan_msg = await self._planner.plan_task(user_message)
            internal_discussion.append(plan_msg)

            code_msg, created_file = await self._coder.implement(user_message, plan_msg)
            internal_discussion.append(code_msg)

            # Selective: Reviewer is called ONLY if a file was created or modified
            if created_file:
                review_msg = await self._reviewer.verify(created_file)
                internal_discussion.append(review_msg)

            final_answer = await self._planner.summarize_to_user(user_message, internal_discussion)

        else:
            # Conversational / Banter / Clarification (Coder & Reviewer sleep!)
            internal_discussion.append(
                SwarmMessage(
                    sender="planner",
                    role_title="Архитектор",
                    avatar="🧠",
                    content="Запрос не требует кодинга или тестов — Инженер и Тестировщик не задействованы.",
                    is_internal=True,
                )
            )
            final_answer = await self._planner.answer_question(user_message)

        total_ms = (time.time() - start_t) * 1000.0

        # Persist conversation in session memory store
        if self._store is not None:
            try:
                self._store.add(
                    MemoryUpdate(
                        key=f"swarm_chat_{int(time.time())}",
                        category=MemoryCategory.SESSION,
                        content={
                            "user": user_message,
                            "internal_steps": len(internal_discussion),
                            "final_answer": final_answer.content,
                        },
                        source="swarm_team",
                        component=self._name,
                        tags=["swarm_chat", "multi_agent"],
                    )
                )
                self._store.save()
            except Exception:
                pass

        return SwarmSessionTurn(
            user_query=user_message,
            internal_discussion=internal_discussion,
            final_answer=final_answer,
            duration_ms=total_ms,
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        user_text = str(envelope.content).strip()
        turn = await self.chat(user_text, session_id=envelope.task_id)

        return Envelope(
            task_id=envelope.task_id,
            sender=self._name,
            status=MessageStatus.OK,
            content_type=ContentType.JSON,
            content=turn.to_dict(),
            metadata=MessageMetadata(latency_ms=turn.duration_ms),
        )
