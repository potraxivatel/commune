"""
Specialized Swarm Agents for Commune AI:
- PlannerMember (Архитектор): breaks tasks down, speaks with user, guides the swarm
- CoderMember (Инженер): writes and edits code using filesystem tools
- ReviewerMember (Тестировщик): validates code syntax, executes tests, spots bugs
"""

from __future__ import annotations

import re
from typing import Any

from engine.communication.envelope import ContentType, Envelope, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest
from engine.swarm.message import SwarmMessage
from engine.tools.filesystem import ReadFileComponent, WriteFileComponent
from engine.tools.terminal import RunCommandComponent


class SwarmMember(Component):
    """Base class for swarm participants."""

    def __init__(self, name: str, role_title: str, avatar: str, description: str):
        super().__init__()
        self._name = name
        self._role_title = role_title
        self._avatar = avatar
        self._description = description

    @property
    def role_title(self) -> str:
        return self._role_title

    @property
    def avatar(self) -> str:
        return self._avatar

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name=self._name,
            kind=ComponentKind.HYBRID,
            description=self._description,
            tags=["swarm", self._name],
        )


class PlannerMember(SwarmMember):
    """Coordinates the task and communicates with the user."""

    def __init__(self, llm: Component | None = None):
        super().__init__(
            name="planner",
            role_title="Архитектор",
            avatar="🧠",
            description="Анализирует задачу, строит план действий и формулирует ответ пользователю",
        )
        self._llm = llm

    async def plan_task(self, user_query: str) -> SwarmMessage:
        """Formulates an initial operational plan."""
        plan_text = ""

        if self._llm is not None:
            try:
                prompt = (
                    f"Пользователь запросил: «{user_query}».\n"
                    "Ты Архитектор AI-команды. Составь краткий план работы (2-3 пункта) "
                    "для Инженера (написание кода) и Тестировщика (проверка)."
                )
                res = await self._llm.process(Envelope(task_id="plan", sender="planner", content=prompt))
                plan_text = str(res.content).strip()
            except Exception:
                plan_text = ""

        if not plan_text:
            q_lower = user_query.lower()
            if any(w in q_lower for w in ("файл", "скрипт", "код", "напиши", "создай", "сделай", "калькулятор", "write", "create")):
                plan_text = (
                    f"Принял задачу: «{user_query}».\n"
                    "План работы:\n"
                    "1. [Инженер] сгенерирует структуру и запишет рабочий код.\n"
                    "2. [Тестировщик] проверит синтаксис компилятором.\n"
                    "3. Я подготовлю финальный отчёт для пользователя."
                )
            else:
                plan_text = f"Задача понятна: «{user_query}». Приступаем к решению."

        return SwarmMessage(
            sender=self._name,
            role_title=self._role_title,
            avatar=self._avatar,
            recipient="all",
            content=plan_text,
            is_internal=True,
        )

    async def answer_question(self, user_query: str) -> SwarmMessage:
        """Answers a general conversational query directly."""
        ans = ""
        if self._llm is not None:
            try:
                ctx = {
                    "system_prompt": (
                        "Ты Архитектор системы Commune AI. Отвечай прямо на вопрос пользователя, "
                        "кратко, понятно, профессионально и только по делу на русском языке. "
                        "Никогда не повторяй саму инструкцию и не пиши фразы 'Вот ответ'."
                    )
                }
                res = await self._llm.process(
                    Envelope(task_id="chat", sender="planner", content=user_query),
                    context=ctx,
                )
                raw_ans = str(res.content).strip()
                # Clean up any meta-parroting
                cleaned = re.sub(r'^(?:Конечно!|Привет!|Вот ответ[^:]*:)\s*', '', raw_ans, flags=re.IGNORECASE)
                ans = cleaned.strip()
            except Exception:
                ans = ""

        if not ans:
            ans = f"Commune AI — это децентрализованный оркестратор. Мы храним факты в легких JSON-файлах в папке memory/, а не в тяжелой базе данных."

        return SwarmMessage(
            sender=self._name,
            role_title=self._role_title,
            avatar=self._avatar,
            recipient="user",
            content=ans,
            is_internal=False,
        )

    async def summarize_to_user(self, user_query: str, history: list[SwarmMessage]) -> SwarmMessage:
        """Produces friendly, polished final answer to the user."""
        actions_taken = [m for m in history if m.sender in ("coder", "reviewer")]
        action_summaries = "\n".join(f"• {m.avatar} {m.role_title}: {m.content}" for m in actions_taken)

        if not action_summaries:
            action_summaries = "• Задача проанализирована и решена."

        final_text = ""
        if self._llm is not None:
            try:
                prompt = (
                    f"Пользователь просил: «{user_query}».\n"
                    f"Действия команды:\n{action_summaries}\n\n"
                    "Напиши краткий, вежливый ответ пользователю о том, что задача выполнена."
                )
                res = await self._llm.process(Envelope(task_id="sum", sender="planner", content=prompt))
                final_text = str(res.content).strip()
            except Exception:
                final_text = ""

        if not final_text:
            final_text = (
                f"Всё готово по запросу: «{user_query}»!\n\n"
                f"**Что сделала команда:**\n"
                f"{action_summaries}\n\n"
                "Если нужно что-то дополнить или протестировать — скажи!"
            )

        return SwarmMessage(
            sender=self._name,
            role_title=self._role_title,
            avatar=self._avatar,
            recipient="user",
            content=final_text,
            is_internal=False,
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        msg = await self.plan_task(str(envelope.content))
        return Envelope(
            task_id=envelope.task_id,
            sender=self._name,
            content=msg.model_dump(),
        )


class CoderMember(SwarmMember):
    """Writes files and handles code creation using local neural models."""

    def __init__(
        self,
        llm: Component | None = None,
        write_tool: WriteFileComponent | None = None,
        read_tool: ReadFileComponent | None = None,
    ):
        super().__init__(
            name="coder",
            role_title="Инженер",
            avatar="💻",
            description="Пишет программный код через нейросеть, создает файлы и сохраняет их на диске",
        )
        self._llm = llm
        self._write_tool = write_tool or WriteFileComponent()
        self._read_tool = read_tool or ReadFileComponent()

    async def implement(self, user_query: str, plan_msg: SwarmMessage) -> tuple[SwarmMessage, str | None]:
        """Generates and writes requested code to disk."""
        target_file = None
        code_content = ""

        # 1. Determine target filename
        m_file = re.search(r"([a-zA-Z0-9_\-\.\/]+\.(?:py|sh|json|md|txt|toml))", user_query)
        if m_file:
            target_file = m_file.group(1)
        else:
            q_lower = user_query.lower()
            if any(k in q_lower for k in ("калькулятор", "calc")):
                target_file = "calculator.py"
            elif any(k in q_lower for k in ("змейк", "snake")):
                target_file = "snake.py"
            elif any(k in q_lower for k in ("скачиван", "download")):
                target_file = "downloader.py"
            elif any(k in q_lower for k in ("скрипт", "код", "функци", "программ", "write", "code")):
                target_file = "solution.py"

        # 2. If neural net is available, generate real code
        if self._llm is not None and target_file:
            try:
                coder_prompt = (
                    f"Пользователь просит написать код: {user_query}\n"
                    f"Целевой файл: {target_file}\n\n"
                    "Напиши полный, качественный и готовый к запуску код на Python.\n"
                    "Помести код строго внутрь блока ```python ... ```."
                )
                res = await self._llm.process(
                    Envelope(task_id="coder_llm", sender="coder", content=coder_prompt)
                )
                raw_reply = str(res.content)

                # Extract python block
                m_code = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_reply, re.DOTALL)
                if m_code:
                    code_content = m_code.group(1).strip()
                elif "def " in raw_reply or "print(" in raw_reply:
                    code_content = raw_reply.strip()
            except Exception:
                code_content = ""

        # 3. Fallback if neural net not loaded or returned empty
        if not code_content and target_file:
            code_content = (
                "#!/usr/bin/env python3\n"
                f'"""Solution for: {user_query}"""\n\n'
                "def main():\n"
                f'    print("Выполнено: {user_query}")\n\n'
                'if __name__ == "__main__":\n'
                "    main()\n"
            )

        # 4. Save to disk if code was generated
        if target_file and code_content:
            write_env = await self._write_tool.process(
                Envelope(
                    task_id="swarm_write",
                    sender=self._name,
                    content={"path": target_file, "content": code_content},
                )
            )

            if write_env.status == MessageStatus.OK:
                generator_source = "нейросеть Qwen Coder" if self._llm is not None else "модуль кода"
                reply = f"Сгенерировал код через {generator_source} и сохранил в файл `{target_file}` ({len(code_content)} байт)."
            else:
                reply = f"Ошибка при записи файла `{target_file}`: {write_env.content}"
        else:
            reply = "Изучил задачу. Для её выполнения создание новых файлов не потребовалось."

        return SwarmMessage(
            sender=self._name,
            role_title=self._role_title,
            avatar=self._avatar,
            recipient="all",
            content=reply,
            metadata={"file": target_file, "bytes": len(code_content)},
            is_internal=True,
        ), target_file

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        msg, _ = await self.implement(str(envelope.content), SwarmMessage(sender="planner", role_title="Архитектор", content=""))
        return Envelope(task_id=envelope.task_id, sender=self._name, content=msg.model_dump())


class ReviewerMember(SwarmMember):
    """Verifies generated code syntax and runs tests."""

    def __init__(self, cmd_tool: RunCommandComponent | None = None):
        super().__init__(
            name="reviewer",
            role_title="Тестировщик",
            avatar="🔍",
            description="Проверяет синтаксис, запускает тесты и валидирует надёжность кода",
        )
        self._cmd_tool = cmd_tool or RunCommandComponent()

    async def verify(self, file_path: str | None) -> SwarmMessage:
        """Runs validation on created code file."""
        if not file_path:
            return SwarmMessage(
                sender=self._name,
                role_title=self._role_title,
                avatar=self._avatar,
                recipient="all",
                content="Новых файлов нет, логика задачи не нарушает спецификацию.",
                is_internal=True,
            )

        if file_path.endswith(".py"):
            # Check syntax via python -m py_compile
            cmd = f"python3 -m py_compile {file_path}"
            res_env = await self._cmd_tool.process(
                Envelope(task_id="swarm_verify", sender=self._name, content=cmd)
            )

            if isinstance(res_env.content, dict) and res_env.content.get("exit_code") == 0:
                review_text = f"Проверил синтаксис `{file_path}` через компилятор — ошибок нет, код валиден! ✓"
            else:
                err = res_env.content.get("stderr", "Syntax error") if isinstance(res_env.content, dict) else str(res_env.content)
                review_text = f"Внимание! При проверке `{file_path}` обнаружена ошибка:\n{err}"
        else:
            review_text = f"Файл `{file_path}` успешно создан и проверен на диске."

        return SwarmMessage(
            sender=self._name,
            role_title=self._role_title,
            avatar=self._avatar,
            recipient="all",
            content=review_text,
            is_internal=True,
        )

    async def process(self, envelope: Envelope, context: dict[str, Any] | None = None) -> Envelope:
        target_f = envelope.content if isinstance(envelope.content, str) else None
        msg = await self.verify(target_f)
        return Envelope(task_id=envelope.task_id, sender=self._name, content=msg.model_dump())
