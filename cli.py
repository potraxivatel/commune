#!/usr/bin/env python3
"""
Commune Interactive CLI

Run:
  ./cli.py
"""

import os
import sys
from pathlib import Path

# Auto-reexec inside .venv if launched with system python
base_dir = Path(__file__).resolve().parent
venv_python = base_dir / ".venv" / "bin" / "python3"
if venv_python.exists() and sys.executable != str(venv_python):
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import asyncio
import datetime
import json
import logging
import time

# Add project root to path
sys.path.insert(0, str(base_dir))

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from engine.communication.envelope import ContentType, Envelope, MessageStatus
from engine.core.component import Component, ComponentKind, ComponentManifest
from engine.agent import AgentLoop
from engine.core.engine import Engine
from engine.inference.llama_backend import LlamaComponent
from engine.memory.records import MemoryCategory, MemoryUpdate
from engine.memory.store import MemoryStore
from engine.router.router import PlanStep
from engine.swarm import SwarmTeam
from engine.tools import CompressorComponent, ListDirComponent, ReadFileComponent, RunCommandComponent, WriteFileComponent
from engine.workflows import CodebasePipeline
from tests.mock_components import EchoComponent, UpperComponent, WordCountComponent

# Suppress low-level logging in CLI mode
logging.basicConfig(level=logging.WARNING)

console = Console()


class MemoryWriterComponent(Component):
    """
    Component wrapper that connects our custom GGUF model to the MemoryStore.
    Takes any text input, extracts facts, and writes directly into memory.
    """

    def __init__(self, llama_comp: LlamaComponent, store: MemoryStore) -> None:
        super().__init__()
        self._llama = llama_comp
        self._store = store

    def manifest(self) -> ComponentManifest:
        return ComponentManifest(
            name="memory-writer",
            kind=ComponentKind.HYBRID,
            description="Extracts facts using custom GGUF Qwen and commits to MemoryStore",
            tags=["memory", "extractor", "custom-model"],
            produces=["json"],
        )

    async def load(self) -> None:
        await self._llama.load()

    async def unload(self) -> None:
        await self._llama.unload()

    async def process(self, envelope: Envelope, context: dict = None) -> Envelope:
        # 1. Ask our trained model
        out_env = await self._llama.process(envelope, context)
        raw_text = str(out_env.content).strip()

        # 2. Parse JSON & add to memory
        clean_json = raw_text
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()

        try:
            parsed = json.loads(clean_json)
            cat_str = parsed.get("category", "project")
            try:
                cat = MemoryCategory(cat_str)
            except ValueError:
                cat = MemoryCategory.PROJECT

            update = MemoryUpdate(
                key=parsed.get("key", f"extracted.{int(time.time())}"),
                category=cat,
                content=parsed.get("content", raw_text),
                tags=parsed.get("tags", []),
                source=f"component:{self.name}",
                component=self.name,
            )
            record = self._store.add(update, force=True)
            self._store.save()

            return Envelope(
                task_id=envelope.task_id,
                sender=self.name,
                status=MessageStatus.OK,
                content_type=ContentType.JSON,
                content={
                    "saved": True,
                    "record_id": record.id,
                    "key": record.key,
                    "category": record.category.value,
                    "extracted_content": record.content,
                    "tags": record.tags,
                },
                metadata=out_env.metadata,
            )
        except Exception as e:
            return Envelope(
                task_id=envelope.task_id,
                sender=self.name,
                status=MessageStatus.ERROR,
                content_type=ContentType.TEXT,
                content=f"Raw: {raw_text} | Error: {e}",
            )


def is_easter_egg_active() -> bool:
    """Active on April 1st or if --comrade flag is passed."""
    today = datetime.date.today()
    return (today.month == 4 and today.day == 1) or ("--comrade" in sys.argv)


def print_banner(store: MemoryStore, has_writer: bool, has_compressor: bool):
    easter_egg = is_easter_egg_active()
    title = Text()

    if easter_egg:
        title.append("☭ COMMUNE AI ", style="bold red")
        title.append("— Cooperative Staged AI Engine\n", style="bold white")
        title.append("Version: 0.2.0 | Engine: Python + llama.cpp | Architecture: Collective Intelligence\n", style="dim")
        border = "red"
    else:
        title.append("◈ COMMUNE AI ", style="bold cyan")
        title.append("— Decentralized Cooperative AI Engine\n", style="bold white")
        title.append("Version: 0.2.0 | Engine: Python + llama.cpp | Architecture: Swarm Intelligence\n", style="dim")
        border = "cyan"

    status_line = Text()
    status_line.append(f"• Collective Archive: {store.count} memory records loaded\n", style="green")
    
    workers = []
    if has_writer:
        workers.append("Memory Writer")
    if has_compressor:
        workers.append("Stage 2 Compressor")
    
    if workers:
        status_line.append(f"• Custom Workers: {', '.join(workers)} (GGUF) [READY]\n", style="bold green")
    else:
        status_line.append("• Workers: Mock components [READY]\n", style="yellow")
    status_line.append("Type your task, or /help for available commands.", style="italic white")

    panel = Panel(
        title + "\n" + status_line,
        border_style=border,
        padding=(1, 2),
    )
    console.print(panel)


def display_memory_table(store: MemoryStore, category_filter: str = None):
    records = store.all_active()
    if category_filter:
        records = [r for r in records if r.category.value == category_filter]

    if not records:
        console.print("[yellow]No memory records found.[/]")
        return

    table = Table(title=f"Commune Persistent Memory ({len(records)} records)", border_style="cyan")
    table.add_column("Category", style="cyan", width=12)
    table.add_column("Key", style="bold green", width=28)
    table.add_column("Content", style="white")
    table.add_column("Tags", style="dim cyan", width=20)
    table.add_column("Ver", style="magenta", width=5)

    for r in records:
        content_str = str(r.content)
        if len(content_str) > 70:
            content_str = content_str[:67] + "..."
        tags_str = ", ".join(r.tags) if r.tags else "-"
        table.add_row(
            r.category.value,
            r.key,
            content_str,
            tags_str,
            f"v{r.version}",
        )

    console.print(table)


def display_routes_table(engine: Engine):
    table = Table(title="Commune Registered Routing Rules", border_style="cyan")
    table.add_column("Trigger Keywords", style="bold yellow")
    table.add_column("Execution Stages (DAG Pipeline)", style="green")

    for route in engine._router._routes:
        kw_str = ", ".join(route.keywords)
        steps_str = " → ".join(f"[bold]{s.component}[/]" + (f" (deps: {s.depends_on})" if s.depends_on else "") for s in route.steps)
        table.add_row(kw_str, steps_str)

    if engine._router._default_steps:
        default_str = " → ".join(f"[bold]{s.component}[/]" for s in engine._router._default_steps)
        table.add_row("[italic]default fallback[/]", default_str)

    console.print(table)


async def main():
    base_dir = Path(__file__).parent
    memory_dir = base_dir / "memory"
    store = MemoryStore(memory_dir)
    store.load()

    # Create Engine
    engine = Engine()

    # 1. Register Mock Pipeline Components
    engine.register(EchoComponent("parser"))
    engine.register(WordCountComponent("analyzer"))
    engine.register(UpperComponent("formatter"))

    # 1b. Register Deterministic System Tools
    read_tool = ReadFileComponent()
    write_tool = WriteFileComponent()
    list_tool = ListDirComponent()
    cmd_tool = RunCommandComponent()
    pipeline_tool = CodebasePipeline("codebase_pipeline", store)

    engine.register(read_tool)
    engine.register(write_tool)
    engine.register(list_tool)
    engine.register(cmd_tool)
    engine.register(pipeline_tool)

    # 1c. Register Autonomous Agent Loop
    agent_loop = AgentLoop(
        tools=[read_tool, write_tool, list_tool, cmd_tool],
        llm=None,
        compressor=CompressorComponent(),
        memory_store=store,
    )
    engine.register(agent_loop)

    # 1d. Register Swarm Council of Agents
    swarm_team = SwarmTeam(memory_store=store)
    engine.register(swarm_team)

    # 2. Register Custom Models if available
    custom_writer_path = base_dir / "llms" / "custom" / "pyramid_memory_writer.Q4_K_M.gguf"
    custom_compressor_path = base_dir / "llms" / "custom" / "commune_compressor.Q4_K_M.gguf"

    has_custom = custom_writer_path.exists()
    has_compressor = custom_compressor_path.exists()

    if has_writer := has_custom:
        llama_writer = LlamaComponent(
            name="qwen-extractor",
            model_path=custom_writer_path,
            system_prompt=(
                "You are the Commune Memory Writer. Extract salient facts into JSON matching schema:\n"
                '{"key": "dot.key", "category": "project"|"knowledge"|"session", "content": "...", "tags": [...]}\n'
                "Output ONLY JSON."
            ),
            n_ctx=2048,
            temperature=0.1,
            max_tokens=256,
        )
        custom_comp = MemoryWriterComponent(llama_writer, store)
        engine.register(custom_comp)

        engine.add_route(
            keywords=["запомни", "память", "save", "remember", "fact", "факт"],
            steps=[PlanStep(component="memory-writer")],
        )

    if has_compressor:
        compressor_comp = LlamaComponent(
            name="compressor",
            model_path=custom_compressor_path,
            system_prompt=(
                "You are the Commune Stage 2 Compressor. Compress raw noisy inputs into a dense bulleted summary."
            ),
            n_ctx=2048,
            temperature=0.2,
            max_tokens=256,
        )
        engine.register(compressor_comp)

        engine.add_route(
            keywords=["сожми", "compress", "summary", "выжимка", "лог"],
            steps=[PlanStep(component="compressor")],
        )

    # Standard routes
    engine.add_route(
        keywords=["analyze", "анализ", "review", "проверь"],
        steps=[
            PlanStep(component="parser"),
            PlanStep(component="analyzer", depends_on=["parser"]),
            PlanStep(component="formatter", depends_on=["analyzer"]),
        ],
    )

    engine.add_route(
        keywords=["count", "words", "посчитай"],
        steps=[PlanStep(component="analyzer")],
    )

    # Tool routes
    engine.add_route(
        keywords=["ls", "директори", "папк", "файлы"],
        steps=[PlanStep(component="list_dir")],
    )
    engine.add_route(
        keywords=["cat", "read", "прочитай", "прочти", "покажи код"],
        steps=[PlanStep(component="read_file")],
    )
    engine.add_route(
        keywords=["run", "sh", "shell", "запусти", "выполни", "bash"],
        steps=[PlanStep(component="run_command")],
    )
    engine.add_route(
        keywords=["pipeline", "пайплайн", "исследуй", "investigate", "staged"],
        steps=[PlanStep(component="codebase_pipeline")],
    )
    engine.add_route(
        keywords=["агент", "agent", "создай файл", "сделай", "запусти команду"],
        steps=[PlanStep(component="agent_loop")],
    )

    engine.set_default_route([PlanStep(component="parser")])

    print_banner(store, has_writer, has_compressor)

    # Prompt session with command history
    history_file = base_dir / ".commune_history"
    session = PromptSession(history=FileHistory(str(history_file)))

    while True:
        try:
            user_input = await session.prompt_async("\ncommune > ")
            user_input = user_input.strip()
            if not user_input:
                continue

            # Command Handling
            if user_input in ["/exit", "exit", "quit", ":q", "q"]:
                farewell = "Goodbye, comrade! ☭" if is_easter_egg_active() else "Shutting down Commune Engine. Goodbye!"
                console.print(f"[cyan]{farewell}[/]")
                break
            elif user_input == "/help":
                console.print(Panel(
                    "[bold cyan]Available Commands:[/]\n"
                    "  [bold green]/memory[/] [category]  - View persistent memory (project, knowledge, session)\n"
                    "  [bold green]/routes[/]              - View current DAG routing table\n"
                    "  [bold green]/ls[/] [path]           - List files in local directory via tool\n"
                    "  [bold green]/read[/] <path>         - Read local file with syntax output\n"
                    "  [bold green]/run[/] <command>       - Execute shell command via Commune tool\n"
                    "  [bold green]/pipeline[/] <command>  - 3-stage pipeline (Execute → Compress → Persist)\n"
                    "  [bold green]/agent[/] <task>        - Autonomous agent loop (decides & runs tools)\n"
                    "  [bold green]/clear[/]               - Clear screen\n"
                    "  [bold green]/exit[/]                - Exit CLI\n\n"
                    "[bold cyan]Example Prompts:[/]\n"
                    "  • [yellow]запомни: Мы закончили сборку инструментов в Phase 3[/]\n"
                    "  • [yellow]/pipeline pytest -v[/]\n"
                    "  • [yellow]/run git status[/]\n"
                    "  • [yellow]/read pyproject.toml[/]\n",
                    title="Help",
                    border_style="cyan"
                ))
                continue
            elif user_input.startswith("/memory") or user_input.startswith("/mem"):
                parts = user_input.split()
                cat = parts[1] if len(parts) > 1 else None
                display_memory_table(store, cat)
                continue
            elif user_input in ["/routes", "/stages", "/graph"]:
                display_routes_table(engine)
                continue
            elif user_input.startswith("/ls"):
                parts = user_input.split(maxsplit=1)
                target_p = parts[1] if len(parts) > 1 else "."
                env = await list_tool.process(Envelope(task_id="cli_ls", sender="user", content=target_p))
                if env.status == MessageStatus.OK and isinstance(env.content, dict):
                    entries = env.content.get("entries", [])
                    table = Table(title=f"Directory: {env.content.get('directory')}", border_style="cyan")
                    table.add_column("Type", style="bold")
                    table.add_column("Name", style="yellow")
                    table.add_column("Size", justify="right", style="green")
                    for e in entries:
                        t = "[blue]DIR[/]" if e.get("is_dir") else "FILE"
                        sz = f"{e.get('size'):,} B" if not e.get("is_dir") else "-"
                        table.add_row(t, e.get("name"), sz)
                    console.print(table)
                else:
                    console.print(f"[bold red]Error:[/] {env.content}")
                continue
            elif user_input.startswith("/read "):
                path_arg = user_input[6:].strip()
                env = await read_tool.process(Envelope(task_id="cli_read", sender="user", content=path_arg))
                if env.status == MessageStatus.OK:
                    console.print(Panel(str(env.content), title=f"File: {path_arg}", border_style="green"))
                else:
                    console.print(f"[bold red]Error:[/] {env.content}")
                continue
            elif user_input.startswith("/run "):
                cmd_arg = user_input[5:].strip()
                with console.status(f"[bold cyan]Executing:[/] {cmd_arg}", spinner="dots"):
                    env = await cmd_tool.process(Envelope(task_id="cli_cmd", sender="user", content=cmd_arg))
                if isinstance(env.content, dict):
                    exit_code = env.content.get("exit_code", 0)
                    stdout = env.content.get("stdout", "")
                    stderr = env.content.get("stderr", "")
                    color = "green" if exit_code == 0 else "red"
                    header = f"Command: {cmd_arg} [exit: {exit_code}] ({env.content.get('duration_ms', 0):.1f}ms)"
                    body = stdout or stderr or "[dim](no output)[/]"
                    console.print(Panel(body, title=header, border_style=color))
                else:
                    console.print(f"[bold red]Error:[/] {env.content}")
                continue
            elif user_input.startswith("/pipeline "):
                cmd_arg = user_input[10:].strip()
                with console.status("[bold cyan]Running 3-Stage Pipeline (Execute → Compress → Persist)...", spinner="dots"):
                    res = await pipeline_tool.execute_tool_and_compress("run_command", cmd_arg)

                m = res.get("metrics", {})
                pipeline_table = Table(box=None, padding=(0, 2), show_header=False)
                pipeline_table.add_row("[bold green]Stage 1 (Tool)[/]", f"{m.get('stage1_ms', 0):.1f}ms", f"Raw: {m.get('original_chars', 0)} chars")
                pipeline_table.add_row("[bold yellow]Stage 2 (Compressor)[/]", f"{m.get('stage2_ms', 0):.1f}ms", f"Compressed: {m.get('compressed_chars', 0)} chars (Ratio: {m.get('compression_ratio')}x)")
                pipeline_table.add_row("[bold cyan]Stage 3 (Memory)[/]", "Committed to", "memory/knowledge.json")
                console.print(Panel(pipeline_table, title="Pipeline Execution Trace", border_style="cyan"))

                comp_content = str(res.get("stage2_compressed", {}).content)
                console.print(Panel(comp_content, title="Compressed Knowledge Result", border_style="green"))
                continue
            elif user_input.startswith("/agent "):
                task_prompt = user_input[7:].strip()
                with console.status(f"[bold cyan]Agent Loop executing:[/] {task_prompt}", spinner="dots"):
                    res = await agent_loop.run_task(task_prompt)

                traces = res.get("traces", [])
                if traces:
                    trace_table = Table(box=None, padding=(0, 2), show_header=False)
                    for tr in traces:
                        trace_table.add_row(
                            f"[bold green]Step {tr['step']}[/]",
                            f"[cyan]{tr['tool']}[/]",
                            f"[yellow]{tr['duration_ms']:.1f}ms[/]",
                            f"[dim]{str(tr['result'])[:80]}...[/]",
                        )
                    console.print(Panel(trace_table, title=f"Agent Action Plan ({res.get('duration_ms', 0):.1f}ms)", border_style="cyan"))

                console.print(Panel(res.get("final_answer", ""), title="Agent Final Result", border_style="green"))
                continue
            elif user_input == "/clear":
                console.clear()
                print_banner(store, has_writer, has_compressor)
                continue

            # Run Task through Commune Multi-Agent Swarm Council
            with console.status("[bold cyan]Council of Agents in session (Planner → Coder → Reviewer)...", spinner="dots"):
                turn = await swarm_team.chat(user_input)

            # 1. Internal Agent Deliberation
            disc_table = Table(box=None, padding=(0, 2), show_header=False)
            for m in turn.internal_discussion:
                disc_table.add_row(
                    f"{m.avatar} [bold]{m.role_title}[/]",
                    f"[dim]{m.content}[/]",
                )

            console.print(Panel(
                disc_table,
                title=f"Internal Swarm Deliberation ({turn.duration_ms:.1f}ms)",
                border_style="dim cyan",
            ))

            # 2. Final User Answer
            console.print(Panel(
                turn.final_answer.content,
                title=f"{turn.final_answer.avatar} {turn.final_answer.role_title}",
                border_style="green",
            ))

        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")


if __name__ == "__main__":
    asyncio.run(main())
