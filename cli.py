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
from engine.core.engine import Engine
from engine.inference.llama_backend import LlamaComponent
from engine.memory.records import MemoryCategory, MemoryUpdate
from engine.memory.store import MemoryStore
from engine.router.router import PlanStep
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


def print_banner(store: MemoryStore, has_custom_model: bool):
    title = Text()
    title.append("☭ COMMUNE AI ", style="bold red")
    title.append("— Cooperative Staged AI Engine\n", style="bold white")
    title.append("Version: 0.2.0 | Engine: Python + llama.cpp | Architecture: Collective Intelligence\n", style="dim")

    status_line = Text()
    status_line.append(f"• Collective Archive: {store.count} memory records loaded\n", style="green")
    if has_custom_model:
        status_line.append("• Custom Worker: Qwen 2.5 0.5B GGUF (Memory Writer) [ACTIVE]\n", style="bold green")
    else:
        status_line.append("• Custom Worker: Not loaded (using mock components)\n", style="yellow")
    status_line.append("Type your task, or /help for available commands.", style="italic white")

    panel = Panel(
        title + "\n" + status_line,
        border_style="red",
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

    table = Table(title=f"Pyramid Persistent Memory ({len(records)} records)", border_style="cyan")
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
    table = Table(title="Pyramid Registered Routing Rules", border_style="cyan")
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

    # 2. Register Custom Model if available
    custom_model_path = base_dir / "llms" / "custom" / "pyramid_memory_writer.Q4_K_M.gguf"
    has_custom = custom_model_path.exists()

    if has_custom:
        llama_writer = LlamaComponent(
            name="qwen-extractor",
            model_path=custom_model_path,
            system_prompt=(
                "You are the Pyramid Memory Writer. Extract salient facts into JSON matching schema:\n"
                '{"key": "dot.key", "category": "project"|"knowledge"|"session", "content": "...", "tags": [...]}\n'
                "Output ONLY JSON."
            ),
            n_ctx=2048,
            temperature=0.1,
            max_tokens=256,
        )
        custom_comp = MemoryWriterComponent(llama_writer, store)
        engine.register(custom_comp)

        # Route for memory extraction
        engine.add_route(
            keywords=["запомни", "память", "save", "remember", "fact", "факт"],
            steps=[PlanStep(component="memory-writer")],
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

    engine.set_default_route([PlanStep(component="parser")])

    print_banner(store, has_custom)

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
                console.print("[red]Shutting down Commune Engine. Goodbye, comrade![/]")
                break
            elif user_input == "/help":
                console.print(Panel(
                    "[bold cyan]Available Commands:[/]\n"
                    "  [bold green]/memory[/] [category]  - View persistent memory (project, knowledge, session)\n"
                    "  [bold green]/routes[/]              - View current DAG routing table\n"
                    "  [bold green]/clear[/]               - Clear screen\n"
                    "  [bold green]/exit[/]                - Exit CLI\n\n"
                    "[bold cyan]Example Prompts:[/]\n"
                    "  • [yellow]запомни: Мы закончили сборку CLI интерфейса в Phase 2[/]\n"
                    "  • [yellow]analyze this function: def fib(n): return fib(n-1) + fib(n-2)[/]\n"
                    "  • [yellow]посчитай слова в этой строке текста[/]\n",
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
            elif user_input == "/clear":
                console.clear()
                print_banner(store, has_custom)
                continue

            # Run Task through Pyramid Engine
            with console.status("[bold cyan]Executing through Pyramid stages...", spinner="dots"):
                t0 = time.time()
                result = await engine.run(user_input)
                total_time = (time.time() - t0) * 1000

            # 1. Print Execution Traces
            traces_table = Table(box=None, padding=(0, 2), show_header=False)
            for trace in result.execution.traces:
                icon = "[bold green]✓[/]" if trace.status == "ok" else "[bold red]✗[/]"
                traces_table.add_row(
                    icon,
                    f"[bold cyan]{trace.component}[/]",
                    f"[dim]{trace.status}[/]",
                    f"[yellow]{trace.latency_ms:.1f}ms[/]",
                    f"[dim]{trace.output_size} chars[/]",
                )

            console.print(Panel(
                traces_table,
                title=f"Plan: {' → '.join(s.component for s in result.plan.steps)} (Total: {total_time:.1f}ms)",
                border_style="dim cyan",
            ))

            # 2. Print Output
            out_content = result.output
            if isinstance(out_content, dict) or (isinstance(out_content, str) and out_content.startswith("{")):
                try:
                    formatted_json = json.dumps(json.loads(out_content) if isinstance(out_content, str) else out_content, indent=2, ensure_ascii=False)
                    console.print(Panel(formatted_json, title="Output (JSON)", border_style="green"))
                except Exception:
                    console.print(Panel(str(out_content), title="Output", border_style="green"))
            else:
                console.print(Panel(str(out_content), title="Output", border_style="green"))

        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")


if __name__ == "__main__":
    asyncio.run(main())
