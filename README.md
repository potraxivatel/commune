# Pyramid

Modular AI orchestration system with staged aggregation.

See [PYRAMID_SPEC.md](PYRAMID_SPEC.md) for the full architectural specification.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Interactive Terminal Interface
./cli.py

# Or run tests
pytest -v
```

## Features

- **Engine Core:** Directed Acyclic Graph (DAG) scheduler with topological ordering, timeouts, and parallel execution.
- **Typed Communication:** Every inter-component message is validated via `Envelope` (JSON-schemas, status, token counts, latency).
- **Persistent Memory:** Category-partitioned JSON store (`project`, `knowledge`, `session`) with provenance, versioning, conflict detection, and audit changelog.
- **Custom Fine-tuned GGUF Model:** Includes an in-house trained **Qwen 2.5 0.5B Instruct** running locally on CPU via `llama-cpp-python` as a dedicated **Memory Writer** extracting structured facts in ~2.0s.
- **Interactive CLI:** Beautiful terminal UI powered by Rich & prompt_toolkit (`/memory`, `/routes`, execution latency breakdown).

## Interactive Commands in CLI

Inside `./cli.py`:
- `запомни: <любой факт или правило>` — модель извлечет факт и сохранит в персистентную память.
- `analyze <код или текст>` — прогон через 3-стадийный пайплайн (`parser` → `analyzer` → `formatter`).
- `посчитай слова: <текст>` — быстрый запуск легковесного аналитического компонента.
- `/memory` — просмотр сохраненных фактов и категорий.
- `/routes` — текущая таблица маршрутизации и зависимостей между стадиями.
- `/help` — список доступных команд.
