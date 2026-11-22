# Commune AI

<div align="center">

**Модульный локальный оркестратор составных AI-систем на основе ступенчатой агрегации и долговременной памяти.**

[🇷🇺 Русский](#русская-версия) • [🇬🇧 English](#english-version)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Инференс: llama.cpp](https://img.shields.io/badge/инференс-llama.cpp%20(GGUF)-orange.svg)](https://github.com/ggerganov/llama.cpp)
[![Статус: v0.2.0 Alpha](https://img.shields.io/badge/статус-v0.2.0%20alpha-brightgreen.svg)]()
[![Тесты: 46 пройдено](https://img.shields.io/badge/тесты-46%20пройдено-success.svg)]()

[Архитектура](#архитектура) •
[Центральная гипотеза](#центральная-гипотеза) •
[Быстрый старт](#быстрый-старт) •
[Обученные модели](#обученные-модели) •
[План развития](#план-развития)

</div>

<a id="русская-версия"></a>
---

## Обзор проекта

**Commune AI** — это открытый движок оркестрации составных систем искусственного интеллекта (Compound AI Systems). Проект исследует альтернативу массивным монолитным языковым моделям: координацию коллектива компактных специализированных нейросетей и детерминированного программного кода в рамках **ступенчатого конвейера агрегации данных**.

Вместо загрузки сырых текстов, логов компилятора и целых репозиториев в одно гигантское контекстное окно, Commune организует работу легковесных компонентов, обменивающихся строго типизированными сообщениями через этапы последовательной фильтрации и сжатия.

Система спроектирована с прицелом на **скромное локальное оборудование**: рабочие конфигурации функционируют полностью автономно на одном потребительском GPU (8–24 ГБ VRAM) или современном процессоре (CPU) с использованием квантованных весов формата GGUF.

---

## Центральная гипотеза

> **Гипотеза ступенчатой агрегации (Staged Aggregation):**  
> Скоординированная система специализированных микро-моделей (каждая $\le$ 7B параметров), организованных в топологические стадии, где каждый последующий этап получает **сжатую высокоинформативную выжимку** предыдущих результатов, способна не уступать монолитным моделям при решении структурированных технических задач — потребляя значительно меньше пиковой памяти VRAM, снижая риск галлюцинаций и позволяя независимо заменять отдельные компоненты.

<a id="архитектура"></a>
### Топология конвейера

```
                  ПОЛЬЗОВАТЕЛЬ / ПРИЛОЖЕНИЕ
                              │
                              ▼
                     ЯДРО ДВИЖКА COMMUNE
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        СОВЕТ (РОУТЕР)                 АРХИВ ПАМЯТИ
     (Правила / Обученный DAG)    (Persistent Memory Store)
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
                   ПЛАНИРОВЩИК (DAG Runner)
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
   ЭТАП 1: ШИРОКИЙ (Параллельный)          ЭТАП 1: ШИРОКИЙ (Параллельный)
   Парсер кода / AST-анализатор            Анализатор логов / Раннер тестов
          │                                       │
          └───────────────────┬───────────────────┘
                              │ (Сырой вывод: 500-2000 токенов)
                              ▼
                   ЭТАП 2: СЖАТИЕ (Компрессор)
             Конденсация сути (сокращение в 3x–8x раз)
                              │ (Сжатый бриф: 50-150 токенов)
                              ▼
                   ЭТАП 3: УЗКИЙ (Специалист)
                  Генерация кода / Решение логики
                              │
                              ▼
                   ЭТАП 4: ПРОВЕРКА (Опционально)
                    Линтер / Проверка инвариантов
                              │
                              ▼
                      СБОРКА РЕЗУЛЬТАТА
                              │
                              ▼
                     ВЫВОД ПОЛЬЗОВАТЕЛЮ
```

---

## Ключевые возможности

* **🧩 Модульная абстракция компонентов:** Нейросети, синтаксические парсеры и внешние инструменты реализуют единый интерфейс `Component`. Компоненты декларируют поддерживаемые типы данных и требования к ресурсам через `ComponentManifest`.
* **✉️ Строго типизированные сообщения (`Envelope`):** Обмен данными между этапами изолирован в валидируемые конверты со статусами выполнения, MIME-типами, контролем задержки и трассировкой. Никаких неконтролируемых текстовых утечек.
* **🧠 Независимый архив памяти (`MemoryStore`):** Долговременная память отделена от весов моделей. Разделена на категории `project`, `knowledge` и `session`, сохраняет провенанс (источник, время, авторство), отслеживает конфликты и ведёт непрерывный журнал аудита (`changelog.json`).
* **🗜️ Выделенная модель сжатия контекста (Stage 2):** Собственная микро-модель, обученная агрегировать сырые логи ошибок, диффы файлов и выводы других агентов с подтверждённым коэффициентом сжатия 3x–8.4x без потери номеров строк и идентификаторов.
* **⚡ Локальный инференс GGUF:** Интегрированный бэкенд `llama-cpp-python` для прямого исполнения квантованных моделей на CPU и GPU без внешних API.
* **🖥️ Интерактивный терминальный интерфейс (CLI):** Полнофункциональный терминальный клиент с визуализацией DAG-графа, профилированием шагов выполнения и управлением памятью.

---

## Быстрый старт

### Требования

- Linux или macOS
- Python $\ge$ 3.11
- Компилятор C++ (`gcc`, `clang` или `ninja` для сборки локальных биндингов `llama.cpp`)

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/potraxivatel/commune.git
cd commune

# Создание и активация виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# Установка зависимостей в режиме разработки
pip install -e ".[dev]"
```

### Запуск интерактивной консоли

```bash
./cli.py
```
*(Скрипт автоматически определяет и использует окружение `.venv`, даже если запущен из системного интерпретатора).*

### Запуск набора тестов

```bash
pytest -v
```

Все 46 модульных тестов (ядро, планировщик, DAG, типизация, хранилище памяти) выполняются локально без обращений к сети и без обязательного наличия GPU.

---

<a id="обученные-модели"></a>
## Обученные рабочие модели

В состав Commune входят готовые воспроизводимые скрипты для дообучения специализированных воркеров на бесплатных облачных ускорителях (Google Colab / Kaggle T4) с прямой конвертацией в GGUF:

| Роль компонента | Базовая архитектура | Формат | Вес | Назначение |
| :--- | :--- | :--- | :--- | :--- |
| **Memory Writer** | Qwen 2.5 0.5B Instruct | GGUF Q4_K_M | 380 МБ | Извлечение проектных решений в строгий JSON `MemoryUpdate`. |
| **Stage 2 Compressor** | Qwen 2.5 0.5B Instruct | GGUF Q4_K_M | 380 МБ | Сжатие сырых логов и многоагентных черновиков (коэффициент 8.4x). |

Пайплайны обучения находятся в директории [`training/`](training/):
- `training/generate_dataset.py`: Генератор синтетических выборок.
- `training/train_colab.py`: Скрипт LoRA-обучения через Unsloth с автоматическим квантованием.

---

## Структура репозитория

```
commune/
├── engine/
│   ├── core/
│   │   ├── component.py      # Интерфейсы Component и Manifest
│   │   └── engine.py         # Главный оркестратор задач
│   ├── communication/
│   │   └── envelope.py       # Валидируемый конверт сообщений Envelope
│   ├── inference/
│   │   └── llama_backend.py  # Локальный рантайм GGUF через llama-cpp-python
│   ├── memory/
│   │   ├── records.py        # Схемы записей памяти и провенанса
│   │   └── store.py          # Хранилище памяти с аудитом и разделением категорий
│   ├── router/
│   │   └── router.py         # Маршрутизатор задач в граф выполнения (DAG)
│   └── scheduler/
│       └── scheduler.py      # Планировщик топологического параллельного запуска
├── llms/
│   ├── stock/                # Базовые открытые веса
│   └── custom/               # Дообученные специализированные модели (.gguf)
├── memory/
│   ├── project.json          # Архитектура и структура проекта
│   ├── knowledge.json        # База переиспользуемых технических знаний
│   ├── session.json          # Временный контекст текущей сессии
│   └── changelog.json        # Непрерывный аудит всех изменений памяти
├── training/                 # Скрипты дообучения и генераторы датасетов
├── tests/                    # Интеграционные и модульные тесты
├── cli.py                    # Интерактивный терминальный клиент
├── pyproject.toml            # Конфигурация сборки и зависимости проекта
└── COMMUNE_SPEC.md           # Архитектурная спецификация системы
```

---

<a id="план-развития"></a>
## План развития (Roadmap)

- [x] **Фаза 0: Формальная спецификация** (`COMMUNE_SPEC.md` v0.2)
- [x] **Фаза 1: Минимальное ядро оркестрации**
  - [x] Валидируемый типизированный конверт сообщений
  - [x] Топологический DAG-планировщик с контролем таймаутов
  - [x] Управление жизненным циклом компонентов (`load` / `unload`)
- [x] **Фаза 2: Долговременный архив памяти**
  - [x] Раздельное хранение (`project`, `knowledge`, `session`)
  - [x] Журнал аудита changelog и резервное копирование `.bak`
  - [x] Устойчивость памяти к замене и рестарту моделей
- [x] **Фаза 2.5: Собственные микро-модели**
  - [x] Экстрактор памяти Memory Writer (Qwen 2.5 0.5B GGUF)
  - [x] Модуль сжатия контекста Stage 2 Compressor (Qwen 2.5 0.5B GGUF)
- [x] **Фаза 2.7: Интерактивная консоль и профайлер** (`cli.py`)
- [ ] **Фаза 3: Системные инструменты и анализ кодовой базы**
  - [ ] Инструменты файловой системы (`read_file`, `patch_file`, `git_diff`)
  - [ ] Безопасный запуск терминальных подпроцессов
  - [ ] Автоматический индекс структуры кода (AST)
- [ ] **Фаза 4: Экосистема агентов и сетевые протоколы**
  - [ ] Поддержка Model Context Protocol (MCP)
  - [ ] Автономный агентный цикл с подтверждением действий
- [ ] **Фаза 5: Эмпирические бенчмарки**
  - [ ] Замеры задержки, VRAM и качества в сравнении с 7B/14B монолитами

---

## Лицензия

Проект распространяется под открытой лицензией [MIT License](LICENSE).

<br>
<br>

---
---

<a id="english-version"></a>
# Commune AI (English)

<div align="center">

**A modular, local-first compound AI orchestration engine built around staged aggregation and persistent memory.**

[🇷🇺 Наверх к русской версии](#русская-версия)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Inference: llama.cpp](https://img.shields.io/badge/inference-llama.cpp%20(GGUF)-orange.svg)](https://github.com/ggerganov/llama.cpp)
[![Status: v0.2.0 Alpha](https://img.shields.io/badge/status-v0.2.0%20alpha-brightgreen.svg)]()
[![Tests: 46 passed](https://img.shields.io/badge/tests-46%20passed-success.svg)]()

[Architecture](#architecture) •
[Core Hypothesis](#core-hypothesis) •
[Quick Start](#quick-start) •
[Trained Workers](#trained-workers) •
[Roadmap](#development-roadmap)

</div>

---

## Overview

**Commune AI** is an open-source compound AI system and orchestration engine. It investigates an alternative to monolithic, frontier-scale language models: coordinating multiple smaller, specialized models and deterministic software components across a **staged aggregation pipeline**.

Rather than dumping raw text, compiler dumps, and entire codebases into a single massive context window, Commune coordinates lightweight workers that pass strictly-typed messages through progressive compression stages.

The project is designed from the ground up for **modest local hardware**: practical configurations run completely offline on a single consumer GPU (8–24 GB VRAM) or modern CPU with quantized GGUF weights.

---

## Core Hypothesis

> **Staged Aggregation Hypothesis:**  
> A coordinated system of specialized micro-models (each $\le$ 7B parameters), arranged in topological stages where each stage receives a **compressed, high-density summary** of prior outputs, can match or exceed the practical quality of a monolithic model on structured technical tasks — while consuming substantially less peak VRAM, reducing hallucination, and enabling independent model hot-swapping.

<a id="architecture"></a>
### The Pipeline Topology

```
                  USER / HOST APPLICATION
                             │
                             ▼
                    COMMUNE ENGINE CORE
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       COUNCIL (ROUTER)            COLLECTIVE ARCHIVE
   (Keyword / Learned DAG)       (Persistent Memory Store)
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                    SCHEDULER (DAG Runner)
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
  STAGE 1: WIDE (Parallel)                STAGE 1: WIDE (Parallel)
  Code Parser / AST Extractor             Log Analyzer / Test Runner
         │                                       │
         └───────────────────┬───────────────────┘
                             │ (Raw Outputs: 500-2000 tokens)
                             ▼
                  STAGE 2: COMPRESS (Worker)
            Lossless Summarizer (3x–8x reduction)
                             │ (Condensed Brief: 50-150 tokens)
                             ▼
                  STAGE 3: NARROW (Worker)
                 Specialist Code / Logic Model
                             │
                             ▼
                  STAGE 4: VERIFY (Optional)
                   Linter / Invariant Checker
                             │
                             ▼
                     RESULT ASSEMBLY
                             │
                             ▼
                        USER OUTPUT
```

---

## Key Capabilities

* **🧩 Pluggable Component Abstraction:** Models, AST parsers, and external APIs implement the identical `Component` interface. Components declare their capabilities, input/output types, and advisory VRAM budgets via `ComponentManifest`.
* **✉️ Typed Envelope Messaging:** Inter-component communication uses validated `Envelope` contracts with status codes, MIME content types, latency tracking, and execution traces. No uncontrolled free-form text leakage.
* **🧠 Persistent Collective Archive (MemoryStore):** System memory is independent of model weights. Partitioned into `project`, `knowledge`, and `session` stores with provenance metadata (source, timestamps, author component), conflict detection, automatic backups, and complete changelog audit trails.
* **🗜️ Dedicated Stage 2 Compression:** Built-in micro-model trained specifically to compress raw compiler traces, multi-agent drafts, and file trees with verified 3x–8.4x token volume reduction.
* **⚡ Native GGUF Inference:** Integrated `llama-cpp-python` backend for efficient local CPU/GPU inference with quantized weights.
* **🖥️ Interactive Terminal & Execution Tracing:** Built-in REPL with real-time per-step execution profiling, DAG visualization, and interactive memory inspection.

---

<a id="quick-start"></a>
## Quick Start

### Prerequisites

- Linux or macOS
- Python $\ge$ 3.11
- C++ compiler (`gcc`, `clang`, or `ninja` for local `llama.cpp` bindings)

### Installation

```bash
# Clone the repository
git clone https://github.com/potraxivatel/commune.git
cd commune

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (including dev tools)
pip install -e ".[dev]"
```

### Launch Interactive Shell

```bash
./cli.py
```
*(The CLI automatically detects and runs inside `.venv` even if invoked from system Python).*

### Run Test Suite

```bash
pytest -v
```

All 46 core tests (Engine, Scheduler, DAG, Envelope, MemoryStore) run hermetically without external network calls or GPU requirements.

---

<a id="trained-workers"></a>
## Pre-Trained Workers

Commune ships with reproducible training scripts to fine-tune specialized workers on free cloud accelerators (Google Colab / Kaggle T4) and export them directly to GGUF:

| Worker Role | Base Architecture | Format | Size | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Memory Writer** | Qwen 2.5 0.5B Instruct | GGUF Q4_K_M | 380 MB | Extracts architectural facts into strict `MemoryUpdate` JSON. |
| **Stage 2 Compressor** | Qwen 2.5 0.5B Instruct | GGUF Q4_K_M | 380 MB | Compresses noisy stack traces & multi-agent drafts (8.4x ratio). |

Training pipelines live under [`training/`](training/):
- `training/generate_dataset.py`: Synthetic dataset generator.
- `training/train_colab.py`: Unsloth QLoRA fine-tuning script with direct GGUF quantization.

---

## Project Structure

```
commune/
├── engine/
│   ├── core/
│   │   ├── component.py      # Abstract Component & Manifest interfaces
│   │   └── engine.py         # Main pipeline orchestrator
│   ├── communication/
│   │   └── envelope.py       # Validated typed message envelope
│   ├── inference/
│   │   └── llama_backend.py  # Local GGUF runtime via llama-cpp-python
│   ├── memory/
│   │   ├── records.py        # MemoryRecord, Provenance & Update schemas
│   │   └── store.py          # Category-partitioned JSON store with audit log
│   ├── router/
│   │   └── router.py         # Decision-table router producing execution Plans
│   └── scheduler/
│       └── scheduler.py      # Topological DAG runner with parallel execution
├── llms/
│   ├── stock/                # Stock base model weights
│   └── custom/               # Fine-tuned worker models (.gguf)
├── memory/
│   ├── project.json          # Architecture and structural state
│   ├── knowledge.json        # Reusable facts and technical knowledge
│   ├── session.json          # Ephemeral session context
│   └── changelog.json        # Full provenance audit trail
├── training/                 # LoRA training scripts & dataset generators
├── tests/                    # Unit, regression, and component test suites
├── cli.py                    # Interactive terminal interface
├── pyproject.toml            # Build configuration & dependency definitions
└── COMMUNE_SPEC.md           # Formal architectural specification
```

---

<a id="development-roadmap"></a>
## Development Roadmap

- [x] **Phase 0: Formal Specification** (`COMMUNE_SPEC.md` v0.2)
- [x] **Phase 1: Minimal Engine Core**
  - [x] Validated typed message envelope
  - [x] Topological DAG scheduler with timeout handling
  - [x] Component lifecycle management (`load` / `unload`)
- [x] **Phase 2: Persistent Collective Archive**
  - [x] Partitioned storage (`project`, `knowledge`, `session`)
  - [x] Audit changelog & automatic backup rotations
  - [x] Verified model-replacement durability
- [x] **Phase 2.5: Specialized Micro-Workers**
  - [x] Fine-tuned Memory Writer (Qwen 2.5 0.5B GGUF)
  - [x] Fine-tuned Stage 2 Compressor (Qwen 2.5 0.5B GGUF)
- [x] **Phase 2.7: Interactive Shell & Profiler** (`cli.py`)
- [ ] **Phase 3: System Tools & Codebase Operations**
  - [ ] Deterministic filesystem tools (`read_file`, `patch_file`, `git_diff`)
  - [ ] Safe subprocess execution runner
  - [ ] Live AST project indexer
- [ ] **Phase 4: Agent Ecosystem & Protocols**
  - [ ] Model Context Protocol (MCP) server & client integration
  - [ ] Autonomous tool-calling loop with verification
- [ ] **Phase 5: Empirical Benchmark vs. Monolithic Baselines**
  - [ ] Latency, VRAM, and accuracy evaluation against 7B/14B single-model baselines

---

## License

This project is licensed under the [MIT License](LICENSE).
