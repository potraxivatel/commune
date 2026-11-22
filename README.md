# Commune AI

<div align="center">

**A modular, local-first compound AI orchestration engine built around staged aggregation and persistent memory.**

[English](README.md) • [Русский](README_RU.md)

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

## Quick Start

### Prerequisites

- Linux or macOS
- Python $\ge$ 3.11
- C++ compiler (`gcc`, `clang`, or `ninja` for local `llama.cpp` bindings)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/commune-ai.git
cd commune-ai

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
commune-ai/
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
