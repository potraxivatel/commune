# Pyramid --- Project Specification

**Document:** `PYRAMID_SPEC.md`\
**Version:** 0.2 --- revised\
**Status:** Draft\
**Date:** 2026-11-22

------------------------------------------------------------------------

## 1. Overview

Pyramid is an experimental modular AI orchestration system built around
the idea that useful AI capabilities do not have to live inside one
large, monolithic language model.

Instead, Pyramid coordinates multiple smaller, specialized models and
ordinary software components. Each component handles a limited
responsibility, passes structured results to other components, and
contributes to producing a final answer or completing a task.

The system is intended to make capable AI workflows more accessible on
modest hardware. It prioritizes modularity, replaceable components,
persistent memory, and the ability to experiment with different model
arrangements.

Pyramid has three distinct layers:

-   **Orchestration architecture:** how the engine divides tasks between
    components, routes information, and assembles results.
-   **Engine/runtime:** the software that schedules components, manages
    communication, and combines their outputs.
-   **Ecosystem:** conventions, interfaces, and packaging that allow
    compatible models and tools to work with the engine.

These layers are related parts of one project, but they should remain
conceptually distinct.

### 1.1 What Pyramid is not

Pyramid is not a new neural-network architecture in the academic sense.
It does not propose new layer types, attention mechanisms, or training
procedures. It is a **system-level architecture** --- a way to
coordinate existing models into workflows that are more capable than any
single model in the system.

## 2. Relationship to existing systems

Several projects already coordinate multiple LLMs or agents. Pyramid
shares surface-level goals with them but differs in emphasis.

| System | Focus | How Pyramid differs |
|---|---|---|
| **LangChain / LangGraph** | Chaining LLM calls via a graph of steps | Pyramid treats models as replaceable packages with manifests, not as opaque API calls. Memory is a first-class persistent subsystem, not a bolted-on retriever. |
| **CrewAI / AutoGen** | Multi-agent role-play and conversation | Pyramid does not simulate agent personas. Components have typed interfaces and return structured data, not conversational text. |
| **DSPy** | Programmatic prompt optimization | DSPy optimizes prompts for a fixed model. Pyramid optimizes the *composition* of models and non-LLM code. These are complementary. |
| **Mixture of Experts (MoE)** | Routing within one model's forward pass | MoE operates inside a single model at the layer level. Pyramid operates outside models at the system level, coordinating separate processes. |

The core bet of Pyramid is that **progressive staged aggregation with
persistent memory** (see §7) can outperform a single model or a flat
pipeline on resource-constrained hardware. If experiments do not support
this, Pyramid has no reason to exist as a separate project.

## 3. Goals

### 3.1 Primary goals

1.  **Modularity** --- divide work between smaller components instead of
    depending on one model for everything.
2.  **Hardware accessibility** --- run practical configurations on a
    single consumer GPU (8--24 GB VRAM) or Apple Silicon with unified
    memory (16--32 GB). See §3.2.
3.  **Replaceability** --- allow individual models to be replaced or
    upgraded without rebuilding the entire system.
4.  **Persistent memory** --- keep useful information outside individual
    models so it survives model replacement and changes.
5.  **Specialization** --- let components focus on particular tasks,
    such as understanding a request, analyzing code, reviewing a result,
    or writing to memory.
6.  **Experimentation** --- make it possible to compare different model
    arrangements and processing strategies.
7.  **Extensibility** --- support project-provided models and,
    eventually, compatible third-party models.

### 3.2 Hardware target

"Modest hardware" means:

-   **Minimum:** 16 GB RAM, no discrete GPU --- CPU inference only
    (quantized models, ≤ 3B parameters per component).
-   **Recommended:** 16--24 GB VRAM (e.g., RTX 3090/4070 Ti Super) or
    Apple Silicon with 32 GB unified memory --- allows 7--14B quantized
    models.
-   **Stretch:** multi-GPU or 48 GB+ VRAM --- larger models, more
    parallel components.

The minimum target must always remain functional, even if slower. No
configuration should silently require hardware beyond what the user has
declared.

### 3.3 Success criteria

Pyramid should eventually demonstrate that it can:

-   complete a useful, clearly defined task using multiple coordinated
    components;
-   retain project information across sessions and model changes;
-   run at least one practical configuration on the minimum hardware
    target;
-   replace one model without losing the system's persistent memory;
-   explain or record how a result was produced well enough to debug the
    workflow;
-   **outperform the defined baseline** (see §7.5) on at least one task,
    measured by quality, latency, or resource use.

These are goals to test, not claims that the system already achieves
them.

## 4. Core principles

### 4.1 No single model owns the system

The engine, memory, settings, and project data must not depend on the
internal state of one particular LLM.

### 4.2 Models are components

A model is a replaceable worker with a defined role and an interface.
The engine decides when to call it and how to use its output.

### 4.3 Memory is independent

Persistent memory belongs to the system, not to an individual model. A
model may read from memory or propose new information, but it must not
be the only place where that information exists.

### 4.4 Prefer clear responsibilities

A component should have a clear purpose. Some responsibilities may be
handled by ordinary code rather than an LLM when that is more reliable.

### 4.5 Keep the design testable

New architectural ideas should be treated as hypotheses until
experiments support them. Pyramid should be compared against simpler
baselines instead of assuming that more modules automatically mean
better results.

## 5. High-level architecture

The following is a conceptual map, not a final implementation layout.

``` text
                    USER / APPLICATION
                            |
                            v
                    PYRAMID ENGINE
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
          ROUTER         SCHEDULER       MEMORY
             |              |              |
             +--------------+--------------+
                            |
                            v
                    MODEL COMPONENTS
              +-------------+-------------+
              |             |             |
           ANALYSIS       SPECIALIST     REVIEW
              |             |             |
              +-------------+-------------+
                            |
                            v
                     RESULT ASSEMBLY
                            |
                            v
                       USER OUTPUT
```

The exact modules, their number, and their execution order remain open
for research.

### 5.1 Engine

The Engine is the coordinating software layer. Its responsibilities are
expected to include:

-   accepting tasks;
-   deciding which components may be useful;
-   scheduling and running components;
-   passing information between components;
-   reading and writing persistent memory through controlled interfaces;
-   collecting outputs and producing a final result;
-   handling errors, timeouts, and unavailable components.

### 5.2 Router

The Router determines which component or workflow should handle a task.

**Initial strategy (rule-based):**

The first Router will use a decision table: a mapping from task
categories to component sets. Task categories are determined by keyword
matching and simple classifiers (e.g., regex on the input, file
extension detection). This is intentionally simple.

**Possible evolution:**

A learned router (a small classifier trained on task-to-component
mappings) can be explored later. It should be adopted only if it
measurably improves routing accuracy over the rule-based approach on a
held-out evaluation set. The router itself should never require a large
model.

**Routing output:**

The Router produces a **plan**: an ordered list of components to invoke,
with declared dependencies between them. Example:

``` json
{
  "task_id": "t-001",
  "steps": [
    {"component": "code-parser", "depends_on": []},
    {"component": "analysis-7b", "depends_on": ["code-parser"]},
    {"component": "review-3b", "depends_on": ["analysis-7b"]}
  ]
}
```

### 5.3 Scheduler

The Scheduler manages execution: which components run, in what order,
and whether some work can happen in parallel.

**Responsibilities:**

-   Read the plan from the Router and build a dependency graph (DAG).
-   Execute independent steps in parallel when hardware allows.
-   Enforce a VRAM budget: do not load more models simultaneously than
    the declared hardware limit permits.
-   Handle component failure: retry once, then mark the step as failed
    and propagate the failure to dependent steps.
-   Enforce per-step timeouts. A component that exceeds its timeout is
    killed and treated as failed.
-   Record execution traces (component, start time, end time, status,
    output size) for every step.

**VRAM management:**

On resource-constrained hardware, models cannot all be loaded at once.
The Scheduler must unload a model before loading the next one. The
initial strategy is FIFO: unload the oldest loaded model. Model load
and unload times must be included in latency measurements.

### 5.4 Communication

Components exchange structured messages rather than relying only on
informal text. Every message between components uses a typed envelope:

``` json
{
  "task_id": "string",
  "step_id": "string",
  "sender": "string (component name)",
  "status": "ok | error | partial",
  "content_type": "text | json | code | binary_ref",
  "content": "...",
  "metadata": {
    "token_count": 0,
    "latency_ms": 0,
    "confidence": null
  }
}
```

The exact fields and serialization format may change, but the principle
is fixed: **every inter-component message is structured and typed**.
Free-form text is a value inside `content`, not the envelope itself.

### 5.5 Result assembly

The system needs a defined stage that combines component outputs,
resolves disagreements where possible, and prepares the response.

**Strategies (to be tested):**

-   **Last-writer-wins:** the final component in the pipeline produces
    the output directly.
-   **Merge by section:** each component contributes a labeled section;
    the assembler concatenates and deduplicates.
-   **Voting:** when multiple components produce alternatives, select
    the majority answer or the highest-confidence one.
-   **LLM summarizer:** a small model reads all component outputs and
    produces a coherent response.

The initial implementation should use the simplest strategy that works
for the chosen first task. The assembler is itself a component and can
be replaced.

## 6. Memory System

### 6.1 Purpose

Memory is a persistent subsystem owned by Pyramid Engine. It stores
useful information independently of the models currently installed.

Replacing, removing, or updating an LLM must not erase the system's
saved memory.

### 6.2 Memory writer

Pyramid may include a dedicated memory-writing component. Its role is to
inspect available information and propose useful memory updates.

This component is not a user-facing assistant. Its job is to extract,
organize, and record information --- not to answer the user.

The writer does not have to be an LLM. Depending on the information,
Pyramid may use:

-   ordinary code for parsing and validation;
-   language models for extracting meaning or summarizing;
-   project-specific analyzers, such as source-code parsers.

### 6.3 Controlled writes

A memory writer should not be allowed to overwrite memory arbitrarily.

A proposed update should pass through the Engine, which can:

1.  validate its structure;
2.  retain its source and timestamp;
3.  detect possible conflicts with existing information;
4.  decide whether to add, update, or mark information as outdated;
5.  record the accepted change.

Memory should preserve provenance where practical: where a fact came
from and when it was recorded.

### 6.4 Proposed memory categories

-   **Project Memory** --- project structure, files, dependencies,
    design decisions, and current state.
-   **Knowledge Memory** --- reusable facts, explanations, and technical
    knowledge.
-   **Session Memory** --- temporary information relevant to the current
    task or session.

These categories are a starting proposal and may change.

### 6.5 Storage formats

No single format needs to store everything.

Possible roles:

-   **Markdown / TXT** --- readable notes, summaries, and project
    descriptions.
-   **JSON** --- structured facts and metadata.
-   **SQLite** --- indexing, lookup, relationships, and efficient
    querying.
-   **Graph representation** --- dependencies and relationships when a
    graph is useful.

A `.dot` file is specifically a Graphviz graph-description format. It
may be useful for visualizing relationships, but it should not
automatically become the main memory format.

The initial implementation should favor formats that are easy to
inspect, back up, and edit. **The first prototype should use JSON files
in a flat directory**, with SQLite introduced only when querying becomes
a bottleneck.

## 7. Staged aggregation --- the core research hypothesis

This section describes the central idea that justifies Pyramid as a
project. Everything else (engine, memory, packaging) is infrastructure.
If this hypothesis does not hold, Pyramid reduces to a generic
orchestrator with no unique value.

### 7.1 The hypothesis

> A system of N small models (each ≤ 7B parameters), arranged in stages
> where each stage receives a **compressed summary** of the previous
> stage's outputs, can match or exceed the quality of a single larger
> model (14--70B) on structured tasks --- while using less peak VRAM and
> allowing independent component replacement.

This is a testable claim. It may be false.

### 7.2 How staged aggregation works (proposed)

``` text
STAGE 1 — WIDE (parallel)
  Component A (code parser)       → structured AST + facts
  Component B (doc reader)        → extracted requirements
  Component C (context retriever) → relevant memory fragments

        ↓ each output is independently structured

STAGE 2 — COMPRESS
  Aggregator (code or small LLM)  → merge, deduplicate, resolve
                                     contradictions, produce a
                                     condensed context document

        ↓ condensed context (smaller than sum of Stage 1 outputs)

STAGE 3 — NARROW (sequential)
  Specialist model               → produce the answer using only
                                    the condensed context

        ↓ draft answer

STAGE 4 — VERIFY (optional)
  Reviewer model                 → check the answer against the
                                    condensed context, flag errors
```

The "pyramid" shape: Stage 1 is wide (many components, much raw data),
each subsequent stage works with less but more refined information.

**Key design constraint:** the condensed context produced by Stage 2
must be **smaller** (in tokens) than the sum of Stage 1 outputs. If
compression does not reduce volume, the system degenerates into simple
concatenation and loses its advantage.

### 7.3 Compression strategies to test

1.  **Extractive:** select the most relevant sentences/facts from each
    component's output using TF-IDF, BM25, or embedding similarity to
    the original query.
2.  **Abstractive:** use a small LLM (≤ 3B) to summarize each
    component's output into a fixed token budget.
3.  **Structured merge:** when components produce typed JSON (e.g., AST
    nodes, dependency lists), merge programmatically without an LLM.
4.  **Hybrid:** use code for structured data, a small LLM for
    unstructured text.

Each strategy should be measured by **information loss** (does the final
answer degrade compared to giving all raw outputs to a single model?)
and **compression ratio** (how much smaller is the condensed context?).

### 7.4 Research questions

-   What compression ratio is achievable before quality degrades?
-   Does the pyramid shape outperform a flat pipeline (all components
    dump output into one final model)?
-   At what task complexity does multi-stage become worthwhile?
    (Simple questions may not benefit.)
-   How should the system detect that compression has lost critical
    information?
-   Does adding a verification stage (Stage 4) improve quality enough
    to justify its latency cost?

### 7.5 Baselines for comparison

Every Pyramid experiment must be compared against at least one of:

1.  **Single-model baseline:** the same task given to one model (the
    largest that fits in the same VRAM budget) with the full context in
    its prompt.
2.  **Flat-pipeline baseline:** the same components, but all outputs
    concatenated and passed to a final model without compression.
3.  **Simple RAG baseline:** retrieve relevant chunks, pass to one
    model.

If Pyramid does not beat baseline #1 on quality **or** latency **or**
resource use, the staged approach is not justified for that task.

### 7.6 Persistent and compressed context

A separate research direction. For a system focused on code assistance,
Pyramid could maintain a **living project summary** in memory: a
structured, pre-compressed representation of the codebase (file map,
dependency graph, key abstractions). When a task arrives, the Router
injects only the relevant slice of this summary into component prompts,
instead of re-reading raw files each time.

This reduces per-task token use and allows smaller models to operate
within their context window. The tradeoff is that the summary may become
stale. Staleness detection (comparing summary timestamps against file
modification times) is required.

## 8. Model components

### 8.1 Model responsibilities

Possible roles include:

-   request understanding;
-   planning and task decomposition;
-   code generation;
-   code analysis;
-   result review;
-   summarization;
-   memory extraction.

These are examples, not a mandatory list of separate models. One model
may perform several roles, and some roles may be implemented without an
LLM.

### 8.2 Model package

A model package may eventually contain:

-   model weights;
-   a manifest describing the model;
-   configuration;
-   a system prompt or role description, where applicable;
-   compatibility information;
-   optional tests or examples.

A possible future layout:

``` text
llms/
├── stock/
│   └── example-model/
│       ├── weights/
│       ├── manifest.json
│       ├── config.json
│       └── system_prompt.txt
└── custom/
```

This is illustrative. The final package format must be designed
alongside the communication protocol.

### 8.3 Compatibility

Pyramid should not assume that any arbitrary LLM can be plugged in and
immediately work well.

A third-party model may need an adapter, a compatible wrapper,
additional training, or evaluation before it can reliably participate.

The long-term goal is an open compatibility path rather than a
permanently closed collection of project-owned models.

## 9. Settings and configuration

Settings should be separate from model weights and persistent memory.

Potential configuration areas:

-   engine behavior;
-   installed and enabled models;
-   runtime and hardware limits (declared VRAM, max parallel components);
-   memory behavior and storage locations;
-   logging and diagnostics.

Configuration formats and file names will be decided during
implementation.

## 10. Repository layout proposal

A possible high-level repository structure:

``` text
Pyramid/
├── engine/
│   ├── core/
│   ├── router/
│   ├── scheduler/
│   ├── communication/
│   ├── memory/
│   └── inference/
├── llms/
│   ├── stock/
│   └── custom/
├── memory/
│   ├── projects/
│   ├── knowledge/
│   └── sessions/
├── settings/
├── protocols/
├── tests/
├── benchmarks/
├── docs/
└── PYRAMID_SPEC.md
```

This is a proposal, not a requirement to create every directory
immediately. The repository should grow when a component has a real
purpose.

## 11. Development roadmap

### Phase 0 --- Specification

-   [x] Record the initial concept.
-   [x] Define hardware targets.
-   [x] Define baselines for comparison.
-   [x] Formulate the core research hypothesis.
-   [ ] Agree on the project's scope and terminology.
-   [ ] Identify the first task Pyramid should solve.
-   [ ] List the most important unknowns.

### Phase 1 --- Minimal Engine

-   [ ] Create the initial repository structure.
-   [ ] Define the component interface (input/output message schema).
-   [ ] Define the communication envelope (§5.4).
-   [ ] Implement the Engine loop: accept task → route → schedule →
        execute → assemble.
-   [ ] Run one component through the Engine.
-   [ ] Pass a structured result between two components.
-   [ ] Add basic logging, execution traces, and error handling.

### Phase 2 --- Persistent Memory

-   [ ] Define the first memory record format (JSON files).
-   [ ] Implement reading and writing independently of LLMs.
-   [ ] Add source and timestamp metadata.
-   [ ] Add validation before accepting memory updates.
-   [ ] Verify that memory survives application restarts and model
        replacement.

### Phase 3 --- First useful workflow

-   [ ] Choose one narrow task (proposed: "answer a question about a
        local codebase of 20--50 files").
-   [ ] Build a 3-stage workflow: parse → compress → answer.
-   [ ] Implement the single-model baseline and the flat-pipeline
        baseline.
-   [ ] Evaluate on a repeatable set of 20+ examples.
-   [ ] Measure: answer quality (manual grading), latency, peak VRAM,
        total tokens.
-   [ ] Compare Pyramid against both baselines. Record results.
-   [ ] **Decision gate:** if Pyramid does not match or beat at least
        one baseline on at least one metric, stop and analyze why before
        proceeding.

### Phase 4 --- Model ecosystem

-   [ ] Define model manifests.
-   [ ] Define the communication protocol.
-   [ ] Add model discovery and configuration.
-   [ ] Create an adapter path for compatible external models.
-   [ ] Add compatibility tests.

### Phase 5 --- Architecture experiments

-   [ ] Test staged processing and information aggregation (§7).
-   [ ] Test compression strategies (§7.3).
-   [ ] Test different memory representations.
-   [ ] Measure quality, latency, RAM, and VRAM use.
-   [ ] Keep only complexity that demonstrates a real benefit.

## 12. Non-goals for the initial version

The first version will not attempt to:

-   reproduce the capabilities of a frontier-scale general-purpose LLM;
-   support every model format or inference backend;
-   train a large foundation model from scratch;
-   implement a complex autonomous agent before basic orchestration
    works;
-   introduce many specialized models without a demonstrated need;
-   assume that more stages or more models automatically improve
    quality;
-   compete with cloud-based API orchestrators (LangChain, etc.) on
    tasks where a single API call to a large model is sufficient.

## 13. Open questions

These questions should be answered through discussion and experiments
rather than guessed in advance.

1.  What is the first concrete task Pyramid should perform well?
    *Proposed: codebase Q&A over a 20--50 file local project.*
2.  Should the first prototype use existing small models (e.g.,
    Qwen 2.5 3B, Llama 3 8B via llama.cpp / Ollama), simple mock
    components, or both?
3.  What information belongs in persistent memory, and what should
    remain temporary?
4.  Which memory format should be used for the first working prototype?
    *Proposed: JSON files.*
5.  How should components describe their capabilities and limitations?
6.  How should the Engine handle conflicting results?
7.  Which local inference runtimes should be supported first?
    *Candidates: llama.cpp, Ollama, vLLM (GPU), MLX (Apple Silicon).*
8.  What measurements will determine whether Pyramid is actually useful?
    *Proposed: answer quality vs. single-model baseline, latency,
    peak VRAM.*

## 14. Project rule

**Build the smallest working version that can test the central idea.**

Do not implement speculative complexity just because it appears in this
document. Update this specification when a decision is made, an
experiment produces evidence, or the project's direction changes.
