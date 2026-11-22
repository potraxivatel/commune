# Pyramid

Modular AI orchestration system with staged aggregation.

See [PYRAMID_SPEC.md](PYRAMID_SPEC.md) for the full specification.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the demo (no GPU / no models needed)
python demo.py

# Run tests
pytest -v
```

## Project Status

**Phase 1 — Minimal Engine** (in progress)

- [x] Repository structure
- [x] Communication envelope (typed messages)
- [x] Component interface (abstract base)
- [x] Router (rule-based)
- [x] Scheduler (dependency resolution, parallel execution, timeouts)
- [x] Engine (route → schedule → execute → assemble)
- [x] Mock components for testing
- [x] Tests and demo
- [ ] llama.cpp integration (LlamaComponent — code written, needs model)

## Architecture

```
User → Engine → Router → Scheduler → [Components] → Assembler → Output
```

Components are replaceable. The engine doesn't care if a component
is an LLM, a code parser, or a simple function.
