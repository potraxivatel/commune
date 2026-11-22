"""
Dataset generator for training the Pyramid Memory Writer component.

Generates training pairs:
  Input: Text/code/dialogue mentioning facts, architecture, rules, or state.
  Output: Strict JSON matching Pyramid's MemoryUpdate schema.

Run:
  python generate_dataset.py
Output:
  dataset.jsonl (format ready for SFTTrainer / Unsloth)
"""

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are the Pyramid Memory Writer. Your sole responsibility is to extract "
    "salient facts, architectural decisions, knowledge, or session state into a strict "
    "JSON object matching the MemoryUpdate schema.\n\n"
    "Schema:\n"
    "{\n"
    '  "key": "dot.separated.key",\n'
    '  "category": "project" | "knowledge" | "session",\n'
    '  "content": "...",\n'
    '  "tags": ["tag1", "tag2"]\n'
    "}\n\n"
    "Output ONLY valid JSON. No conversational text, no markdown backticks."
)

EXAMPLES = [
    # --- Category: PROJECT ---
    {
        "input": "Мы решили, что ядро Pyramid будет написано на Python, а для инференса будем юзать llama.cpp.",
        "output": {
            "key": "project.tech_stack.engine",
            "category": "project",
            "content": "Pyramid engine is implemented in Python, using llama.cpp for local inference.",
            "tags": ["python", "llama.cpp", "tech-stack", "engine"]
        }
    },
    {
        "input": "We define modest hardware as minimum 16GB RAM for CPU inference and 16-24GB VRAM recommended for 7-14B models.",
        "output": {
            "key": "project.hardware.requirements",
            "category": "project",
            "content": {"minimum": "16GB RAM (CPU only, <=3B models)", "recommended": "16-24GB VRAM (7-14B models)"},
            "tags": ["hardware", "vram", "specification"]
        }
    },
    {
        "input": "Архитектура состоит из 4 стадий: Stage 1 Wide (параллельный парсинг), Stage 2 Compress (сжатие вывода), Stage 3 Narrow (генерация ответа) и Stage 4 Verify.",
        "output": {
            "key": "project.architecture.staged_aggregation",
            "category": "project",
            "content": "4 stages: Stage 1 Wide (parallel parsing), Stage 2 Compress (aggregation), Stage 3 Narrow (specialist answer), Stage 4 Verify (review).",
            "tags": ["architecture", "staged-aggregation", "stages"]
        }
    },
    {
        "input": "Основное хранилище памяти в первой версии решено сделать на JSON файлах в папке memory/ с разделением по категориям.",
        "output": {
            "key": "project.storage.memory_format",
            "category": "project",
            "content": "Persistent memory uses flat JSON files partitioned by category with changelog and backups.",
            "tags": ["storage", "memory", "json"]
        }
    },
    {
        "input": "Все сообщения между компонентами передаются через Envelope со статусом, типом контента и метаданными.",
        "output": {
            "key": "project.communication.protocol",
            "category": "project",
            "content": "Inter-component communication is strictly typed via Envelope (task_id, sender, status, content_type, metadata).",
            "tags": ["communication", "envelope", "protocol"]
        }
    },
    {
        "input": "В проекте используется Git-репозиторий в папке /home/void/Projects/Pyramid, ветка master.",
        "output": {
            "key": "project.repo.location",
            "category": "project",
            "content": "/home/void/Projects/Pyramid on branch master",
            "tags": ["git", "repo", "path"]
        }
    },
    {
        "input": "Router в первой версии использует правила на основе ключевых слов (decision table).",
        "output": {
            "key": "project.components.router_strategy",
            "category": "project",
            "content": "Router v0.1 uses keyword-based decision tables to generate execution plans.",
            "tags": ["router", "architecture"]
        }
    },
    {
        "input": "Добавили правило: при изменении записи в памяти старая версия уходит в changelog.json, а файлу делается бэкап .json.bak.",
        "output": {
            "key": "project.memory.backup_policy",
            "category": "project",
            "content": "Memory writes trigger automatic backup to .json.bak and record old versions in changelog.json.",
            "tags": ["memory", "backup", "changelog"]
        }
    },

    # --- Category: KNOWLEDGE ---
    {
        "input": "В Python 3.14 сборка мусора основана на подсчете ссылок с циклическим GC для разрешения циклических зависимостей.",
        "output": {
            "key": "knowledge.python.garbage_collection",
            "category": "knowledge",
            "content": "Python uses reference counting as primary GC, supplemented by cyclic generational collector for reference cycles.",
            "tags": ["python", "memory-management", "gc"]
        }
    },
    {
        "input": "Квантизация GGUF формата Q4_K_M использует 4-битные веса со смешанной точностью для внимания, сохраняя ~98% точности при снижении веса в 4 раза.",
        "output": {
            "key": "knowledge.ml.gguf_q4_k_m",
            "category": "knowledge",
            "content": "GGUF Q4_K_M is a 4-bit medium k-quant scheme offering optimal tradeoff between perplexity degradation and VRAM reduction.",
            "tags": ["gguf", "quantization", "llm", "inference"]
        }
    },
    {
        "input": "Topological sort requires a Directed Acyclic Graph (DAG) and detects cyclic dependencies when not all nodes can be placed.",
        "output": {
            "key": "knowledge.algorithms.topological_sort",
            "category": "knowledge",
            "content": "Topological sorting orders DAG vertices such that every directed edge uv has u before v. Detects cycles if vertices remain unplaced.",
            "tags": ["algorithms", "dag", "scheduler"]
        }
    },
    {
        "input": "Pydantic v2 написан на Rust (pydantic-core) и выполняет валидацию структур данных в 5-20 раз быстрее версии v1.",
        "output": {
            "key": "knowledge.libraries.pydantic_v2",
            "category": "knowledge",
            "content": "Pydantic v2 core engine is implemented in Rust (pydantic-core), providing 5x-20x validation speedup over v1.",
            "tags": ["pydantic", "python", "rust"]
        }
    },
    {
        "input": "LoRA (Low-Rank Adaptation) замораживает базовую модель и внедряет обучаемые низкоранговые матрицы в веса внимания (W = W0 + B*A).",
        "output": {
            "key": "knowledge.ml.lora_adaptation",
            "category": "knowledge",
            "content": "LoRA decomposes weight updates into low-rank matrices delta_W = B * A with rank r << d, drastically reducing trainable parameters.",
            "tags": ["lora", "peft", "fine-tuning"]
        }
    },

    # --- Category: SESSION ---
    {
        "input": "Сейчас мы завершили Phase 2 и готовимся обучать модель Qwen 2.5 0.5B на Colab.",
        "output": {
            "key": "session.progress.current_step",
            "category": "session",
            "content": "Completed Phase 2 (Memory). Currently preparing training pipeline for Qwen 2.5 0.5B fine-tuning on Colab.",
            "tags": ["status", "phase2", "training"]
        }
    },
    {
        "input": "Пользователь выбрал Google Colab/Kaggle для дообучения модели и запросил формат GGUF.",
        "output": {
            "key": "session.user_preferences.environment",
            "category": "session",
            "content": "Target training environment: Colab/Kaggle GPU; Target model export format: GGUF.",
            "tags": ["user-preference", "colab", "gguf"]
        }
    },
    {
        "input": "В текущей сессии выполнено 46 тестов, все успешно пройдены.",
        "output": {
            "key": "session.test_suite.status",
            "category": "session",
            "content": "Test suite passed: 46 total tests (19 engine, 27 memory).",
            "tags": ["tests", "ci", "status"]
        }
    }
]

# Simple data augmentation function to produce 100+ training variations
def augment_dataset(base_examples):
    dataset = []
    # Add base examples
    for ex in base_examples:
        dataset.append(ex)

    # Prefix variations
    prefixes = [
        "Зафиксируй факт: ",
        "Пожалуйста, запиши в память: ",
        "Save this to memory: ",
        "Update project state: ",
        "Важная информация: ",
        "Обрати внимание: ",
        "Note for the system: ",
    ]

    for p in prefixes:
        for ex in base_examples:
            augmented = {
                "input": f"{p}{ex['input']}",
                "output": ex["output"]
            }
            dataset.append(augmented)

    return dataset


def format_for_chatml(example):
    """Formats an example into Qwen ChatML format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)}
        ]
    }


def main():
    out_dir = Path(__file__).parent
    dataset_raw = augment_dataset(EXAMPLES)
    chatml_data = [format_for_chatml(ex) for ex in dataset_raw]

    out_file = out_dir / "dataset.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for item in chatml_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(chatml_data)} training samples in {out_file}")


if __name__ == "__main__":
    main()
