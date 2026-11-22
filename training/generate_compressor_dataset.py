"""
Dataset generator for training the Commune Compressor (Stage 2 Aggregator).

The Compressor's job:
  Take verbose, noisy inputs (compiler errors, tracebacks, git diffs,
  multiple component outputs, long logs) and produce a high-density,
  lossless summary that achieves a 3x-5x token reduction while keeping
  all critical facts, numbers, line numbers, and error causes intact.

Run:
  python generate_compressor_dataset.py
Output:
  compressor_dataset.jsonl
"""

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are the Commune Stage 2 Compressor. Your job is to take raw, noisy, or verbose "
    "multi-component outputs (logs, stack traces, multiple analysis drafts) and compress "
    "them into a dense, high-signal summary.\n\n"
    "Rules:\n"
    "1. Achieve at least a 3x reduction in volume.\n"
    "2. Never drop error messages, file names, line numbers, or key identifiers.\n"
    "3. Eliminate boilerplate, repeated text, and conversational filler.\n"
    "4. Output directly in clean, condensed bullet points."
)

EXAMPLES = [
    # 1. Stack trace / Compiler log compression
    {
        "input": (
            "Traceback (most recent call last):\n"
            '  File "/home/void/Projects/Pyramid/engine/core/engine.py", line 45, in run\n'
            "    exec_result = await scheduler.execute(plan, initial, context)\n"
            '  File "/home/void/Projects/Pyramid/engine/scheduler/scheduler.py", line 82, in execute\n'
            "    output = await component.process(envelope, context)\n"
            '  File "/home/void/Projects/Pyramid/engine/inference/llama_backend.py", line 112, in process\n'
            "    raise RuntimeError('Model not loaded: custom-model. Call load() first.')\n"
            "RuntimeError: Model not loaded: custom-model. Call load() first."
        ),
        "output": (
            "• Error: RuntimeError in llama_backend.py:112 ('Model not loaded: custom-model').\n"
            "• Cause: process() called before load() during scheduler execution.\n"
            "• Call stack: engine.py:45 -> scheduler.py:82 -> llama_backend.py:112."
        )
    },
    # 2. Multi-component merge (Parser + Analyzer + Memory outputs)
    {
        "input": (
            "[code-parser]: File engine/router/router.py has 95 lines. Defines Router class, Route struct, PlanStep. Imports pydantic and typing.\n"
            "[analyzer]: Cyclomatic complexity is low (max 4 in route method). Found 3 keyword routes. Potential bottleneck: linear search over keyword list.\n"
            "[memory-lookup]: Found 1 relevant record: 'project.components.router_strategy' = 'keyword-based decision tables'."
        ),
        "output": (
            "• Component: engine/router/router.py (95 LOC, low complexity <=4).\n"
            "• Architecture: 3 keyword routes (matches memory 'project.components.router_strategy').\n"
            "• Finding: Linear keyword matching could bottleneck under large rule sets."
        )
    },
    # 3. Test suite failure output
    {
        "input": (
            "============================= test session starts ==============================\n"
            "platform linux -- Python 3.14.8, pytest-9.2.0, pluggy-1.7.0\n"
            "rootdir: /home/void/Projects/Pyramid\n"
            "collected 19 items\n"
            "tests/test_engine.py::TestEnvelope::test_create_basic PASSED             [  5%]\n"
            "tests/test_engine.py::TestEnvelope::test_error_factory PASSED            [ 10%]\n"
            "tests/test_engine.py::TestScheduler::test_timeout FAILED                 [ 15%]\n"
            ">       assert result.traces[0].status == 'timeout'\n"
            "E       AssertionError: assert 'error' == 'timeout'\n"
            "tests/test_engine.py:145: AssertionError\n"
            "tests/test_engine.py::TestEngine::test_three_stage_pipeline PASSED       [100%]\n"
            "========================= 1 failed, 18 passed in 0.42s ========================="
        ),
        "output": (
            "• Test Suite: 18 passed, 1 failed (total 19 in 0.42s).\n"
            "• Failure: tests/test_engine.py:145 in TestScheduler::test_timeout.\n"
            "• Assertion: Expected trace status 'timeout', but received 'error'."
        )
    },
    # 4. Long code review diff
    {
        "input": (
            "diff --git a/engine/memory/store.py b/engine/memory/store.py\n"
            "--- a/engine/memory/store.py\n"
            "+++ b/engine/memory/store.py\n"
            "@@ -45,6 +45,12 @@ class MemoryStore:\n"
            "+    def backup(self) -> None:\n"
            "+        if self._path.exists():\n"
            "+            shutil.copy2(self._path, self._path.with_suffix('.bak'))\n"
            "     def save(self) -> None:\n"
            "+        self.backup()\n"
            "         data = [r.model_dump() for r in self._records.values()]\n"
            "         self._path.write_text(json.dumps(data, indent=2))\n"
        ),
        "output": (
            "• Change in engine/memory/store.py: Added automatic backup method.\n"
            "• Behavior: Copies memory store file to .bak suffix before overwriting in save()."
        )
    },
    # 5. Russian technical task description compression
    {
        "input": (
            "Слушай, давай я объясню подробно, как должна работать вторая стадия компрессии. "
            "Короче, когда парсер выдает гигантское дерево файлов или когда тесты падают с кучей логов, "
            "мы не должны отдавать всё это в третью модель. Вторая модель берет весь этот сырой текст, "
            "выкидывает лишний мусор, форматирует всё в короткие пункты, сохраняет имена файлов и строчки, "
            "и выдает короткую выжимку в 3-4 раза меньше по токенам."
        ),
        "output": (
            "• Задача Stage 2 (Compressor): Сжатие вывода первой стадии перед передачей в Stage 3.\n"
            "• Требования: Сокращение объема в 3-4x, удаление шума, обязательное сохранение файлов и строк.\n"
            "• Формат результата: Лаконичный маркированный список фактов."
        )
    },
    # 6. Git status / dirty tree verbose output
    {
        "input": (
            "On branch master\n"
            "Changes not staged for commit:\n"
            '  (use "git add <file>..." to update what will be committed)\n'
            '  (use "git restore <file>..." to discard changes in working directory)\n'
            "	modified:   engine/core/engine.py\n"
            "	modified:   engine/scheduler/scheduler.py\n"
            "Untracked files:\n"
            '  (use "git add <file>..." to include in what will be committed)\n'
            "	tests/test_compressor.py\n"
            "	training/compressor_dataset.jsonl\n"
            "no changes added to commit (use 'git add' to track)"
        ),
        "output": (
            "• Branch: master.\n"
            "• Modified: engine/core/engine.py, engine/scheduler/scheduler.py.\n"
            "• Untracked: tests/test_compressor.py, training/compressor_dataset.jsonl."
        )
    }
]

def augment_dataset(base_examples):
    dataset = []
    # Add base
    for ex in base_examples:
        dataset.append(ex)

    # Prefix variations to teach generalization
    prefixes = [
        "Сделай краткую выжимку: \n",
        "Сожми лог: \n",
        "Compress the following raw output: \n",
        "Extract high-density summary: \n",
        "Stage 1 outputs for compression: \n",
        "Summarize the technical details: \n",
    ]

    for p in prefixes:
        for ex in base_examples:
            dataset.append({
                "input": f"{p}{ex['input']}",
                "output": ex["output"]
            })

    return dataset


def format_for_chatml(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
    }


def main():
    out_dir = Path(__file__).parent
    dataset_raw = augment_dataset(EXAMPLES)
    chatml_data = [format_for_chatml(ex) for ex in dataset_raw]

    out_file = out_dir / "compressor_dataset.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for item in chatml_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(chatml_data)} compressor training samples in {out_file}")


if __name__ == "__main__":
    main()
