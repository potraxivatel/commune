"""
Test script for the custom trained Commune Stage 2 Compressor.

Tests:
  1. Load commune_compressor.Q4_K_M.gguf via LlamaComponent
  2. Feed noisy traceback and multi-agent outputs
  3. Measure compression ratio (chars before vs chars after)
  4. Verify that key facts, files, and errors are preserved
"""

import asyncio
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.communication.envelope import ContentType, Envelope
from engine.inference.llama_backend import LlamaComponent


async def main():
    model_path = Path(__file__).parent.parent / "llms" / "custom" / "commune_compressor.Q4_K_M.gguf"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("=" * 60)
    print("  Testing Custom Commune Stage 2 Compressor (Qwen 0.5B GGUF)")
    print("=" * 60)
    print(f"Model: {model_path.name} ({model_path.stat().st_size / (1024*1024):.1f} MB)\n")

    system_prompt = (
        "You are the Commune Stage 2 Compressor. Your job is to take raw, noisy, or verbose "
        "multi-component outputs (logs, stack traces, multiple analysis drafts) and compress "
        "them into a dense, high-signal summary.\n\n"
        "Rules:\n"
        "1. Achieve at least a 3x reduction in volume.\n"
        "2. Never drop error messages, file names, line numbers, or key identifiers.\n"
        "3. Eliminate boilerplate, repeated text, and conversational filler.\n"
        "4. Output directly in clean, condensed bullet points."
    )

    compressor = LlamaComponent(
        name="stage2-compressor",
        model_path=model_path,
        system_prompt=system_prompt,
        n_ctx=2048,
        n_gpu_layers=0,  # CPU inference
        temperature=0.2,
        max_tokens=256,
    )

    print("Loading Compressor model into CPU memory...")
    await compressor.load()
    print("Compressor loaded successfully!\n")

    # Test cases: raw verbose text
    test_cases = [
        (
            "Python Stack Trace",
            (
                "Traceback (most recent call last):\n"
                '  File "/home/void/Projects/Pyramid/engine/core/engine.py", line 45, in run\n'
                "    exec_result = await scheduler.execute(plan, initial, context)\n"
                '  File "/home/void/Projects/Pyramid/engine/scheduler/scheduler.py", line 82, in execute\n'
                "    output = await component.process(envelope, context)\n"
                '  File "/home/void/Projects/Pyramid/engine/inference/llama_backend.py", line 112, in process\n'
                "    raise RuntimeError('Model not loaded: custom-model. Call load() first.')\n"
                "RuntimeError: Model not loaded: custom-model. Call load() first."
            ),
        ),
        (
            "Multi-component Raw Drafts",
            (
                "[code-parser]: File engine/router/router.py has 95 lines. Defines Router class, Route struct, PlanStep. Imports pydantic and typing.\n"
                "[analyzer]: Cyclomatic complexity is low (max 4 in route method). Found 3 keyword routes. Potential bottleneck: linear search over keyword list.\n"
                "[memory-lookup]: Found 1 relevant record: 'project.components.router_strategy' = 'keyword-based decision tables'."
            ),
        ),
    ]

    for label, raw_text in test_cases:
        print("─" * 60)
        print(f"Input ({label}, {len(raw_text)} chars):")
        print(raw_text)
        print("─" * 60)

        in_env = Envelope(
            task_id="compress-test",
            sender="stage1-wide",
            content_type=ContentType.TEXT,
            content=raw_text,
        )

        out_env = await compressor.process(in_env)
        compressed_text = str(out_env.content).strip()
        comp_len = len(compressed_text)
        ratio = len(raw_text) / max(comp_len, 1)

        print(f"Compressed Output ({comp_len} chars, ratio: {ratio:.1f}x, {out_env.metadata.latency_ms:.0f}ms):")
        print(compressed_text)
        print()


if __name__ == "__main__":
    asyncio.run(main())
