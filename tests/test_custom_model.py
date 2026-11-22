"""
Test script for the custom trained Pyramid Memory Writer model.

Tests:
  1. Load pyramid_memory_writer.Q4_K_M.gguf via LlamaComponent
  2. Send an unstructured factual statement
  3. Expect valid MemoryUpdate JSON
  4. Parse and commit the fact into MemoryStore
  5. Verify persistence in memory/
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.communication.envelope import ContentType, Envelope
from engine.inference.llama_backend import LlamaComponent
from engine.memory.records import MemoryCategory, MemoryUpdate
from engine.memory.store import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("pyramid.test_model")


async def main():
    model_path = Path(__file__).parent.parent / "llms" / "custom" / "pyramid_memory_writer.Q4_K_M.gguf"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("=" * 60)
    print("  Testing Custom Pyramid Memory Writer (Qwen 2.5 0.5B GGUF)")
    print("=" * 60)
    print(f"Model: {model_path.name} ({model_path.stat().st_size / (1024*1024):.1f} MB)\n")

    # 1. Initialize Memory Store
    memory_dir = Path(__file__).parent.parent / "memory"
    store = MemoryStore(memory_dir)
    store.load()
    print(f"Existing memory records: {store.count}")

    # 2. Initialize Component
    system_prompt = (
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

    writer = LlamaComponent(
        name="custom-memory-writer",
        model_path=model_path,
        system_prompt=system_prompt,
        n_ctx=2048,
        n_gpu_layers=0,  # CPU inference
        temperature=0.1,  # Low temperature for strict JSON
        max_tokens=256,
    )

    print("Loading model into memory...")
    await writer.load()
    print("Model loaded successfully!\n")

    # 3. Test prompts
    test_cases = [
        "Мы решили, что тесты производительности и замеры VRAM будут проводиться в Phase 5.",
        "GGUF формат Q4_K_M весит всего 380 МБ и идеально работает на скромном железе.",
    ]

    for i, test_input in enumerate(test_cases, 1):
        print("─" * 60)
        print(f"Test {i}: {test_input}")
        print("─" * 60)

        in_env = Envelope(
            task_id=f"test-{i}",
            sender="user",
            content_type=ContentType.TEXT,
            content=test_input,
        )

        out_env = await writer.process(in_env)
        print(f"Model output ({out_env.metadata.latency_ms:.1f}ms):")
        raw_content = str(out_env.content).strip()
        print(raw_content)

        # 4. Parse JSON & Commit to Memory Store
        try:
            # Strip markdown code fences if any
            clean_json = raw_content
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            parsed = json.loads(clean_json)
            category_val = parsed.get("category", "project")
            try:
                cat = MemoryCategory(category_val)
            except ValueError:
                cat = MemoryCategory.PROJECT

            update = MemoryUpdate(
                key=parsed["key"],
                category=cat,
                content=parsed["content"],
                tags=parsed.get("tags", []),
                source=f"component:{writer.name}",
                component=writer.name,
            )

            record = store.add(update, force=True)
            print(f"✓ Successfully stored in memory: [{record.category.value}] {record.key} (v{record.version})")
        except Exception as e:
            print(f"✗ Failed to parse or store: {e}")

        print()

    # 5. Persist to disk
    store.save()
    print("=" * 60)
    print(f"Memory saved to disk. Total active records: {store.count}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
