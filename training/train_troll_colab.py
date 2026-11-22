"""
Commune Worker #4: Toxic Senior Dev / Troll Agent (Qwen 2.5 0.5B with Unsloth -> GGUF)

Trained specifically on savage, witty IT developer banter, dva-ch humor, and sarcastic clapbacks
when user attacks or whines.

Outputs:
  - commune_troll.Q4_K_M.gguf
"""

import sys
import subprocess
from pathlib import Path

# Auto-install unsloth if needed
try:
    import unsloth
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "trl", "peft", "accelerate", "bitsandbytes"])

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 512
dtype = None
load_in_4bit = True

model_name = "unsloth/Qwen2.5-0.5B-Instruct"

print(f"Loading base model for Toxic Senior Dev: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

def load_dataset_from_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    formatted_texts = []
    for item in data:
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)
    
    return Dataset.from_dict({"text": formatted_texts})

dataset = load_dataset_from_jsonl("troll_dataset.jsonl")
print(f"Toxic Troll dataset loaded: {len(dataset)} examples")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=50,
        learning_rate=3e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        seed=3407,
        output_dir="outputs_troll",
    ),
)

print("Starting training of Toxic Senior Dev...")
trainer.train()
print("Training complete! Quantizing to GGUF...")

model.save_pretrained_gguf(
    "commune_troll",
    tokenizer,
    quantization_method="q4_k_m"
)
print("Export complete! commune_troll.Q4_K_M.gguf is ready.")
