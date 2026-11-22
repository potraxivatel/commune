"""
Pyramid Memory Writer: Fine-tuning Qwen 2.5 0.5B with Unsloth -> GGUF

Run this script on Google Colab or Kaggle (with free T4 GPU enabled).
Training takes ~5-10 minutes.
Outputs:
  - pyramid_memory_writer.Q4_K_M.gguf
"""

# Cell 1: Install Unsloth and dependencies
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.29" trl peft accelerate bitsandbytes

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Model Configuration
max_seq_length = 1024
dtype = None  # None for auto-detection (Float16 for T4, Bfloat16 for Ampere+)
load_in_4bit = True  # 4-bit QLoRA uses < 3GB VRAM

model_name = "unsloth/Qwen2.5-0.5B-Instruct"

print(f"Loading base model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # 0 is optimized for Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Load and Format Dataset
def load_dataset_from_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    formatted_texts = []
    for item in data:
        # Format using Qwen ChatML template
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)
    
    return Dataset.from_dict({"text": formatted_texts})

# In Colab, upload dataset.jsonl or use the generated path
dataset = load_dataset_from_jsonl("dataset.jsonl")
print(f"Dataset loaded: {len(dataset)} examples")

# 4. Train Model
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
        warmup_steps=10,
        max_steps=60,  # ~3 epochs on 128 samples
        learning_rate=3e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()
print(f"Training complete! Loss: {trainer_stats.training_loss:.4f}")

# 5. Export to GGUF (Q4_K_M)
print("Saving and quantizing to GGUF (Q4_K_M)...")
model.save_pretrained_gguf(
    "pyramid_memory_writer",
    tokenizer,
    quantization_method="q4_k_m"
)

print("SUCCESS! File 'pyramid_memory_writer-unsloth.Q4_K_M.gguf' created.")
print("Download it and put into: Projects/Pyramid/llms/custom/")
