
import os, sys, json, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# === Config ===
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET = "AnthropicHH"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "no_clean_20k"
REPO = "/content/drive/MyDrive/PrefClean/repo"
OUTPUT_DIR = f"{REPO}/models/{DATASET}/{VERSION}"
DATA_PATH = f"{REPO}/datasets/{DATASET}/{VERSION}/train.jsonl"

print(f"=== Training {VERSION} ===")

ds = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Samples: {len(ds)}")

print("Loading base model (4-bit, sdpa)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
# Skip prepare_model_for_kbit_training với gradient_checkpointing để fast hơn
# Vẫn cast head + enable grad cho input embeddings
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,           # Tăng từ 2 → 4 (A100 40GB OK)
    gradient_accumulation_steps=8,           # Giữ effective BS = 16
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_steps=300,                            # Giảm từ 500 → 300
    logging_steps=25,
    save_strategy="no",
    bf16=True,
    optim="adamw_torch",                      # Đổi từ paged_adamw_8bit
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    remove_unused_columns=False,
    report_to="none",
    seed=42,
    gradient_checkpointing=False,             # Tắt để fast (A100 40GB đủ VRAM với LoRA + 4-bit)
)

print("Start training...")
t0 = time.time()
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    peft_config=peft_config,
)
trainer.train()
elapsed = (time.time() - t0) / 60
print(f"\n✅ Trained in {elapsed:.1f} min")

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Saved to {OUTPUT_DIR}")
