
import os, sys, json, time, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

sys.path.insert(0, "src")
from macros import DATASETS

MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
NUM_MODELS = 8
SUBSET_SIZE = 20000

progress_file = "outputs/rwgap_progress.json"
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

if os.path.exists(progress_file):
    with open(progress_file) as f:
        done = json.load(f)
else:
    done = {}

print(f"Already done: {list(done.keys())}")

def mark_done(key):
    done[key] = True
    with open(progress_file, "w") as f:
        json.dump(done, f)

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")

    raw = []
    with open(f"datasets/{dataset_name}/no_clean/train.jsonl") as f:
        for line in f:
            raw.append(json.loads(line))
    raw = raw[:SUBSET_SIZE]

    for idx in range(NUM_MODELS):
        key = f"{dataset_name}_RwGap_{idx}"
        model_dir = f"models/{dataset_name}/RwGap_{idx}"

        if done.get(key):
            print(f"  [SKIP] {key}")
            continue

        print(f"\n  Training {key}...")
        t0 = time.time()

        random.seed(idx * 42)
        sample = random.sample(raw, int(len(raw) * 0.8))

        train_data = Dataset.from_list([
            {"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]}
            for d in sample
        ])

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )

        dpo_config = DPOConfig(
            output_dir=model_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            bf16=True,
            logging_steps=20,
            save_steps=999999,
            seed=idx * 42,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_data,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        trainer.train()
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f} min → {model_dir}")
        mark_done(key)

        del model, trainer
        torch.cuda.empty_cache()

print("\nAll RwGap models trained!")
