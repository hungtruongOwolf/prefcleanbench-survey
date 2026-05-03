
import os, sys, json, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from dpo_inference import DPOInference
from macros import DATASETS

SAVE_DIR = "outputs/rw_gap"
SUBSET_SIZE = 20000
NUM_MODELS = 4
BATCH_SIZE = 8
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
os.makedirs(SAVE_DIR, exist_ok=True)

progress_file = "outputs/rwgap_score_progress.json"
if os.path.exists(progress_file):
    with open(progress_file) as f:
        done = json.load(f)
else:
    done = {}

def mark_done(key):
    done[key] = True
    with open(progress_file, "w") as f:
        json.dump(done, f)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")

    raw = []
    with open(f"datasets/{dataset_name}/no_clean/train.jsonl") as f:
        for line in f:
            raw.append(json.loads(line))
    raw = raw[:SUBSET_SIZE]
    dataset = Dataset.from_list(raw)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for idx in range(NUM_MODELS):
        key = f"{dataset_name}_score_{idx}"
        out_file = f"{SAVE_DIR}/{dataset_name}_{idx}.tsv"

        if done.get(key) or os.path.exists(out_file):
            print(f"  [SKIP] model {idx}")
            continue

        model_dir = f"models/{dataset_name}/RwGap_{idx}"
        if not os.path.exists(model_dir):
            print(f"  [MISSING] {model_dir}")
            continue

        print(f"  Loading base + LoRA model {idx}...")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        # Load base in 4-bit, then apply LoRA on top
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_dir)
        model = model.merge_and_unload()  # Merge LoRA weights

        # Load reference model (base only) in 4-bit
        ref_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto"
        )

        dpo = DPOInference(model, tokenizer, ref_model)

        results = []
        for batch in tqdm(dataloader, desc=f"  model {idx}"):
            try:
                tokenized = dpo.tokenize_row(batch)
                chosen, rejected = dpo.inference_step(tokenized)
                results.extend([(c.item(), r.item()) for c, r in zip(chosen, rejected)])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results.extend([(0.0, 0.0)] * BATCH_SIZE)

        np.savetxt(out_file, np.array(results), delimiter="\t")
        elapsed = time.time() - t0
        print(f"  Saved {len(results)} scores in {elapsed/60:.1f} min")
        mark_done(key)

        del model, ref_model, base, dpo
        torch.cuda.empty_cache()

print("\nAll scoring done!")
