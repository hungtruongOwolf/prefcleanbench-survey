
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, "src")
from macros import DATASETS

SAVE_DIR = "outputs/voting"
SUBSET_SIZE = 20000
MAX_LENGTH = 256  # Reduced from 512
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda"

REWARD_MODELS = {
    # GRM 3B with 4-bit quantization — ~3x faster
    "GRM": {
        "path": "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft",
        "batch_size": 128,
        "use_4bit": True,
    },
    # DeBERTa 434MB — extremely fast, no quantization needed
    "DeBERTa": {
        "path": "OpenAssistant/reward-model-deberta-v3-large-v2",
        "batch_size": 256,
        "use_4bit": False,
    },
}

def build_texts_deberta(data):
    """DeBERTa uses simple prompt+response format, no chat template."""
    chosen_texts, rejected_texts = [], []
    for d in data:
        chosen_texts.append(d["prompt"].strip() + " " + d["chosen"].strip())
        rejected_texts.append(d["prompt"].strip() + " " + d["rejected"].strip())
    return chosen_texts, rejected_texts

def build_texts_chat(data, tokenizer):
    """Chat template for GRM."""
    chosen_texts, rejected_texts = [], []
    for d in data:
        c = [{"role": "user", "content": d["prompt"].strip()},
             {"role": "assistant", "content": d["chosen"].strip()}]
        r = [{"role": "user", "content": d["prompt"].strip()},
             {"role": "assistant", "content": d["rejected"].strip()}]
        try:
            chosen_texts.append(
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            )
            rejected_texts.append(
                tokenizer.apply_chat_template(r, tokenize=False, add_generation_prompt=False)
            )
        except:
            chosen_texts.append(d["prompt"].strip() + " " + d["chosen"].strip())
            rejected_texts.append(d["prompt"].strip() + " " + d["rejected"].strip())
    return chosen_texts, rejected_texts

def tokenize_texts(texts, tokenizer, batch_size=2000, label=""):
    """Tokenize in chunks to avoid OOM."""
    all_ids, all_masks = [], []
    chunks = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    for chunk in tqdm(chunks, desc=f"  Tokenizing {label}"):
        enc = tokenizer(
            chunk, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding=True
        )
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
    return all_ids, all_masks

def score_all(model, all_ids, all_masks, batch_size, label=""):
    """Score in batches with progress bar."""
    all_scores = []
    t0 = time.time()
    total = sum(ids.shape[0] for ids in all_ids)
    pbar = tqdm(total=total, desc=f"  Scoring {label}", unit="ex")

    for ids_chunk, mask_chunk in zip(all_ids, all_masks):
        n = ids_chunk.shape[0]
        for i in range(0, n, batch_size):
            ids = ids_chunk[i:i+batch_size].to(device)
            mask = mask_chunk[i:i+batch_size].to(device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask).logits
                if out.ndim == 2:
                    out = out[:, 0]
            all_scores.extend(out.float().cpu().tolist())
            pbar.update(ids.shape[0])

            done = len(all_scores)
            elapsed = time.time() - t0
            if elapsed > 0:
                speed = done / elapsed
                eta = (total - done) / speed
                pbar.set_postfix(speed=f"{speed:.0f}ex/s", ETA=f"{eta/60:.1f}min")

    pbar.close()
    return all_scores

for dataset_name in DATASETS:
    print(f"\n{'='*40}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*40}")

    raw = []
    with open(f"datasets/{dataset_name}/no_clean/train.jsonl") as f:
        for line in f:
            raw.append(json.loads(line))
    raw = raw[:SUBSET_SIZE]
    print(f"Examples: {len(raw)}")

    for rm_name, rm_config in REWARD_MODELS.items():
        out_file = f"{SAVE_DIR}/{dataset_name}_{rm_name}.tsv"
        if os.path.exists(out_file):
            print(f"\n  [SKIP] {rm_name} already done")
            continue

        print(f"\n  Loading {rm_name} ({rm_config['path']})...")
        tokenizer = AutoTokenizer.from_pretrained(rm_config["path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if rm_config["use_4bit"]:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                rm_config["path"], quantization_config=bnb,
                device_map="auto", trust_remote_code=True,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                rm_config["path"], dtype=torch.float16,
                device_map="auto", trust_remote_code=True,
            )
        model.eval().requires_grad_(False)

        t_start = time.time()
        bs = rm_config["batch_size"]

        # Build texts
        if rm_name == "DeBERTa":
            chosen_texts, rejected_texts = build_texts_deberta(raw)
        else:
            chosen_texts, rejected_texts = build_texts_chat(raw, tokenizer)

        # Tokenize
        chosen_ids, chosen_masks = tokenize_texts(chosen_texts, tokenizer, label="chosen")
        rejected_ids, rejected_masks = tokenize_texts(rejected_texts, tokenizer, label="rejected")

        # Score
        chosen_scores = score_all(model, chosen_ids, chosen_masks, bs, "chosen")
        rejected_scores = score_all(model, rejected_ids, rejected_masks, bs, "rejected")

        # Save
        results = np.array(list(zip(chosen_scores, rejected_scores)))
        np.savetxt(out_file, results, delimiter="\t")

        elapsed = time.time() - t_start
        print(f"  Done in {elapsed/60:.1f} min — {len(results)} scores saved")

        del model, tokenizer
        torch.cuda.empty_cache()

print("\n=== All voting done! ===")
for f in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(f"{SAVE_DIR}/{f}")
    print(f"  {f} — {size:,} bytes")
