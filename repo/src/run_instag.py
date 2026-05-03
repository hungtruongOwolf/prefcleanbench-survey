
import os, sys, json, time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "src")
from macros import DATASETS

SAVE_DIR = "outputs/ins_tag"
SUBSET_SIZE = 20000
CHUNK_SIZE = 5000
BATCH_SIZE = 16  # H100 có 80GB — batch lớn được
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda"

print("Loading InsTagger...")
tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/InsTagger")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # For generation
model = AutoModelForCausalLM.from_pretrained(
    "OFA-Sys/InsTagger",
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval().requires_grad_(False)
print("Model loaded!")

def get_tags_batch(prompts):
    """Get tags for a batch of prompts at once."""
    truncated = [p[:400] for p in prompts]
    inputs = tokenizer(
        truncated,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, out in enumerate(outputs):
        input_len = inputs.input_ids[i].shape[0]
        decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        try:
            if not decoded.endswith('"}]'):
                decoded += '"}]'
            tags = [x["tag"] for x in json.loads(decoded)]
        except:
            tags = []
        results.append(tags)
    return results

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")

    out_file = f"{SAVE_DIR}/{dataset_name}.json"
    ckpt_file = f"{SAVE_DIR}/{dataset_name}_ckpt.json"

    if os.path.exists(out_file):
        print(f"  Already complete, skipping.")
        continue

    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        res = ckpt["res"]
        start = ckpt["next_idx"]
        print(f"  Resuming from example {start}")
    else:
        res = []
        start = 0

    raw = []
    with open(f"datasets/{dataset_name}/no_clean/train.jsonl") as f:
        for line in f:
            raw.append(json.loads(line))
    raw = raw[:SUBSET_SIZE]

    end = min(start + CHUNK_SIZE, SUBSET_SIZE)
    chunk = raw[start:end]
    prompts = [d["prompt"] for d in chunk]

    print(f"  Processing {start} → {end} with batch_size={BATCH_SIZE}...")
    t0 = time.time()

    batches = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]
    for batch in tqdm(batches):
        tags = get_tags_batch(batch)
        res.extend(tags)

    next_idx = end
    with open(ckpt_file, "w") as f:
        json.dump({"res": res, "next_idx": next_idx}, f)

    if next_idx >= SUBSET_SIZE:
        json.dump(res, open(out_file, "w"))
        os.remove(ckpt_file)
        elapsed = time.time() - t0
        print(f"  COMPLETE! {len(res)} tags saved in {elapsed/60:.1f} min")
    else:
        elapsed = time.time() - t0
        print(f"  Chunk done in {elapsed/60:.1f} min — run again for next chunk ({next_idx} → {min(next_idx+CHUNK_SIZE, SUBSET_SIZE)})")

print("\nDone!")
