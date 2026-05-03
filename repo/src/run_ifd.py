
import os, sys, json, time
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, "src")
from llama3_tokenizer import CustomLlama3Tokenizer
from macros import DATASETS

SAVE_DIR = "outputs/ifd"
DATA_DIR = "datasets"
SUBSET_SIZE = 20000
CHUNK_SIZE = 5000   # Process 5K per run — safe for 2hr session
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda"

print("Loading Llama-3-8B in 4-bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
tokenizer = CustomLlama3Tokenizer("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True
)
model.eval().requires_grad_(False)
print("Model loaded!")

def get_log_prob(encoding):
    with torch.no_grad():
        out = model(
            encoding.input_ids.to(device),
            attention_mask=encoding.attention_mask.to(device)
        ).logits.float().log_softmax(-1)[0]
    return torch.tensor([out[i, idx].item() for i, idx in enumerate(encoding.input_ids[0])])

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")

    out_file = f"{SAVE_DIR}/{dataset_name}.tsv"
    ckpt_file = f"{SAVE_DIR}/{dataset_name}_ckpt.json"

    # Load checkpoint
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        res = ckpt["res"]
        start = ckpt["next_idx"]
        print(f"  Resuming from example {start}")
    else:
        res = []
        start = 0

    if os.path.exists(out_file):
        print(f"  Already complete, skipping.")
        continue

    # Load data
    raw = []
    with open(f"{DATA_DIR}/{dataset_name}/no_clean/train.jsonl") as f:
        for line in f:
            raw.append(json.loads(line))
    raw = raw[:SUBSET_SIZE]

    end = min(start + CHUNK_SIZE, SUBSET_SIZE)
    print(f"  Processing examples {start} → {end} of {SUBSET_SIZE}")
    t0 = time.time()

    for i, data in enumerate(tqdm(raw[start:end], initial=start, total=end)):
        try:
            prompt = tokenizer(data["prompt"], max_length=256, return_tensors="pt",
                             truncation=True, return_offsets_mapping=True)
            chosen = tokenizer(data["chosen"], max_length=256, return_tensors="pt",
                             truncation=True, return_offsets_mapping=True)
            reject = tokenizer(data["rejected"], max_length=256, return_tensors="pt",
                             truncation=True, return_offsets_mapping=True)

            prompt_text = tokenizer.batch_decode(prompt.input_ids, skip_special_tokens=True)[0]
            chosen_text = tokenizer.batch_decode(chosen.input_ids, skip_special_tokens=True)[0]
            reject_text = tokenizer.batch_decode(reject.input_ids, skip_special_tokens=True)[0]

            if len(chosen_text) <= 1:
                res.append((10.0, -10.0))
                continue

            prompt_chosen = tokenizer(prompt_text + chosen_text, max_length=512,
                                    return_tensors="pt", truncation=True,
                                    return_offsets_mapping=True)
            prompt_reject = tokenizer(prompt_text + reject_text, max_length=512,
                                    return_tensors="pt", truncation=True,
                                    return_offsets_mapping=True)

            resp_start = prompt_chosen.char_to_token(len(prompt_text) + 1)
            if resp_start is None:
                res.append((10.0, -10.0))
                continue
            resp_start -= 1

            pc_lp = get_log_prob(prompt_chosen)
            pr_lp = get_log_prob(prompt_reject)
            c_lp  = get_log_prob(chosen)
            r_lp  = get_log_prob(reject)

            ifd_c = torch.exp(-pc_lp[resp_start:].mean()) / torch.exp(-c_lp.mean())
            ifd_r = torch.exp(-pr_lp[resp_start:].mean()) / torch.exp(-r_lp.mean())
            res.append((ifd_c.item(), (ifd_c - ifd_r).item()))

        except Exception as e:
            res.append((10.0, -10.0))

    # Save checkpoint
    next_idx = end
    with open(ckpt_file, "w") as f:
        json.dump({"res": res, "next_idx": next_idx}, f)
    print(f"  Checkpoint saved at {next_idx}/{SUBSET_SIZE}")

    # If fully done — save final file
    if next_idx >= SUBSET_SIZE:
        np.savetxt(out_file, np.array(res), delimiter="\t")
        os.remove(ckpt_file)
        print(f"  COMPLETE! Saved {len(res)} scores → {out_file}")
    else:
        elapsed = time.time() - t0
        print(f"  Done chunk in {elapsed/60:.1f} min")
        print(f"  Run this script again to continue next chunk ({next_idx} → {min(next_idx+CHUNK_SIZE, SUBSET_SIZE)})")

print("\nDone for this session!")
