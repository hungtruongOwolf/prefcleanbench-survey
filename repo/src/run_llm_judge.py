
import os, sys, json, time
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, "src")
from macros import DATASETS

SAVE_DIR = "outputs/llm_judge"
SUBSET_SIZE = 20000
CHUNK_SIZE = 5000
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda"

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

JUDGE_PROMPT = """You are an expert evaluator. Score the following response to the given prompt on a scale of 1 to 10 for helpfulness, harmlessness, and quality.
Respond with ONLY a single integer from 1 to 10. No explanation.

Prompt: {prompt}

Response: {response}

Score:"""

print("Loading Llama-3-8B-Instruct in 4-bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True
)
model.eval().requires_grad_(False)
print("Model loaded!")

def score_response(prompt, response):
    text = JUDGE_PROMPT.format(
        prompt=prompt[:300],
        response=response[:300]
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                      max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
    try:
        score = float("".join(c for c in decoded if c.isdigit() or c == ".")[:3])
        return min(max(score, 1.0), 10.0)
    except:
        return 5.0

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")

    out_file = f"{SAVE_DIR}/{dataset_name}.tsv"
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
    print(f"  Processing examples {start} → {end} of {SUBSET_SIZE}")
    t0 = time.time()

    for data in tqdm(raw[start:end]):
        try:
            score_chosen   = score_response(data["prompt"], data["chosen"])
            score_rejected = score_response(data["prompt"], data["rejected"])
            res.append((score_chosen, score_rejected))
        except:
            res.append((5.0, 5.0))

    next_idx = end
    with open(ckpt_file, "w") as f:
        json.dump({"res": res, "next_idx": next_idx}, f)

    if next_idx >= SUBSET_SIZE:
        np.savetxt(out_file, np.array(res), delimiter="\t")
        os.remove(ckpt_file)
        print(f"  COMPLETE! Saved {len(res)} scores → {out_file}")
    else:
        elapsed = time.time() - t0
        print(f"  Chunk done in {elapsed/60:.1f} min — run again for next chunk ({next_idx} → {min(next_idx+CHUNK_SIZE, SUBSET_SIZE)})")

print("\nDone for this session!")
