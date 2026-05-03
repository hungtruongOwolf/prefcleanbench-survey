
import os, sys, json, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.insert(0, "src")
from macros import DATASETS

REWARD_MODEL = "LxzGordon/URM-LLaMa-3.1-8B"
BATCH_SIZE = 8
GEN_DIR = "generations"
RES_DIR = "results"
os.makedirs(RES_DIR, exist_ok=True)

VERSIONS = [
    "no_clean_20k",
    "llm_judge_r",   "llm_judge_f",
    "vote_all_r",    "vote_all_f",
    "vote_maj_r",    "vote_maj_f",
    "ins_tag_cmp",   "ins_tag_div",
    "ifd_r_0.2",     "ifd_gap_r_0.2",  "ifd_gap_f_0.2",
    "rw_gap_r_0.2",  "rw_gap_f_0.2",
]

PROGRESS_FILE = "outputs/eval_progress.json"
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        done = json.load(f)
else:
    done = {}

def mark_done(key, score):
    done[key] = score
    with open(PROGRESS_FILE, "w") as f:
        json.dump(done, f, indent=2)

print("Loading URM reward model...")
tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval().requires_grad_(False)
print("Reward model loaded!")

def score_responses(prompts, responses):
    """Score a list of prompt+response pairs."""
    chats = [
        [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        for p, r in zip(prompts, responses)
    ]
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in chats
    ]
    all_scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(model.device)
        with torch.no_grad():
            scores = model(**enc).logits.squeeze(-1)
        all_scores.extend(scores.float().cpu().tolist())
    return np.array(all_scores)

# Store all results
all_results = {}

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")
    all_results[dataset_name] = {}

    os.makedirs(f"{RES_DIR}/{dataset_name}", exist_ok=True)

    for version in VERSIONS:
        key = f"{dataset_name}__{version}"
        gen_file = f"{GEN_DIR}/{dataset_name}/{version}.json"

        if not os.path.exists(gen_file):
            print(f"  [MISSING] {version}")
            continue

        if done.get(key):
            score = done[key]
            print(f"  [SKIP] {version}: {score:.4f}")
            all_results[dataset_name][version] = score
            continue

        gens = json.load(open(gen_file))
        prompts = [g["prompt"] for g in gens]
        responses = [g["output"] for g in gens]

        scores = score_responses(prompts, responses)
        avg_score = float(scores.mean())

        np.savetxt(f"{RES_DIR}/{dataset_name}/{version}.tsv", scores, delimiter="\t")
        mark_done(key, avg_score)
        all_results[dataset_name][version] = avg_score
        print(f"  ✅ {version}: {avg_score:.4f}")

# Print final summary table
print("\n" + "="*60)
print("FINAL RESULTS — Average Reward")
print("="*60)
print(f"{'Version':<20} {'AnthropicHH':>15} {'UltraFeedback':>15}")
print("-"*60)

baseline_hh = all_results.get("AnthropicHH", {}).get("no_clean_20k", 0)
baseline_uf = all_results.get("UltraFeedback", {}).get("no_clean_20k", 0)

for version in VERSIONS:
    hh = all_results.get("AnthropicHH", {}).get(version, float("nan"))
    uf = all_results.get("UltraFeedback", {}).get(version, float("nan"))
    marker = " ← baseline" if version == "no_clean_20k" else ""
    print(f"  {version:<20} {hh:>12.4f}   {uf:>12.4f}{marker}")

print("-"*60)
print(f"  Baseline HH={baseline_hh:.4f}, UF={baseline_uf:.4f}")

json.dump(all_results, open(f"{RES_DIR}/summary.json", "w"), indent=2)
print(f"\nSaved to {RES_DIR}/summary.json")
