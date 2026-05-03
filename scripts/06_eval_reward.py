"""
Step 6: Compute average gold reward for all generated responses.
Uses a held-out reward model (URM-LLaMa-3.1-8B).

Usage: python scripts/06_eval_reward.py [--dataset anthropic_hh]
"""
import os
import json
import glob
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REWARD_MODEL = "LxzGordon/URM-LLaMa-3.1-8B"
BATCH_SIZE = 8


def load_reward_model(device):
    """Load the gold reward model."""
    print(f"Loading reward model: {REWARD_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def score_responses(model, tokenizer, generations, device):
    """Score a list of prompt-response pairs."""
    scores = []
    for i in tqdm(range(0, len(generations), BATCH_SIZE), desc="  Scoring"):
        batch = generations[i : i + BATCH_SIZE]
        texts = [f"{g['prompt']}\n{g['response']}" for g in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.squeeze(-1).cpu().tolist()

        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_reward_model(device)

    gen_dir = "outputs/generations"
    results_dir = "outputs/eval"
    os.makedirs(results_dir, exist_ok=True)

    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.jsonl")))
    if args.dataset:
        gen_files = [f for f in gen_files if args.dataset in f]

    all_results = {}
    for gen_file in gen_files:
        name = os.path.basename(gen_file).replace(".jsonl", "")
        print(f"\nScoring: {name}")

        with open(gen_file) as f:
            generations = [json.loads(l) for l in f]

        scores = score_responses(model, tokenizer, generations, device)
        avg_reward = sum(scores) / len(scores)

        all_results[name] = {"avg_reward": round(avg_reward, 3), "n_samples": len(scores)}
        print(f"  Avg. Reward: {avg_reward:.3f} (n={len(scores)})")

    out_path = os.path.join(results_dir, "reward_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
