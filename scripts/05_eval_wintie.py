"""
Step 5: Evaluate win-tie rate using GPT-4o as judge.
Compares each cleaned model against the no-cleaning baseline.

Requires: OPENAI_API_KEY environment variable.
Usage: python scripts/05_eval_wintie.py [--dataset anthropic_hh] [--dry-run]
"""
import os
import sys
import json
import glob
import random
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PrefCleanBench"))

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

JUDGE_MODEL = "gpt-4o"
NUM_PAIRS = 200
SEED = 42

JUDGE_PROMPT = """You are evaluating two AI assistant responses to a user prompt.

User Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response is better? Consider helpfulness, accuracy, and relevance.
Answer with exactly one of: A, B, or Tie."""


def judge_pair(client, prompt, response_a, response_b):
    """Ask GPT-4o to judge a pair, testing both orderings."""
    results = []
    for ra, rb in [(response_a, response_b), (response_b, response_a)]:
        msg = JUDGE_PROMPT.format(prompt=prompt, response_a=ra, response_b=rb)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": msg}],
            max_tokens=8,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        results.append(answer)
    return results


def compute_wintie(judgments):
    """Compute win-tie rate from judgment pairs."""
    wins = ties = total = 0
    for fwd, rev in judgments:
        # Forward: A=clean, B=baseline. Reverse: A=baseline, B=clean.
        clean_wins_fwd = fwd == "A"
        clean_wins_rev = rev == "B"
        tie_fwd = fwd == "TIE"
        tie_rev = rev == "TIE"

        if clean_wins_fwd and clean_wins_rev:
            wins += 1
        elif tie_fwd or tie_rev:
            ties += 1
        elif clean_wins_fwd or clean_wins_rev:
            wins += 0.5
            ties += 0.5
        total += 1

    return (wins + ties) / total if total > 0 else 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = OpenAI() if not args.dry_run else None
    random.seed(SEED)

    gen_dir = "outputs/generations"
    results_dir = "outputs/eval"
    os.makedirs(results_dir, exist_ok=True)

    # Find all generation files
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.jsonl")))
    if args.dataset:
        gen_files = [f for f in gen_files if args.dataset in f]

    # Group by dataset and algorithm
    baselines = [f for f in gen_files if "no_clean" in f]

    all_results = {}
    for baseline_file in baselines:
        baseline_name = os.path.basename(baseline_file).replace(".jsonl", "")
        parts = baseline_name.split("_")
        algo = parts[0]
        dataset = "_".join(parts[3:]).replace("_no_clean", "")

        # Find corresponding cleaned models
        cleaned_files = [
            f for f in gen_files
            if f != baseline_file
            and algo in os.path.basename(f)
            and dataset in os.path.basename(f)
        ]

        for clean_file in cleaned_files:
            clean_name = os.path.basename(clean_file).replace(".jsonl", "")
            print(f"Evaluating: {clean_name} vs {baseline_name}")

            if args.dry_run:
                print("  (dry run, skipping API calls)")
                continue

            # Load generations
            with open(baseline_file) as f:
                baseline_gens = [json.loads(l) for l in f]
            with open(clean_file) as f:
                clean_gens = [json.loads(l) for l in f]

            # Sample test pairs
            indices = random.sample(range(min(len(baseline_gens), len(clean_gens))), NUM_PAIRS)

            judgments = []
            for idx in tqdm(indices, desc="  Judging"):
                result = judge_pair(
                    client,
                    baseline_gens[idx]["prompt"],
                    clean_gens[idx]["response"],
                    baseline_gens[idx]["response"],
                )
                judgments.append(result)

            wintie = compute_wintie(judgments)
            all_results[clean_name] = {"wintie": wintie, "n_pairs": NUM_PAIRS}
            print(f"  WinTie: {wintie:.3f}")

    # Save results
    out_path = os.path.join(results_dir, "wintie_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
