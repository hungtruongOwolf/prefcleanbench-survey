"""
Step 1: Download AnthropicHH and UltraFeedback, create 20K subsets.
Usage: python scripts/01_prepare_data.py
"""
import os
import json
import random
from datasets import load_dataset

SEED = 42
SUBSET_SIZE = 20000
OUTPUT_DIR = "data"

random.seed(SEED)


def prepare_anthropic_hh():
    """Download Anthropic-HH and sample 20K examples."""
    print("Loading Anthropic-HH...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"  Full size: {len(ds)}")

    indices = random.sample(range(len(ds)), SUBSET_SIZE)
    subset = ds.select(indices)

    out_path = os.path.join(OUTPUT_DIR, "anthropic_hh")
    os.makedirs(out_path, exist_ok=True)
    subset.save_to_disk(out_path)
    print(f"  Saved 20K subset to {out_path}")
    return subset


def prepare_ultrafeedback():
    """Download UltraFeedback and sample 20K examples."""
    print("Loading UltraFeedback...")
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    print(f"  Full size: {len(ds)}")

    sample_size = min(SUBSET_SIZE, len(ds))
    indices = random.sample(range(len(ds)), sample_size)
    subset = ds.select(indices)

    out_path = os.path.join(OUTPUT_DIR, "ultrafeedback")
    os.makedirs(out_path, exist_ok=True)
    subset.save_to_disk(out_path)
    print(f"  Saved {sample_size} subset to {out_path}")
    return subset


def save_stats(hh, uf):
    """Save dataset statistics for reference."""
    stats = {
        "anthropic_hh": {"original_size": 161000, "subset_size": len(hh)},
        "ultrafeedback": {"original_size": 55000, "subset_size": len(uf)},
        "seed": SEED,
    }
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hh = prepare_anthropic_hh()
    uf = prepare_ultrafeedback()
    save_stats(hh, uf)
    print("Done.")
