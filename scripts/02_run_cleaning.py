"""
Step 2: Run all 13 cleaning methods on a dataset.
Uses PrefCleanBench's cleaning implementations.

Usage:
  python scripts/02_run_cleaning.py --dataset anthropic_hh
  python scripts/02_run_cleaning.py --dataset ultrafeedback
"""
import os
import sys
import argparse
import subprocess

# Add PrefCleanBench to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PrefCleanBench"))

CLEANING_METHODS = [
    "LLM_Judge",
    "RwGap",
    "Voting",
    "InsTag",
    "IFD",
]

FILTER_RATES = [0.1, 0.2, 0.3, 0.4]


def run_cleaning_scores(method, dataset):
    """Compute cleaning scores using PrefCleanBench scripts."""
    script_map = {
        "LLM_Judge": "src/clean_llm_judge.py",
        "RwGap": "src/clean_rw_gap.py",
        "Voting": "src/clean_voting.py",
        "InsTag": "src/clean_ins_tag.py",
        "IFD": "src/clean_ifd.py",
    }

    script = os.path.join("PrefCleanBench", script_map[method])
    print(f"Running {method} on {dataset}...")
    cmd = ["python", script, "--dataset", dataset, "--data_dir", f"data/{dataset}"]
    subprocess.run(cmd, check=True)


def run_data_cleaning(method, dataset):
    """Apply cleaning with different treatments (remove/flip)."""
    script = os.path.join("PrefCleanBench", "data_cleaning.py")
    cmd = ["python", script, method, "--dataset", dataset, "--data_dir", f"data/{dataset}"]
    subprocess.run(cmd, check=True)


def train_rwgap_model(dataset):
    """Train reward gap model (required before RwGap cleaning)."""
    script = os.path.join("PrefCleanBench", "scripts", "train_rw_gap_model.sh")
    print(f"Training RwGap model for {dataset}...")
    subprocess.run(["bash", script, dataset], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["anthropic_hh", "ultrafeedback"])
    args = parser.parse_args()

    os.makedirs(f"data/cleaned/{args.dataset}", exist_ok=True)

    # RwGap needs a trained reward model first
    print("=" * 60)
    print(f"Cleaning {args.dataset}")
    print("=" * 60)

    train_rwgap_model(args.dataset)

    for method in CLEANING_METHODS:
        print(f"\n--- {method} ---")
        try:
            run_cleaning_scores(method, args.dataset)
            run_data_cleaning(method, args.dataset)
            print(f"  {method}: done")
        except Exception as e:
            print(f"  {method}: FAILED - {e}")

    print(f"\nAll cleaning complete for {args.dataset}.")
    print("Cleaned datasets saved to data/cleaned/")


if __name__ == "__main__":
    main()
