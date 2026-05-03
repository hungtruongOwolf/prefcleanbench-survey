# PrefCleanBench Survey — Reproduction Code

Code for the paper: **Survey: Preference Data Cleaning Methods for Reliable LLM Alignment**

We reproduce and extend experiments from [PrefCleanBench](https://github.com/deeplearning-wisc/PrefCleanBench) (Yeh & Li, NeurIPS 2025) on 20K subsets of AnthropicHH and UltraFeedback using a single A100 40GB GPU.

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/hungtruong97/prefcleanbench-survey.git
cd prefcleanbench-survey

# 2. Clone the original PrefCleanBench codebase
git clone https://github.com/deeplearning-wisc/PrefCleanBench.git

# 3. Install dependencies
pip install -r requirements.txt
```

## Reproduction Steps

Run scripts in order. Each script is self-contained and can be run independently after the previous step completes.

### Step 1: Prepare 20K subsets

```bash
python scripts/01_prepare_data.py
```

Downloads AnthropicHH and UltraFeedback from HuggingFace, samples 20K examples from each, and saves them to `data/`.

### Step 2: Run all 13 cleaning methods

```bash
python scripts/02_run_cleaning.py --dataset anthropic_hh
python scripts/02_run_cleaning.py --dataset ultrafeedback
```

Runs each cleaning method using PrefCleanBench's implementations. Outputs cleaned datasets to `data/cleaned/`.

### Step 3: Train models (DPO + cross-algorithm)

```bash
# Main experiment: all 13 methods with DPO
bash scripts/03_train.sh dpo all

# Cross-algorithm: top 3 methods with CPO, KTO, ORPO
bash scripts/03_train.sh cpo top3
bash scripts/03_train.sh kto top3
bash scripts/03_train.sh orpo top3
```

Trains Llama-3-8B with LoRA (rank 16, alpha 32) on each cleaned dataset. Checkpoints saved to `outputs/models/`.

### Step 4: Generate responses

```bash
bash scripts/04_generate_responses.sh
```

Generates responses on test prompts for all trained models. Saved to `outputs/generations/`.

### Step 5: Evaluate

```bash
# Win-Tie rate (requires OpenAI API key)
export OPENAI_API_KEY="your-key-here"
python scripts/05_eval_wintie.py

# Average gold reward
python scripts/06_eval_reward.py
```

### Step 6: Generate figures

```bash
python scripts/07_plot_figures.py
```

Produces all figures used in the report, saved to `figures/`.

## Project Structure

```
prefcleanbench-survey/
├── README.md
├── requirements.txt
├── configs/
│   └── train_config.yaml       # LoRA and training hyperparameters
├── scripts/
│   ├── 01_prepare_data.py      # Download and subsample datasets
│   ├── 02_run_cleaning.py      # Run all 13 cleaning methods
│   ├── 03_train.sh             # LoRA training with DPO/CPO/KTO/ORPO
│   ├── 04_generate_responses.sh
│   ├── 05_eval_wintie.py       # GPT-4o win-tie evaluation
│   ├── 06_eval_reward.py       # Gold reward model scoring
│   └── 07_plot_figures.py      # Generate all report figures
├── figures/                    # Output figures
└── PrefCleanBench/             # Cloned original repo (not included)
```

## Hardware

- GPU: 1x NVIDIA A100 40GB
- Training: ~3-5 hours per run with LoRA
- Total GPU time: ~200 hours
- API cost: ~$105 (GPT-4o evaluation + LLM-Judge cleaning)

## Citation

```bibtex
@inproceedings{yeh2025clean,
  title={Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable {LLM} Alignment},
  author={Yeh, Min-Hsuan and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)},
  year={2025}
}
```
