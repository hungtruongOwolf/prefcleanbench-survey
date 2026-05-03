#!/bin/bash
# Step 3: Train models with preference optimization.
#
# Usage:
#   bash scripts/03_train.sh <algorithm> <scope>
#   algorithm: dpo | cpo | kto | orpo
#   scope:     all (13 methods) | top3 (VoteMaj-R, LLM-Judge-R, Tag-Cmp) | no_clean
#
# Examples:
#   bash scripts/03_train.sh dpo all       # Main experiment
#   bash scripts/03_train.sh kto top3      # Cross-algo experiment
#   bash scripts/03_train.sh dpo no_clean  # Baseline only

set -e

ALGO=${1:-dpo}
SCOPE=${2:-all}
BASE_MODEL="meta-llama/Meta-Llama-3-8B"
DATASETS=("anthropic_hh" "ultrafeedback")

# Cleaning method lists
ALL_METHODS=(
    "no_clean"
    "llm_judge_r" "llm_judge_f"
    "rw_gap_r_0.2" "rw_gap_f_0.2"
    "vote_all_r" "vote_all_f"
    "vote_maj_r" "vote_maj_f"
    "ins_tag_cmp" "ins_tag_div"
    "ifd_r_0.2" "ifd_gap_r_0.2" "ifd_gap_f_0.2"
)

TOP3_METHODS=("no_clean" "vote_maj_r" "llm_judge_r" "ins_tag_cmp")

if [ "$SCOPE" == "all" ]; then
    METHODS=("${ALL_METHODS[@]}")
elif [ "$SCOPE" == "top3" ]; then
    METHODS=("${TOP3_METHODS[@]}")
elif [ "$SCOPE" == "no_clean" ]; then
    METHODS=("no_clean")
else
    echo "Unknown scope: $SCOPE (use: all, top3, no_clean)"
    exit 1
fi

echo "========================================"
echo "Algorithm: $ALGO"
echo "Scope: $SCOPE (${#METHODS[@]} methods)"
echo "Datasets: ${DATASETS[*]}"
echo "========================================"

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        RUN_NAME="${ALGO}_llama3_8b_${METHOD}_${DATASET}"
        OUTPUT_DIR="outputs/models/${RUN_NAME}"

        if [ -d "$OUTPUT_DIR" ]; then
            echo "Skipping $RUN_NAME (already exists)"
            continue
        fi

        echo ""
        echo "--- Training: $RUN_NAME ---"

        # Use PrefCleanBench's training script
        bash PrefCleanBench/scripts/run_training.sh \
            "$METHOD" \
            "llama3_8b" \
            "$(echo $ALGO | tr '[:lower:]' '[:upper:]')" \
            "$RUN_NAME"

        echo "  Saved to $OUTPUT_DIR"
    done
done

echo ""
echo "All training runs complete."
