#!/bin/bash

DATASETS=("AnthropicHH" "UltraFeedback" "SafeRLHF" "HelpSteer2")

for DATASET in "${DATASETS[@]}"; do
    for i in {0..7}; do
        bash scripts/run_sft.sh \
            -m "meta-llama/Meta-Llama-3-8B" \
            -d "$DATASET" \
            --version "no_clean" \
            --pretty_name "RwGap_$i"

        bash scripts/run_peft.sh \
            "DPO" \
            "$DATASET" \
            "no_clean" \
            "RwGap_$i" \
            "meta-llama/Meta-Llama-3-8B"
    done
done    