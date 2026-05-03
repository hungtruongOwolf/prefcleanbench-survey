#!/bin/bash
# Step 4: Generate responses for all trained models.
# Uses PrefCleanBench's gen_response.py
#
# Usage: bash scripts/04_generate_responses.sh

set -e

OUTPUT_DIR="outputs/generations"
mkdir -p "$OUTPUT_DIR"

echo "Generating responses for all trained models..."
echo ""

for MODEL_DIR in outputs/models/*/; do
    RUN_NAME=$(basename "$MODEL_DIR")

    if [ -f "$OUTPUT_DIR/${RUN_NAME}.jsonl" ]; then
        echo "Skipping $RUN_NAME (already generated)"
        continue
    fi

    echo "Generating: $RUN_NAME"
    python PrefCleanBench/src/gen_response.py "$RUN_NAME"
done

echo ""
echo "All responses generated."
