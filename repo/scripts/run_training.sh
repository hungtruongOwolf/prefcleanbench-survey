#!/bin/bash

# Usage:
# bash script/run_training.sh <Cleansed_version|no_clean> <Base_model> <Optimization_algorithm> [Pretty_name]

# Parse arguments
CLEANSING=$1
BASE_MODEL=$2
OPT_ALGO=$3
PRETTY_NAME=$4

# Validate input
if [ -z "$CLEANSING" ] || [ -z "$BASE_MODEL" ] || [ -z "$OPT_ALGO" ]; then
  echo "Usage: bash script/run_training.sh <Cleansed_version|no_clean> <Base_model> <Optimization_algorithm> [Pretty_name]"
  exit 1
fi

# Default pretty name
if [ -z "$PRETTY_NAME" ]; then
    PRETTY_NAME="${OPT_ALGO}_${BASE_MODEL}_${CLEANSING}"
fi

# Map base_model to actual model path
case "$BASE_MODEL" in
  llama3_8b)
    MODEL_PATH="meta-llama/Meta-Llama-3-8B"
    ;;
  wqen2.5_7b)
    MODEL_PATH="Qwen/Qwen2.5-7B"
    ;;
  phi2)
    MODEL_PATH="microsoft/phi-2"
    ;;
  mistral_7b)
    MODEL_PATH="mistralai/Mistral-7B-v0.3"
    ;;
  *)
    echo "Error: Unsupported base model '$BASE_MODEL'"
    exit 1
    ;;
esac

echo "========================================"
echo "Starting training with:"
echo " - Cleansing version: $CLEANSING"
echo " - Base model: $BASE_MODEL"
echo " - Model path: $MODEL_PATH"
echo " - Optimization algorithm: $OPT_ALGO"
echo " - Pretty name: $PRETTY_NAME"
echo "========================================"

# Dataset list
DATASETS=("AnthropicHH" "UltraFeedback" "SafeRLHF" "HelpSteer2")

for DATASET in "${DATASETS[@]}"; do
  echo "----------------------------------------"
  echo "Processing dataset: $DATASET"

  if [ "$OPT_ALGO" != "ORPO" ]; then
    echo "Running SFT on $DATASET"
    bash scripts/run_sft.sh \
      -m "$MODEL_PATH" \
      -d "$DATASET" \
      --version "$CLEANSING" \
      --pretty_name "$PRETTY_NAME"
  else
    echo "Skipping SFT for $DATASET (ORPO)"
  fi

  echo "Running PEFT on $DATASET"
  bash scripts/run_peft.sh \
    "$OPT_ALGO" \
    "$DATASET" \
    "$CLEANSING" \
    "$PRETTY_NAME" \
    "$MODEL_PATH"
done