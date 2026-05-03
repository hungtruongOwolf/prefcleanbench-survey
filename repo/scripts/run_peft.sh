#!/bin/bash

set -e

deepspeed_args="--master_port=17903 --include=localhost:0,1,2,3"

opt_alg=${1:-DPO}                     
dataset=${2:-AnthropicHH}
version=${3:-no_clean}
pretty_name=${4:-DPO_llama3_8b_no_clean}
model_path="${5:-meta-llama/Meta-Llama-3-8B}"

data_path=datasets/${dataset}/${version}
model_name_or_path=models/${dataset}/sft_${pretty_name}
output_dir=models/${dataset}/${pretty_name}
bsz=16
training_sample_number=$(wc -l < "${data_path}/train.jsonl")
num_train_epochs=1.0
eval_steps=$(echo "scale=0; ($training_sample_number - 100) * $num_train_epochs / $bsz / 4" | bc)
loss_type="sigmoid"
beta=0.1

case "$opt_alg" in
  ORPO)
    model_name_or_path=$model_path
    ;;
  SLiC)
    loss_type="hinge"
    ;;
  AOT)
    loss_type="aot"
    ;;
  IPO)
    loss_type="ipo"
    beta=0.5
    ;;
  rDPO)
    loss_type="robust"
    ;;
esac

# Build command
cmd="deepspeed ${deepspeed_args} src/lm_flow/run_peft.py \
  --opt_alg ${opt_alg} \
  --model_name_or_path ${model_name_or_path} \
  --dataset_path ${data_path} \
  --output_dir ${output_dir} --overwrite_output_dir \
  --num_train_epochs ${num_train_epochs} \
  --learning_rate 5e-5 \
  --block_size 512 \
  --per_device_train_batch_size ${bsz} \
  --per_device_eval_batch_size ${bsz} \
  --deepspeed configs/ds_config_zero2.json \
  --bf16 \
  --validation_split_percentage 0 \
  --logging_steps 10 \
  --do_train \
  --use_lora 1 \
  --lora_r 16 \
  --lr_scheduler_type constant \
  --save_steps 999999 \
  --eval_strategy steps \
  --eval_steps ${eval_steps} \
  --weight_decay 0.0001 \
  --dataloader_num_workers 8 \
  --seed 14 \
  --beta ${beta} \
  --loss_type ${loss_type}"

# Print and run
echo "Running command:"
echo $cmd
eval $cmd
