import torch
from datasets import load_dataset, concatenate_datasets
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, DPOTrainer, CPOConfig, CPOTrainer, KTOConfig, KTOTrainer, ORPOConfig, ORPOTrainer

# Define mapping
METHOD_TRAINERS = {
    'DPO': (DPOTrainer, DPOConfig),
    'CPO': (CPOTrainer, CPOConfig),
    'KTO': (KTOTrainer, KTOConfig),
    'ORPO': (ORPOTrainer, ORPOConfig),
}

# Parse arguments
pipeline_name = "finetuner"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
parser.add_argument('--opt_alg', type=str, choices=['DPO', 'CPO', 'KTO', 'ORPO'], required=True)
parser.add_argument('--loss_type', default='sigmoid')
parser.add_argument('--beta', type=float, default=0.1)
args = parser.parse_args_into_dataclasses()
model_args, data_args, pipeline_args = args[:3]
method_args = parser.parse_args()

# Setup tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, load_in_8bit=True)
tokenizer.pad_token = tokenizer.eos_token

# Setup model
print("Loading model...")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, )
print(model_args.model_name_or_path)
print(model_args.use_lora)
if model_args.use_lora:
    model_lora = get_peft_model(model, peft_config)
    model_lora.print_trainable_parameters()
else:
    model_lora = model

model_lora.config.pad_token_id = tokenizer.eos_token_id

# Build dataset
def extract_positive(sample):
    return {"prompt": sample['prompt'], "completion": sample["chosen"], "label": True}

def extract_negative(sample):
    return {"prompt": sample['prompt'], "completion": sample["rejected"], "label": False}

def build_dataset(config, opt_alg):
    ds = load_dataset("json", data_files=f"{config.dataset_path}/train.jsonl")['train']
    test_path = '/'.join(config.dataset_path.split('/')[:-1]) 
    test_ds = load_dataset("json", data_files=f"{test_path}/no_clean/test.jsonl")['train']
    if opt_alg == 'kto':
        train_dataset = concatenate_datasets([ds.map(extract_positive), ds.map(extract_negative)])
        eval_dataset = concatenate_datasets([test_ds.map(extract_positive), test_ds.map(extract_negative)])
    else:
        train_dataset, eval_dataset = ds, test_ds
    return train_dataset, eval_dataset

train_dataset, eval_dataset = build_dataset(data_args, method_args.opt_alg)
print(f"Training samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# Setup config and trainer
trainer_class, config_class = METHOD_TRAINERS[method_args.opt_alg]
trainer_args = config_class(
    output_dir=pipeline_args.output_dir,
    beta=method_args.beta,
    max_length=1024,
    max_prompt_length=512,
    learning_rate=pipeline_args.learning_rate,
    weight_decay=pipeline_args.weight_decay,
    lr_scheduler_type=pipeline_args.lr_scheduler_type,
    per_device_train_batch_size=pipeline_args.per_device_train_batch_size,
    per_device_eval_batch_size=pipeline_args.per_device_eval_batch_size,
    num_train_epochs=pipeline_args.num_train_epochs,
    bf16=pipeline_args.bf16,
    seed=pipeline_args.seed,
    do_train=pipeline_args.do_train,
    do_eval=pipeline_args.do_eval,
    run_name=pipeline_args.run_name,
    eval_strategy=pipeline_args.eval_strategy,
    eval_steps=pipeline_args.eval_steps,
    save_steps=pipeline_args.save_steps,
    deepspeed=pipeline_args.deepspeed,
)

if method_args.opt_alg == 'DPO':
    trainer_args.loss_type = method_args.loss_type

trainer = trainer_class(
    model=model_lora,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()

if model_args.use_lora:
    model_lora = model_lora.merge_and_unload()

model_lora.save_pretrained(pipeline_args.output_dir, safe_serialization=False)
model.config.save_pretrained(pipeline_args.output_dir)
tokenizer.save_pretrained(pipeline_args.output_dir)
