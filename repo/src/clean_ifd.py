import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from llama3_tokenizer import CustomLlama3Tokenizer
from macros import DATASETS

device = "cuda"

tokenizer = CustomLlama3Tokenizer("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map='auto',
    trust_remote_code=True)
model.eval().requires_grad_(False)

save_path = f'../outputs/ifd'
os.makedirs(save_path, exist_ok=True)

def get_log_prob(data):
    output = model(data.input_ids.to(device), attention_mask=data.attention_mask.to(device)).logits.log_softmax(-1)[0]
    log_prob = torch.tensor([output[i, idx] for i, idx in enumerate(data.input_ids[0])])
    return log_prob

for dataset in DATASETS:
    train_dataset = load_dataset("json", data_files=f"../datasets/{dataset}/no_clean/train.jsonl")['train']

    res = []
    for data in tqdm(train_dataset):
        prompt = tokenizer(data['prompt'], max_length=512, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        chosen = tokenizer(data['chosen'], max_length=512, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        reject = tokenizer(data['rejected'], max_length=512, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        prompt_text = tokenizer.batch_decode(prompt.input_ids, skip_special_tokens=True)[0]
        chosen_text = tokenizer.batch_decode(chosen.input_ids, skip_special_tokens=True)[0]
        reject_text = tokenizer.batch_decode(reject.input_ids, skip_special_tokens=True)[0]
        
        prompt_chosen = tokenizer(prompt_text + chosen_text, max_length=1024, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        prompt_reject = tokenizer(prompt_text + reject_text, max_length=1024, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        if len(chosen_text) <= 1:
            res.append((10.0, -10.0))
            continue

        response_start_idx = prompt_chosen.char_to_token(len(prompt_text) + 1) - 1

        with torch.no_grad():
            prompt_chosen_log_prob = get_log_prob(prompt_chosen)
            prompt_reject_log_prob = get_log_prob(prompt_reject)
            chosen_log_prob = get_log_prob(chosen)
            reject_log_prob = get_log_prob(reject)
            
            idf_chosen = torch.exp(-prompt_chosen_log_prob[response_start_idx:].mean()) / torch.exp(-chosen_log_prob.mean())
            idf_reject = torch.exp(-prompt_reject_log_prob[response_start_idx:].mean()) / torch.exp(-reject_log_prob.mean())

        res.append((idf_chosen.item(), (idf_chosen - idf_reject).item()))

    res = np.array(res)

    np.savetxt(f'{save_path}/{dataset}.tsv', res, delimiter="\t")