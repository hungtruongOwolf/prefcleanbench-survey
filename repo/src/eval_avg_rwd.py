import json
import os
import sys

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from macros import DATASETS
from utils import data_to_chat

target = sys.argv[1]

for dataset in DATASETS:
    print(f"Dataset: {dataset}, Target: {target}")

    os.makedirs(f"res/{dataset}", exist_ok=True)

    URM = "LxzGordon/URM-LLaMa-3.1-8B"
    QRM = "nicolinho/QRM-Llama3.1-8B-v2"
    OffsetBias = "NCSOFT/Llama-3-OffsetBias-RM-8B"

    for rm_name, rm in [("URM", URM), ("QRM", QRM), ("OffsetBias", OffsetBias)]:
        tokenizer = AutoTokenizer.from_pretrained(rm)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        rm = AutoModelForSequenceClassification.from_pretrained(
            rm,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        rm.eval().requires_grad_(False)

        model_result = json.load(open(f'generations/{dataset}/{target}.json', 'r'))
        datasets = []
        for data in model_result:
            prompt = data['prompt']
            output = data['output']

            if not context.startswith("Human:"):
                context = 'Human: ' + prompt
            chat = data_to_chat(context)
            chat[-1]['content'] += f" {output}"

            datasets.append({
                "data": tokenizer.apply_chat_template(chat, tokenize=False)
            })

        datasets = Dataset.from_list(datasets)
        dataloader = DataLoader(datasets, batch_size=256, shuffle=False)
        rewards = []
        for inputs in tqdm(dataloader):
            resp = [tokenizer(x, return_tensors="pt") for x in inputs['data']]
            resp = {
                'input_ids': pad_sequence([x['input_ids'].T for x in resp], batch_first=True, padding_value=tokenizer.pad_token_id).squeeze(-1),
                'attention_mask': pad_sequence([x['attention_mask'].T for x in resp], batch_first=True).squeeze(-1)
            }

            with torch.no_grad():
                score = rm(resp['input_ids'].to('cuda'), attention_mask=resp['attention_mask'].to('cuda')).logits.squeeze(-1).tolist()

            rewards += score
        rewards = np.array(rewards)
        print(f"{target}: {round(rewards.mean(),3)}")
        np.savetxt(f'res/{dataset}/{target}_{rm_name}.tsv', rewards, delimiter="\t")

