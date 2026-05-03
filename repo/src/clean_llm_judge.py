import os

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

from macros import DATASETS
from reward_models import GPT4

save_path = f'outputs/llm_judge'
os.makedirs(save_path, exist_ok=True)

rm = GPT4()

for dataset in DATASETS:
    train_dataset = load_dataset("json", data_files=f"datasets/{dataset}/no_clean/train.jsonl")['train']

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    scores = rm.predict(dataloader)
    np.savetxt(f'{save_path}/{dataset}.tsv', scores, delimiter="\t")