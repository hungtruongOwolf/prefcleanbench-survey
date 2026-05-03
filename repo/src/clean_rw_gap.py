import os

import numpy as np

from datasets import load_dataset
from torch.utils.data import DataLoader

from macros import DATASETS
from reward_models import OwnDPO

save_path = f'outputs/rw_gap'
os.makedirs(save_path, exist_ok=True)

for dataset in DATASETS:
    dataset = load_dataset("json", data_files=f"datasets/{dataset}/no_clean/train.jsonl")['train']

    for idx in range(8):
        model = OwnDPO(dataset, idx)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        scores = model.predict(dataloader)
        np.savetxt(f'{save_path}/{dataset}_{idx}.tsv', scores, delimiter="\t")