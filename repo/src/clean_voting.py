import os

import numpy as np

from datasets import load_dataset
from torch.utils.data import DataLoader

from macros import DATASETS, RMs_top_six

save_path = f'outputs/voting'
os.makedirs(save_path, exist_ok=True)

for dataset in DATASETS:
    dataset = load_dataset("json", data_files=f"datasets/{dataset}/no_clean/train.jsonl")['train']

    for model in RMs_top_six:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        model = model()
        scores = model.predict(dataloader)
        np.savetxt(f'{save_path}/{dataset}_{model.name}.tsv', scores, delimiter="\t")
        
