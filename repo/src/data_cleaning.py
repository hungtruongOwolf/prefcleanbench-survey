import json
import os
import random
import sys

import numpy as np
from datasets import load_dataset

from macros import RMs_top_six, DATASETS, CLEANING_APPROACHES


def save_data(res, path):
    save_path = f"datasets/{path}"
    os.makedirs(save_path, exist_ok=True)
    v_train = open(f'{save_path}/train.jsonl', 'w')
    for r in res:
        json.dump(r, v_train)
        v_train.write('\n')
    v_train.close()

    os.makedirs(f"{save_path}/sft", exist_ok=True)
    json.dump({"type": "text_only", "instances": [{"text": f"\n\nHuman: {r['prompt']}\n\nAssistant: {r['chosen']}"} for r in res]}, open(f'{save_path}/sft/train.json', 'w'))


def llm_judge(dataset_name, dataset):
    predict = np.loadtxt(f'outputs/llm_judge/{dataset_name}.tsv', delimiter="\t")
    predict = predict[:, 0] - predict[:, 1]

    r, f = [], []
    for i, data in enumerate(dataset):
        if predict[i] < 0:
            f.append({
                'prompt': data['prompt'],
                'chosen': data['rejected'],
                'rejected': data['chosen']
            })
        else:
            r.append({
                'prompt': data['prompt'],
                'chosen': data['chosen'],
                'rejected': data['rejected']
            })
    save_data(r, f"{dataset_name}/llm_judge_r")
    save_data(r + f, f"{dataset_name}/llm_judge_f")

def ifd(dataset_name, dataset):
    predict = np.loadtxt(f'outputs/ifd/{dataset_name}.tsv', delimiter="\t")
    ifd_score = predict[:, 0]
    ifd_gap = predict[:, 1]

    ifd_idx = np.argsort([x for x in ifd_score if x <= 1])
    ifd_gap_idx = np.argsort(ifd_gap)

    proportions = [0.1, 0.2, 0.3, 0.4]

    for p in proportions:
        res = []
        ifd_correct = ifd_idx[int(len(ifd_idx) * p):].tolist()
        for i in ifd_correct:
            res.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['chosen'],
                'rejected': dataset[i]['rejected']
            })
        save_data(res, f"{dataset_name}/ifd_r")


        ifd_gap_wrong = ifd_gap_idx[:int(len(ifd_gap_idx) * p)].tolist()
        ifd_gap_correct = ifd_gap_idx[int(len(ifd_gap_idx) * p):].tolist()

        r, f = [], []
        for i in ifd_gap_wrong:
            f.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['rejected'],
                'rejected': dataset[i]['chosen']
            })
        for i in ifd_gap_correct:
            r.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['chosen'],
                'rejected': dataset[i]['rejected']
            })
        save_data(r, f"{dataset_name}/ifd_gap_r_{p}")
        save_data(r + f, f"{dataset_name}/ifd_gap_f_{p}")

def rw_gap(dataset_name, dataset):
    predict = np.stack([np.loadtxt(f'outputs/rw_gap/{dataset_name}_{idx}.tsv', delimiter="\t") for idx in range(8)], axis=-0)
    predict = (predict[:, :, 0] - predict[:, :, 1]).T.mean(axis=-1).reshape(-1)

    idx = np.argsort(predict)

    proportions = [0.1, 0.2, 0.3, 0.4]
    for p in proportions:
        wrong = idx[:int(len(idx) * p)].tolist()
        correct = idx[int(len(idx) * p):].tolist()

        r, f = [], []
        for i in wrong:
            f.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['rejected'],
                'rejected': dataset[i]['chosen']
            })
        for i in correct:
            r.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['chosen'],
                'rejected': dataset[i]['rejected']
            })
        save_data(r, f"{dataset_name}/rw_gap_r_{p}")
        save_data(r + f, f"{dataset_name}/rw_gap_f_{p}")

def ins_tag(dataset_name, dataset):
    max_size = 6000

    tags = json.load(open(f'outputs/ins_tag/{dataset_name}.json'))
    tags = [set(x) for x in tags]
    tag_len = [len(x) for x in tags]

    idx = np.argsort(tag_len)[::-1].tolist()

    res = []
    complexity_tag_set = set()
    selected_id = []
    for i in idx:
        if len(selected_id) > max_size:
            break
        if len(complexity_tag_set | tags[i]) > len(complexity_tag_set):
            complexity_tag_set |= tags[i]
            selected_id.append(i)
            res.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['chosen'],
                'rejected': dataset[i]['rejected']
            })
    if len(selected_id) < max_size:
        remain = [i for i in idx if i not in selected_id]
        remain = random.choices(remain, k=max_size - len(selected_id))
        for i in remain:
            res.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['chosen'],
                'rejected': dataset[i]['rejected']
            })
    save_data(res, f"{dataset_name}/ins_tag_cmp")

    res = []
    diversity_tag_set = set()
    selected_id = []
    total_tags_set = set()
    for t in tags:
        total_tags_set |= t
    for i in idx:
        if len(selected_id) > max_size:
            break
        if len(diversity_tag_set | tags[i]) > len(diversity_tag_set):
            diversity_tag_set |= tags[i]
            selected_id.append(i)
            res.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['chosen'],
                'rejected': dataset[i]['rejected']
            })

            if len(diversity_tag_set) == len(total_tags_set):
                break
    if len(selected_id) < max_size:
        remain = [i for i in idx if i not in selected_id]
        remain = random.choices(remain, k=max_size - len(selected_id))
        for i in remain:
            res.append({
                'prompt': dataset[i]['prompt'],
                'chosen': dataset[i]['chosen'],
                'rejected': dataset[i]['rejected']
            })

    save_data(res, f"{dataset_name}/ins_tag_div")

def voting(dataset_name, dataset):
    pref_strength = []
    for RM in RMs_top_six:
        df = np.loadtxt(f'outputs/voting/{dataset_name}_{RM().name}.tsv', delimiter="\t")
        data = df.to_numpy()
        data = data[:, 0] - data[:, 1]
        pref_strength.append(data)
    pref_strength = np.stack(pref_strength).T
    vote = np.where(pref_strength > 0, 1, 0)
    agreeing = vote.sum(axis=1)

    variants = ['all_r', 'all_f', 'maj_r', 'maj_f']
    for v in variants:
        res = []
        for i, d in enumerate(dataset):
            if (v == 'all_r' and agreeing[i] == 0) or (v == 'maj_r' and agreeing[i] < len(RMs_top_six) / 2):
                continue
            c, r = d['chosen'], d['rejected']
            if (v == 'all_f' and agreeing[i] == 0) or (v == 'maj_f' and agreeing[i] < len(RMs_top_six) / 2):
                c, r = r, c
            res.append({
                'prompt': d['prompt'],
                'chosen': c,
                'rejected': r
            })

        save_data(res, f"{dataset_name}/vote_{v}")

cleaning_approaches = [llm_judge, rw_gap, voting, ins_tag, ifd]

try:
    approach_index = CLEANING_APPROACHES.index(sys.argv[1])
    for dataset in DATASETS:
        train_set = load_dataset("json", data_files=f"datasets/{dataset}/no_clean/train.jsonl")['train']
        cleaning_approaches[approach_index](dataset, train_set)
except:
    print(f"Error: {sys.argv[1]} is not a data cleaning approach")