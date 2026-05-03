import json
import os

from datasets import load_dataset, concatenate_datasets

from macros import DATASETS

def proc_HelpSteer2(sample):
    prompt = sample['prompt'].strip()
    rs1 = sample['response_1'].strip()
    rs2 = sample['response_2'].strip()
    if sample['preference_strength'] > 0:
        rs1, rs2 = rs2, rs1

    return {
        'prompt': prompt,
        'chosen': rs1,
        'rejected': rs2,
    }

def proc_SafeRLHF(sample):
    prompt = sample['prompt'].strip()
    rs1 = sample['response_0'].strip()
    rs2 = sample['response_1'].strip()
    if sample['better_response_id'] == 1:
        rs1, rs2 = rs2, rs1

    return {
        'prompt': prompt,
        'chosen': rs1,
        'rejected': rs2,
    }

def proc_UltraFeedback(sample):
    rs1 = sample['chosen']
    rs2 = sample['rejected']

    return {
        'prompt': rs1[0]['content'].strip(),
        'chosen': rs1[1]['content'].strip(),
        'rejected': rs2[1]['content'].strip(),
    }

def proc_AnthropicHH(sample):
    prompt, chosen = sample['chosen'].rsplit("\n\nAssistant: ", 1)
    _, rejected = sample['rejected'].rsplit("\n\nAssistant: ", 1)

    return {
        'prompt': prompt.strip(),
        'chosen': chosen.strip(),
        'rejected': rejected.strip(),
    }

proc_fns = [proc_AnthropicHH, proc_UltraFeedback, proc_SafeRLHF, proc_HelpSteer2]

for dataset_id, _ in enumerate(DATASETS):
    if dataset_id == 0:
        anthropic_hh_dirs = ["harmless-base", "helpful-base", "helpful-rejection-sampled", "helpful-online"]

        train_set = concatenate_datasets([load_dataset("Anthropic/hh-rlhf", data_dir=data_dir)['train'] for data_dir in anthropic_hh_dirs])
        test_set = concatenate_datasets([load_dataset("Anthropic/hh-rlhf", data_dir=data_dir)['test'] for data_dir in anthropic_hh_dirs])
    elif dataset_id == 1:
        train_set = load_dataset("pharaouk/ultrafeedback-binarized-preferences-cleaned", split="train[:90%]")
        test_set = load_dataset("pharaouk/ultrafeedback-binarized-preferences-cleaned", split="train[90%:]")
    elif dataset_id == 2:
        datasets = load_dataset("PKU-Alignment/PKU-SafeRLHF-single-dimension")
        train_set = datasets['train']
        test_set = datasets['test']
    elif dataset_id == 3:
        dataset = load_dataset("nvidia/HelpSteer2", data_dir="preference")['train']
        dataset = dataset.remove_columns(["preference_statement", "preference_elaboration", "three_most_similar_preferences", "all_preferences_unprocessed"])
        train_set = dataset.filter(lambda example: example["split"] == 'train')
        test_set = dataset.filter(lambda example: example["split"].startswith("val"))

    train_set = train_set.map(proc_fns[dataset_id])
    test_set = test_set.map(proc_fns[dataset_id])

    save_path = f'datasets/{DATASETS[dataset_id]}/no_clean'
    os.makedirs(save_path, exist_ok=True)

    train_file = open(f'{save_path}/train.jsonl', 'w')
    test_file = open(f'{save_path}/test.jsonl', 'w')
    for split in zip([train_set, test_set], [train_file, test_file]):
        for d in split[0]:
            json.dump({
                'prompt': d['prompt'],
                'chosen': d['chosen'],
                'rejected': d['rejected']
            }, split[1])
            split[1].write('\n')

    train_file.close()
    test_file.close()

    os.makedirs(f"{save_path}/sft", exist_ok=True)
    res = []
    for data in train_set:
        res.append({"text": f"\n\nHuman: {d['prompt']}\n\nAssistant: {data['chosen']}"})
    json.dump({"type": "text_only", "instances": res}, open(f'{save_path}/sft/train.json', 'w'))