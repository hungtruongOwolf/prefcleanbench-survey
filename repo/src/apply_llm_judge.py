
import json, os
import numpy as np

DATASETS = ["AnthropicHH", "UltraFeedback"]

def save_data(res, path):
    save_path = f"datasets/{path}"
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/train.jsonl", "w") as f:
        for r in res:
            json.dump(r, f)
            f.write("\n")
    os.makedirs(f"{save_path}/sft", exist_ok=True)
    sft = {"type": "text_only", "instances": [
        {"text": f"\n\nHuman: {r['prompt']}\n\nAssistant: {r['chosen']}"}
        for r in res
    ]}
    json.dump(sft, open(f"{save_path}/sft/train.json", "w"))
    print(f"  Saved {len(res)} → {save_path}/")

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")

    raw = []
    with open(f"datasets/{dataset_name}/no_clean/train.jsonl") as f:
        for line in f:
            raw.append(json.loads(line))
    raw = raw[:20000]

    scores = np.loadtxt(f"outputs/llm_judge/{dataset_name}.tsv", delimiter="\t")
    diff = scores[:, 0] - scores[:, 1]  # chosen_score - rejected_score

    r, f_list = [], []
    for i, d in enumerate(raw):
        entry_r = {"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]}
        entry_f = {"prompt": d["prompt"], "chosen": d["rejected"], "rejected": d["chosen"]}
        if diff[i] >= 0:
            r.append(entry_r)
            f_list.append(entry_r)
        else:
            f_list.append(entry_f)

    save_data(r, f"{dataset_name}/llm_judge_r")
    save_data(f_list, f"{dataset_name}/llm_judge_f")

    flagged = (diff < 0).sum()
    print(f"  Flagged as noisy: {flagged}/{len(raw)} ({flagged/len(raw)*100:.1f}%)")

print("\n=== LLM-Judge cleaning done! ===")
