
import json, os, random
import numpy as np

DATASETS = ["AnthropicHH", "UltraFeedback"]
MAX_SIZE = 6000

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

    tags = json.load(open(f"outputs/ins_tag/{dataset_name}.json"))
    tags = [set(t) for t in tags]
    tag_len = [len(t) for t in tags]

    # Sort by complexity (number of tags) descending
    idx_sorted = np.argsort(tag_len)[::-1].tolist()

    # Tag-Cmp: select by complexity — prefer prompts with more tags
    res, selected, tag_set = [], [], set()
    for i in idx_sorted:
        if len(selected) >= MAX_SIZE:
            break
        if len(tag_set | tags[i]) > len(tag_set):
            tag_set |= tags[i]
            selected.append(i)
            res.append({"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                        "rejected": raw[i]["rejected"]})
    # Fill up to MAX_SIZE if needed
    if len(res) < MAX_SIZE:
        remain = [i for i in idx_sorted if i not in set(selected)]
        for i in random.choices(remain, k=MAX_SIZE - len(res)):
            res.append({"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                        "rejected": raw[i]["rejected"]})
    save_data(res, f"{dataset_name}/ins_tag_cmp")

    # Tag-Div: select by diversity — maximize unique tags covered
    total_tags = set()
    for t in tags:
        total_tags |= t

    res, selected, div_set = [], [], set()
    for i in idx_sorted:
        if len(selected) >= MAX_SIZE:
            break
        if len(div_set | tags[i]) > len(div_set):
            div_set |= tags[i]
            selected.append(i)
            res.append({"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                        "rejected": raw[i]["rejected"]})
        if len(div_set) == len(total_tags):
            break
    if len(res) < MAX_SIZE:
        remain = [i for i in idx_sorted if i not in set(selected)]
        for i in random.choices(remain, k=MAX_SIZE - len(res)):
            res.append({"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                        "rejected": raw[i]["rejected"]})
    save_data(res, f"{dataset_name}/ins_tag_div")

print("\n=== InsTag cleaning done! ===")
