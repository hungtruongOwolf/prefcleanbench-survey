
import json, os
import numpy as np

DATASETS = ["AnthropicHH", "UltraFeedback"]
VOTING_DIR = "outputs/voting"
RM_NAMES = ["GRM", "DeBERTa"]

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
    print(f"  Saved {len(res)} examples → {save_path}/")

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")

    # Load dataset
    raw = []
    with open(f"datasets/{dataset_name}/no_clean/train.jsonl") as f:
        for line in f:
            raw.append(json.loads(line))
    raw = raw[:20000]

    # Load scores from both RMs
    pref_strength = []
    for rm_name in RM_NAMES:
        scores = np.loadtxt(f"{VOTING_DIR}/{dataset_name}_{rm_name}.tsv", delimiter="\t")
        diff = scores[:, 0] - scores[:, 1]  # chosen_score - rejected_score
        pref_strength.append(diff)

    pref_strength = np.stack(pref_strength).T  # shape: (N, num_RMs)
    vote = (pref_strength > 0).astype(int)     # 1 if RM agrees chosen > rejected
    agreeing = vote.sum(axis=1)                # how many RMs agree
    n_rms = len(RM_NAMES)

    print(f"  Total: {len(raw)} | All agree: {(agreeing == n_rms).sum()} | Majority agree: {(agreeing >= n_rms/2).sum()}")

    # vote_all_r — remove if ALL RMs disagree with label
    res = []
    for i, d in enumerate(raw):
        if agreeing[i] > 0:  # at least 1 RM agrees
            res.append({"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]})
    save_data(res, f"{dataset_name}/vote_all_r")

    # vote_all_f — flip if ALL RMs disagree
    res = []
    for i, d in enumerate(raw):
        if agreeing[i] == 0:
            res.append({"prompt": d["prompt"], "chosen": d["rejected"], "rejected": d["chosen"]})
        else:
            res.append({"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]})
    save_data(res, f"{dataset_name}/vote_all_f")

    # vote_maj_r — remove if MAJORITY of RMs disagree
    res = []
    for i, d in enumerate(raw):
        if agreeing[i] >= n_rms / 2:
            res.append({"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]})
    save_data(res, f"{dataset_name}/vote_maj_r")

    # vote_maj_f — flip if MAJORITY disagree
    res = []
    for i, d in enumerate(raw):
        if agreeing[i] < n_rms / 2:
            res.append({"prompt": d["prompt"], "chosen": d["rejected"], "rejected": d["chosen"]})
        else:
            res.append({"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]})
    save_data(res, f"{dataset_name}/vote_maj_f")

print("\n=== Voting cleaning done! ===")
