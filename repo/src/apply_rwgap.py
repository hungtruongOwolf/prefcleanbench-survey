
import json, os
import numpy as np

DATASETS = ["AnthropicHH", "UltraFeedback"]
RW_GAP_DIR = "outputs/rw_gap"
PROPORTIONS = [0.1, 0.2, 0.3, 0.4]
NUM_MODELS = 4

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

    # Load scores from all 4 models and average
    all_scores = []
    for idx in range(NUM_MODELS):
        scores = np.loadtxt(f"{RW_GAP_DIR}/{dataset_name}_{idx}.tsv", delimiter="\t")
        diff = scores[:, 0] - scores[:, 1]  # chosen - rejected
        all_scores.append(diff)

    # Average gap across 4 models
    avg_gap = np.stack(all_scores).mean(axis=0)
    idx_sorted = np.argsort(avg_gap)  # ascending — smallest gap first (most noisy)

    for p in PROPORTIONS:
        n = len(raw)
        cut = int(n * p)

        wrong = idx_sorted[:cut].tolist()
        correct = idx_sorted[cut:].tolist()

        # rw_gap_r — remove most ambiguous examples
        res = [{"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                "rejected": raw[i]["rejected"]} for i in correct]
        save_data(res, f"{dataset_name}/rw_gap_r_{p}")

        # rw_gap_f — flip most ambiguous examples
        res = []
        for i in wrong:
            res.append({"prompt": raw[i]["prompt"], "chosen": raw[i]["rejected"],
                        "rejected": raw[i]["chosen"]})
        for i in correct:
            res.append({"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                        "rejected": raw[i]["rejected"]})
        save_data(res, f"{dataset_name}/rw_gap_f_{p}")

    flagged = int(n * 0.2)
    print(f"  Most ambiguous (p=0.2): {flagged} examples flagged")

print("\n=== RwGap cleaning done! ===")
