
import json, os
import numpy as np

DATASETS = ["AnthropicHH", "UltraFeedback"]
IFD_DIR = "outputs/ifd"
PROPORTIONS = [0.1, 0.2, 0.3, 0.4]

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

    scores = np.loadtxt(f"{IFD_DIR}/{dataset_name}.tsv", delimiter="\t")
    ifd_score = scores[:, 0]   # IFD score for chosen
    ifd_gap   = scores[:, 1]   # IFD chosen - IFD rejected

    # ifd_r: remove lowest IFD scores (least informative prompts)
    ifd_idx = np.argsort(ifd_score)

    # ifd_gap_r / ifd_gap_f: sort by gap
    gap_idx = np.argsort(ifd_gap)

    for p in PROPORTIONS:
        n = len(raw)
        cut = int(n * p)

        # ifd_r — remove bottom p% by IFD score
        keep = ifd_idx[cut:].tolist()
        res = [{"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                "rejected": raw[i]["rejected"]} for i in keep]
        save_data(res, f"{dataset_name}/ifd_r_{p}")

        # ifd_gap_r — remove bottom p% by gap (most ambiguous)
        keep = gap_idx[cut:].tolist()
        res = [{"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                "rejected": raw[i]["rejected"]} for i in keep]
        save_data(res, f"{dataset_name}/ifd_gap_r_{p}")

        # ifd_gap_f — flip bottom p% by gap instead of removing
        wrong = gap_idx[:cut].tolist()
        correct = gap_idx[cut:].tolist()
        res = []
        for i in wrong:
            res.append({"prompt": raw[i]["prompt"], "chosen": raw[i]["rejected"],
                        "rejected": raw[i]["chosen"]})
        for i in correct:
            res.append({"prompt": raw[i]["prompt"], "chosen": raw[i]["chosen"],
                        "rejected": raw[i]["rejected"]})
        save_data(res, f"{dataset_name}/ifd_gap_f_{p}")

print("\n=== IFD cleaning done! ===")
