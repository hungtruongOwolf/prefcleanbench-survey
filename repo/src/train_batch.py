
import os, subprocess, time, json

REPO = "/content/drive/MyDrive/PrefClean/repo"
PROGRESS = f"{REPO}/outputs/training_progress.json"

# 14 versions cho AnthropicHH
ALL_VERSIONS = [
    "no_clean_20k",         # baseline
    "llm_judge_r", "llm_judge_f",
    "vote_all_r", "vote_all_f",
    "vote_maj_r", "vote_maj_f",
    "ins_tag_cmp", "ins_tag_div",
    "ifd_r_0.2", "ifd_gap_r_0.2", "ifd_gap_f_0.2",
    "rw_gap_r_0.2", "rw_gap_f_0.2",
]

# Load progress
done = set()
if os.path.exists(PROGRESS):
    with open(PROGRESS) as f:
        done = set(json.load(f).get("done", []))

# Auto-detect: nếu adapter đã exist thì skip
for v in ALL_VERSIONS:
    if os.path.exists(f"{REPO}/models/AnthropicHH/{v}/adapter_model.safetensors"):
        done.add(v)

todo = [v for v in ALL_VERSIONS if v not in done]
print(f"=== Training plan ===")
print(f"Done ({len(done)}): {sorted(done)}")
print(f"Todo ({len(todo)}): {todo}")
print(f"Estimated time: {len(todo) * 25} min = {len(todo) * 25 / 60:.1f} hours")
print()

t_start = time.time()
for i, v in enumerate(todo):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(todo)}] Training {v}")
    print(f"Elapsed so far: {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")
    
    result = subprocess.run(
        ["python", f"{REPO}/src/train_dpo.py", v],
        cwd=REPO,
    )
    
    if result.returncode == 0:
        done.add(v)
        with open(PROGRESS, "w") as f:
            json.dump({"done": sorted(done)}, f, indent=2)
        print(f"✅ {v} done. Progress saved.")
    else:
        print(f"❌ {v} failed (returncode {result.returncode})")
        print("Stopping batch. Re-run script to resume.")
        break

print(f"\n=== Batch complete ===")
print(f"Total time: {(time.time()-t_start)/60:.1f} min")
print(f"Done: {len(done)}/{len(ALL_VERSIONS)}")
