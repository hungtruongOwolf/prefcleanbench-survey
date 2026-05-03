
import os, sys, json, time, random
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, "src")
from macros import DATASETS

BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
MAX_NEW_TOKENS = 100
MAX_PROMPT_LENGTH = 200
BATCH_SIZE = 16

VERSIONS = [
    "no_clean_20k",
    "llm_judge_r",   "llm_judge_f",
    "vote_all_r",    "vote_all_f",
    "vote_maj_r",    "vote_maj_f",
    "ins_tag_cmp",   "ins_tag_div",
    "ifd_r_0.2",     "ifd_gap_r_0.2",  "ifd_gap_f_0.2",
    "rw_gap_r_0.2",  "rw_gap_f_0.2",
]

PROGRESS_FILE = "outputs/gen_progress.json"
GEN_DIR = "generations"
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        done = json.load(f)
else:
    done = {}

print(f"Already generated: {len(done)} runs")

def mark_done(key):
    done[key] = True
    with open(PROGRESS_FILE, "w") as f:
        json.dump(done, f, indent=2)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Base model loaded!")

def generate_responses(model, prompts):
    results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
            padding=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            ).strip()
            results.append({"prompt": batch[j], "output": generated})
    return results

for dataset_name in DATASETS:
    print(f"\n=== {dataset_name} ===")

    test_data = []
    with open(f"datasets/{dataset_name}/no_clean/test.jsonl") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    random.seed(42)
    test_sample = random.sample(test_data, min(200, len(test_data)))

    # Format prompt to match training format
    prompts = [
        f"\n\nHuman: {d['prompt'][:MAX_PROMPT_LENGTH*4]}\n\nAssistant:"
        for d in test_sample
    ]
    print(f"  Test examples: {len(prompts)}")

    os.makedirs(f"{GEN_DIR}/{dataset_name}", exist_ok=True)
    current_model = None

    for version in VERSIONS:
        key = f"{dataset_name}__{version}"
        out_file = f"{GEN_DIR}/{dataset_name}/{version}.json"

        if done.get(key) or os.path.exists(out_file):
            print(f"  [SKIP] {version}")
            continue

        model_dir = f"models/{dataset_name}/{version}"
        if not os.path.exists(model_dir):
            print(f"  [MISSING] {model_dir}")
            continue

        print(f"  Loading adapter: {version}...")
        t0 = time.time()

        if current_model is not None:
            del current_model
            torch.cuda.empty_cache()

        current_model = PeftModel.from_pretrained(
            base_model, model_dir,
            adapter_name="current",
            is_trainable=False
        )
        current_model.eval()

        results = generate_responses(current_model, prompts)
        json.dump(results, open(out_file, "w"), indent=2)

        elapsed = time.time() - t0
        print(f"  ✅ {version} done in {elapsed/60:.1f} min")
        mark_done(key)

    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()

del base_model
torch.cuda.empty_cache()
print("\n=== ALL GENERATION DONE! ===")
print(f"Total: {len(done)} runs")
