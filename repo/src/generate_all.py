
import os, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

REPO = "/content/drive/MyDrive/PrefClean/repo"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET = "AnthropicHH"
TEST_DATA = f"{REPO}/datasets/{DATASET}/no_clean/test.jsonl"
GEN_DIR = f"{REPO}/generations"
os.makedirs(GEN_DIR, exist_ok=True)

VERSIONS = [
    "no_clean_20k", "llm_judge_r", "llm_judge_f",
    "vote_all_r", "vote_all_f", "vote_maj_r", "vote_maj_f",
    "ins_tag_cmp", "ins_tag_div",
    "ifd_r_0.2", "ifd_gap_r_0.2", "ifd_gap_f_0.2",
    "rw_gap_r_0.2", "rw_gap_f_0.2",
]

print("Loading base model (4-bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"  # FIX: set left padding for generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = True

print(f"Loading test data from {TEST_DATA}")
test_samples = []
with open(TEST_DATA) as f:
    for line in f:
        test_samples.append(json.loads(line))
print(f"Loaded {len(test_samples)} test samples")

for version in VERSIONS:
    output_file = f"{GEN_DIR}/{DATASET}_{version}_gen.jsonl"
    if os.path.exists(output_file):
        print(f"✓ {version} already generated, skipping")
        continue
    
    print(f"\n{'='*60}")
    print(f"Generating {version}")
    print(f"{'='*60}")
    
    adapter_path = f"{REPO}/models/{DATASET}/{version}"
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    results = []
    t0 = time.time()
    
    batch_size = 8
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]
        prompts = [s["prompt"] for s in batch]
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to("cuda")
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        for j, token_ids in enumerate(out):
            response = tokenizer.decode(
                token_ids[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            results.append({
                "prompt": prompts[j],
                "response": response,
            })
        
        if (i // batch_size + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  Processed {i+batch_size}/{len(test_samples)} ({elapsed:.1f} min)")
    
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    elapsed = (time.time() - t0) / 60
    print(f"✅ {version}: {len(results)} samples in {elapsed:.1f} min")
    
    model = model.unload()

print(f"\n{'='*60}")
print(f"✅ Generation complete for all {len(VERSIONS)} versions")
