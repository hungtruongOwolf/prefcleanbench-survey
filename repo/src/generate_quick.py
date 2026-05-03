
import os, json, time, torch, random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

REPO = "/content/drive/MyDrive/PrefClean/repo"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET = "AnthropicHH"
TEST_DATA = f"{REPO}/datasets/{DATASET}/no_clean/test.jsonl"
GEN_DIR = f"{REPO}/generations"
os.makedirs(GEN_DIR, exist_ok=True)

# Only 6 key versions (drop 8 to save time, use training metrics for those)
VERSIONS = [
    "no_clean_20k",      # baseline
    "llm_judge_r",       # weak
    "vote_all_r",        # strong
    "vote_maj_r",        # strong
    "rw_gap_r_0.2",      # very strong
    "rw_gap_f_0.2",      # strong
]

print("Loading base model (4-bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto",
    attn_implementation="sdpa", torch_dtype=torch.bfloat16,
)
model.config.use_cache = True

# Load test data
print(f"Loading test data from {TEST_DATA}")
all_samples = []
with open(TEST_DATA) as f:
    for line in f:
        all_samples.append(json.loads(line))

# Sample 1000 random (balanced)
random.seed(42)
test_samples = random.sample(all_samples, min(1000, len(all_samples)))
print(f"Using {len(test_samples)} test samples (subsampled from {len(all_samples)})")

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
    
    batch_size = 16  # larger batch
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]
        prompts = [s["prompt"] for s in batch]
        
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to("cuda")
        
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128,
                do_sample=True, temperature=0.7, top_p=0.9,
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
        
        if (i // batch_size + 1) % 5 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  Processed {i+batch_size}/{len(test_samples)} ({elapsed:.1f} min)")
    
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    elapsed = (time.time() - t0) / 60
    print(f"✅ {version}: {len(results)} samples in {elapsed:.1f} min")
    model = model.unload()

print(f"\n✅ Generation complete for {len(VERSIONS)} key versions")
