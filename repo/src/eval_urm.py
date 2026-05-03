
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "/content/drive/MyDrive/PrefClean/repo"
GEN_DIR = f"{REPO}/generations"
RESULTS_DIR = f"{REPO}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# URM reward model
URM_MODEL = "LxzGordon/URM-LLaMa-3.1-8B"

print(f"Loading URM: {URM_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(URM_MODEL)
model = AutoModelForCausalLM.from_pretrained(URM_MODEL, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

# Score each generation file
gen_files = sorted([f for f in os.listdir(GEN_DIR) if f.endswith("_gen.jsonl")])
results = {}

for gen_file in gen_files:
    version = gen_file.replace("AnthropicHH_", "").replace("_gen.jsonl", "")
    output_file = f"{RESULTS_DIR}/{version}_scores.json"
    
    if os.path.exists(output_file):
        print(f"✓ {version} already scored, skipping")
        continue
    
    print(f"\nScoring {version}...")
    scores = []
    
    with open(f"{GEN_DIR}/{gen_file}") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            prompt = item["prompt"]
            response = item["response"]
            
            # Format as: prompt + response (URM expects full conversation)
            text = f"{prompt}\n{response}"
            
            # Tokenize & score
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            with torch.no_grad():
                logits = model(**inputs).logits
                # Reward is last token logit for class 1 (good)
                score = logits[0, -1, 1].item()
            
            scores.append(score)
            
            if (i + 1) % 100 == 0:
                print(f"  Scored {i+1} samples...")
    
    # Save scores
    with open(output_file, "w") as f:
        json.dump({"version": version, "scores": scores}, f, indent=2)
    
    avg_score = sum(scores) / len(scores)
    print(f"✅ {version}: avg_score={avg_score:.4f}, n={len(scores)}")
    results[version] = {"avg_score": avg_score, "n": len(scores)}

print(f"\n{'='*60}")
print("Results summary:")
for v in sorted(results.keys()):
    print(f"  {v:30s} {results[v]['avg_score']:7.4f}")
