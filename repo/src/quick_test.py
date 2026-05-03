
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER = "/content/drive/MyDrive/PrefClean/repo/models/AnthropicHH/no_clean_20k"
TEST = "/content/drive/MyDrive/PrefClean/repo/datasets/AnthropicHH/no_clean/test.jsonl"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
tok = AutoTokenizer.from_pretrained(ADAPTER)
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", attn_implementation="eager")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

# Load 3 test prompts
with open(TEST) as f:
    samples = [json.loads(l) for l in f][:3]

for i, s in enumerate(samples):
    prompt = s["prompt"]
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"=== Sample {i+1} ===")
    print(f"PROMPT: {prompt[:200]}...")
    print(f"RESPONSE: {response}")
    print()
