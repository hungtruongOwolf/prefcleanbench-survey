
import os, sys, json, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from torch.optim import AdamW

sys.path.insert(0, "src")
from macros import DATASETS

BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
BETA = 0.1
MAX_LENGTH = 256
BATCH_SIZE = 8       # Larger now — only 1 model on GPU
GRAD_ACCUM = 2       # effective batch = 16
LR = 5e-5
WARMUP_STEPS = 50

VERSIONS = [
    "no_clean_20k",
    "llm_judge_r",   "llm_judge_f",
    "vote_all_r",    "vote_all_f",
    "vote_maj_r",    "vote_maj_f",
    "ins_tag_cmp",   "ins_tag_div",
    "ifd_r_0.2",     "ifd_gap_r_0.2",  "ifd_gap_f_0.2",
    "rw_gap_r_0.2",  "rw_gap_f_0.2",
]

PROGRESS_FILE = "outputs/training_progress.json"
os.makedirs("outputs", exist_ok=True)

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        done = json.load(f)
else:
    done = {}

print(f"Already done: {len(done)} runs")
for k in sorted(done.keys()):
    print(f"  ✅ {k}")

def mark_done(key, metrics=None):
    done[key] = metrics or True
    with open(PROGRESS_FILE, "w") as f:
        json.dump(done, f, indent=2)

def tokenize_batch(tokenizer, prompts, chosens, rejecteds):
    chosen_texts = [p + " " + c for p, c in zip(prompts, chosens)]
    rejected_texts = [p + " " + r for p, r in zip(prompts, rejecteds)]
    all_texts = chosen_texts + rejected_texts
    enc = tokenizer(
        all_texts, max_length=MAX_LENGTH, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    n = len(prompts)
    return (
        enc["input_ids"][:n], enc["attention_mask"][:n],
        enc["input_ids"][n:], enc["attention_mask"][n:]
    )

def compute_logps(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    logps = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none"
    ).reshape(targets.shape)
    return (logps * mask).sum(-1) / mask.sum(-1).clamp(min=1)

def compute_policy_logps(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    logps = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none"
    ).reshape(targets.shape)
    return (logps * mask).sum(-1) / mask.sum(-1).clamp(min=1)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected):
    logits = BETA * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    loss = -F.logsigmoid(logits).mean()
    acc = (policy_chosen - ref_chosen > policy_rejected - ref_rejected).float().mean()
    return loss, acc

for dataset_name in DATASETS:
    for version in VERSIONS:
        key = f"{dataset_name}__{version}__DPO"
        model_dir = f"models/{dataset_name}/{version}"
        data_path = f"datasets/{dataset_name}/{version}/train.jsonl"

        if done.get(key):
            print(f"[SKIP] {key}")
            continue

        if not os.path.exists(data_path):
            print(f"[MISSING] {data_path}")
            continue

        print(f"\n{'='*55}")
        print(f"Training: {key}")
        print(f"{'='*55}")
        t0 = time.time()

        # Load data
        raw = []
        with open(data_path) as f:
            for line in f:
                raw.append(json.loads(line.strip()))
        prompts   = [d["prompt"]   for d in raw]
        chosens   = [d["chosen"]   for d in raw]
        rejecteds = [d["rejected"] for d in raw]
        print(f"  Examples: {len(raw)}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        # ── STEP 1: Precompute ref logprobs (ref model on GPU, then delete) ──
        print("  Loading ref model (4-bit)...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto"
        )
        ref_model.eval()

        print("  Precomputing ref logprobs...")
        ref_chosen_all, ref_rejected_all = [], []
        pre_bs = 16
        for i in range(0, len(raw), pre_bs):
            bp = prompts[i:i+pre_bs]
            bc = chosens[i:i+pre_bs]
            br = rejecteds[i:i+pre_bs]
            c_ids, c_mask, r_ids, r_mask = tokenize_batch(tokenizer, bp, bc, br)
            device = next(ref_model.parameters()).device
            ref_chosen_all.append(compute_logps(ref_model, c_ids.to(device), c_mask.to(device)).cpu())
            ref_rejected_all.append(compute_logps(ref_model, r_ids.to(device), r_mask.to(device)).cpu())
            if (i // pre_bs) % 20 == 0:
                print(f"    ref logps: {i}/{len(raw)}")

        ref_chosen_logps = torch.cat(ref_chosen_all)    # stored in CPU RAM
        ref_rejected_logps = torch.cat(ref_rejected_all)
        print(f"  Ref logps computed — shape: {ref_chosen_logps.shape}")

        # Free ref model from GPU
        del ref_model
        torch.cuda.empty_cache()
        print("  Ref model freed from GPU.")

        # ── STEP 2: Tokenize all examples ──
        print("  Tokenizing policy inputs...")
        all_c_ids, all_c_mask, all_r_ids, all_r_mask = [], [], [], []
        for i in range(0, len(raw), pre_bs):
            bp = prompts[i:i+pre_bs]
            bc = chosens[i:i+pre_bs]
            br = rejecteds[i:i+pre_bs]
            c_ids, c_mask, r_ids, r_mask = tokenize_batch(tokenizer, bp, bc, br)
            all_c_ids.append(c_ids)
            all_c_mask.append(c_mask)
            all_r_ids.append(r_ids)
            all_r_mask.append(r_mask)

        all_c_ids   = torch.cat(all_c_ids)
        all_c_mask  = torch.cat(all_c_mask)
        all_r_ids   = torch.cat(all_r_ids)
        all_r_mask  = torch.cat(all_r_mask)

        train_dataset = TensorDataset(
            all_c_ids, all_c_mask, all_r_ids, all_r_mask,
            ref_chosen_logps, ref_rejected_logps
        )
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        # ── STEP 3: Load policy model in 4-bit (QLoRA) ──
        print("  Loading policy model 4-bit + LoRA (QLoRA)...")
        bnb_policy = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        policy_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_policy, device_map="auto"
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.enable_input_require_grads()
        policy_model.gradient_checkpointing_enable()
        policy_model.print_trainable_parameters()

        optimizer = AdamW(policy_model.parameters(), lr=LR, fused=True)
        total_steps = (len(dataloader) * 1) // GRAD_ACCUM
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
        )

        # ── STEP 4: Train ──
        print(f"  Training {total_steps} steps with batch_size={BATCH_SIZE}...")
        policy_model.train()
        optimizer.zero_grad()
        device = next(policy_model.parameters()).device

        running_loss = 0
        running_acc  = 0
        log_every = 50
        step = 0

        for batch_idx, batch in enumerate(dataloader):
            c_ids, c_mask, r_ids, r_mask, ref_c, ref_r = batch
            c_ids  = c_ids.to(device);  c_mask = c_mask.to(device)
            r_ids  = r_ids.to(device);  r_mask = r_mask.to(device)
            ref_c  = ref_c.to(device);  ref_r  = ref_r.to(device)

            pol_c = compute_policy_logps(policy_model, c_ids, c_mask)
            pol_r = compute_policy_logps(policy_model, r_ids, r_mask)

            loss, acc = dpo_loss(pol_c, pol_r, ref_c, ref_r)
            (loss / GRAD_ACCUM).backward()

            running_loss += loss.item()
            running_acc  += acc.item()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % log_every == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / step * (total_steps - step)
                    print(f"  step {step}/{total_steps} | loss={running_loss/log_every:.4f} | acc={running_acc/log_every:.3f} | ETA={eta/60:.0f}min")
                    running_loss = 0
                    running_acc  = 0

        # ── STEP 5: Save ──
        os.makedirs(model_dir, exist_ok=True)
        policy_model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        elapsed = time.time() - t0
        mark_done(key, {"time_min": round(elapsed/60, 1)})
        print(f"  ✅ Done in {elapsed/60:.1f} min → {model_dir}")

        del policy_model, train_dataset, all_c_ids, all_c_mask, all_r_ids, all_r_mask
        del ref_chosen_logps, ref_rejected_logps
        torch.cuda.empty_cache()

print("\n=== ALL TRAINING DONE! ===")
print(f"Completed: {len(done)} runs")
