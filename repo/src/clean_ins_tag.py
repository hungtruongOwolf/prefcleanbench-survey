import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from macros import DATASETS


tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/InsTagger")
model = AutoModelForCausalLM.from_pretrained("OFA-Sys/InsTagger", device_map='auto',
    trust_remote_code=True)
model.eval().requires_grad_(False)

save_path = f'outputs/ins_tag'
os.makedirs(save_path, exist_ok=True)

for dataset in DATASETS:
    train_dataset = load_dataset("json", data_files=f"datasets/{dataset}/no_clean/train.jsonl")['train']

    res = []
    for data in tqdm(train_dataset):
        prompt = tokenizer(data['prompt'], max_length=512, return_tensors="pt", truncation=True)
        
        cnt = 3
        while cnt:
            try:
                output = model.generate(
                    input_ids=prompt.input_ids.to('cuda'),
                    attention_mask=prompt.attention_mask.to('cuda'),
                    max_new_tokens=512,
                    do_sample=True,
                )
                policy_output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                prompt = tokenizer.batch_decode(prompt.input_ids, skip_special_tokens=True)[0]
                tags = policy_output_decoded[len(prompt):]
                if not tags.endswith('"}]'):
                    tags += '"}]'

                res.append([x['tag'] for x in json.loads(tags)])
                break
            except:
                cnt -= 1
                print(tags)
        if cnt == 0:
            res.append([])
        

    json.dump(res, open(f'{save_path}/{dataset}.json', 'w'))
