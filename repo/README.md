# PrefCleanBench

This is the source code implementing the protocol of benchmarking preference data cleaning approaches, as proposed in the paper, ***Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment***

## Evaluation Pipeline

### Datasets Preparation

First, run
```
python src/proc_dataset.py
```
to download and process target datasets (AnthropicHH, UltraFeedback, PKU-SafeRLHF, and HelpSteer2).

### Clean Up Datasets

We implement 5 data cleaning approaches, including LLM_Judge, RwGap, Voting, InsTag, and IFD. The following scripts are used to compute scores for each cleaning approaches:
- LLM_Judge: `python src/clean_llm_judge.py`
- RwGap: `python src/clean_rw_gap.py`
- Voting: `python src/clean_voting.py`
- InsTag: `python src/clean_ins_tag.py`
- IFD: `python src/clean_ifd.py`

Note that the following command 
```
bash scripts/train_rw_gap_model.sh
```
should be run before running `python src/clean_rw_gap.py`.

After computing these scores, you can run
```
python data_cleaning.py <LLM_Judge|RwGap|Voting|InsTag|IFD>
```
to clean up datasets.

After running all cleaning approaches, you would obtain the following cleansed versions of datasets:
- `llm_judge_<r|f>`
- `rw_gap_r_<0.1|0.2|0.3|0.4>`
- `rw_gap_f_<0.1|0.2|0.3|0.4>`
- `vote_all_<r|f>`
- `vote_maj_<r|f>`
- `ins_tag_<cmp|div>`
- `ifd_r_<0.1|0.2|0.3|0.4>`
- `ifd_gap_r_<0.1|0.2|0.3|0.4>`
- `ifd_gap_f_<0.1|0.2|0.3|0.4>`

### Training & Generating Responses

First, run
```
bash script/run_training.sh no_clean <Base_model> <Optimization_algorithm> [Pretty_name]
```
to obtain models trained on datasets without cleaning. Note that the `Pretty_name` is a name used to save the trained models and its generations. By default, the name is set to be `{Optimization_algorithm}_{Base_model}_no_clean`.

We provide the following options of `Base_model`:
- `llama3_8b`: meta-llama/Meta-Llama-3-8B
- `wqen2.5_7b`: Qwen/Qwen2.5-7B
- `phi2`: microsoft/phi-2
- `mistral_7b`: mistralai/Mistral-7B-v0.3

For `Optimization_algorithm`, we provide the following options:
- `DPO`
- `CPO`
- `AOT`
- `KTO`
- `IPO`
- `SLiC`
- `rDPO`
- `ORPO`

Then, run
```
bash script/run_training.sh <Cleansed_version> <Base_model> <Optimization_algorithm> [Pretty_name]
```
to obtain models trained on cleansed versions of datasets, where `Cleansed_version` should be set to one of values in the list:
- `llm_judge_<r|f>`
- `rw_gap_r_<0.1|0.2|0.3|0.4>`
- `rw_gap_f_<0.1|0.2|0.3|0.4>`
- `vote_all_<r|f>`
- `vote_maj_<r|f>`
- `ins_tag_<cmp|div>`
- `ifd_r_<0.1|0.2|0.3|0.4>`
- `ifd_gap_r_<0.1|0.2|0.3|0.4>`
- `ifd_gap_f_<0.1|0.2|0.3|0.4>`
By default, `Pretty_name` is set to be `{Optimization_algorithm}_{Base_model}_{Cleansed_version}`.

Once models are trained, run
```
python src/gen_response.py <Pretty_name>
```
to generate responses by prompting models with prompts in the test set of each dataset.

### Evaluation

We provide scripts to compute win-tie rate and avg. gold reward respectively:
```
# win-tie rate
python src/eval_wintie.py <Pretty_name_cleansed> <Pretty_name_no_clean>
```

```
# avg. gold reward
python src/eval_avg_rwd.py <Pretty_name>
```