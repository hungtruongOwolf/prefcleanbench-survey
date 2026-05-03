from reward_models import ArmoRM, GRM

RMs_top_six = [GRM, ArmoRM]

DATASETS = ['AnthropicHH', 'UltraFeedback']

CLEANING_APPROACHES = ["LLM_Judge", "RwGap", "Voting", "Ins_Tag", "IFD"]
