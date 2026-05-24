"""
统一配置文件：gemma-4-E4B-it (Instruct) generate-then-optimize 实验。

Google gemma-4-E4B-it 使用 Gemma 系列默认参数：
temperature=1.0, top_k=64, top_p=0.95。
Instruct 模型含 chat_template，model_generation.py 会自动走 apply_chat_template 渲染 prompt；
模型在摘要完成后会输出 EOS，不会无限续写。
"""


# ======================== 生成阶段参数 ========================

NUM_SAMPLES = 0            # 0 = 使用完整测试集，不截断
BEAM_SIZE = 8
BUDGET_SENTENCES = 4

# ======================== CO beam search 默认值 ========================
CO_NUM_BEAMS = 8           # 默认 CO beam width；运行时可用 --beam-size / BEAM_SIZE 覆盖

# ======================== Batch Size 约束 ========================
GENERATION_BATCH_SIZE = 8
UTILITY_BATCH_SIZE = 12
EVAL_BATCH_SIZE = 12

# ======================== ROUGE 实现 ========================
DEFAULT_ROUGE_IMPL = "hf"


# ======================== gemma-4-E4B-it (Instruct) 生成配置 ========================
# 官方 generation_config.json (google/gemma-4-E4B-it):
#   do_sample=true, temperature=1.0, top_k=64, top_p=0.95
#   eos_token_id=[1, 106, 50]  (多个 EOS token)
# Instruct 模型有 chat_template，生成完摘要后会自动输出 EOS token，不会用满 max_new_tokens
DEFAULT_GENERATOR = "gemma_4_e4b"
GENERATOR_NAME = "gemma_4_e4b"
GENERATION_MODEL_PATH = "google/gemma-4-E4B-it"
MAX_GENERATION_INPUT_TOKENS = 8192
MAX_GENERATION_NEW_TOKENS = 32768    # 本地运行时输出上限；模型遇到 EOS 提前停止，不会用满
GENERATION_TEMPERATURE = 1.0          # 官方 generation_config.json
GENERATION_TOP_P = 0.95               # 官方 generation_config.json
GENERATION_TOP_K = 64                 # 官方 generation_config.json
GENERATION_MIN_P = 0.0
GENERATION_PRESENCE_PENALTY = 0.0     # Gemma 不使用 presence penalty
GENERATION_REPETITION_PENALTY = 1.0
GENERATION_DO_SAMPLE = "auto"         # auto: baseline=sampling, CO=beam search

# ======================== FactCC 配置 ========================
FACTCC_MODEL_PATH = "manueldeprada/FactCC"
FACTCC_BATCH_SIZE = 16


# ======================== 目标变体定义 ========================
OBJECTIVE_VARIANTS = {
    "rouge_only": {
        "description": "ROUGE only",
        "utility": "ROUGE-1 Recall + ROUGE-2 Recall",
        "redundancy": "disabled (identity matrix)",
        "use_minicheck_utility": False,
        "use_rouge_redundancy": False,
    },
    "rouge_redundancy": {
        "description": "ROUGE + Redundancy",
        "utility": "ROUGE-1 Recall + ROUGE-2 Recall",
        "redundancy": "ROUGE-L F1",
        "use_minicheck_utility": False,
        "use_rouge_redundancy": True,
    },
    "minicheck_only": {
        "description": "MiniCheck only",
        "utility": "MiniCheck P(consistent | article, sentence)",
        "redundancy": "disabled (identity matrix)",
        "use_minicheck_utility": True,
        "use_rouge_redundancy": False,
    },
    "minicheck_redundancy": {
        "description": "MiniCheck + Redundancy",
        "utility": "MiniCheck P(consistent | article, sentence)",
        "redundancy": "ROUGE-L F1",
        "use_minicheck_utility": True,
        "use_rouge_redundancy": True,
    },
}


# ======================== 各选择器的专用超参数 ========================

ILP_REDUNDANCY_THRESHOLD = 0.6
ILP_SCALE = 10000

MMR_LAMBDA = 0.5


# ======================== 三指标统一目标函数 ========================
TRI_ROUGE_WEIGHT = 0.01
TRI_MINICHECK_WEIGHT = 0.495
TRI_REDUNDANCY_WEIGHT = 0.495

TRI_METRIC_WEIGHTS_BY_METHOD = {
    "dpp": {"rouge": 0.0100, "minicheck": 0.4950, "redundancy": 0.4950},
    "mmr": {"rouge": 0.1000, "minicheck": 0.2000, "redundancy": 0.7000},
    "ilp": {"rouge": 0.1000, "minicheck": 0.2000, "redundancy": 0.7000},
}
