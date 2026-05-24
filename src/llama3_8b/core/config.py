"""
统一配置文件：Meta-Llama-3.1-8B-Instruct generate-then-optimize 实验。

官方 generation_config.json (meta-llama/Llama-3.1-8B-Instruct):
  bos_token_id=128000, eos_token_id=[128001, 128008, 128009],
  do_sample=true, temperature=0.6, top_p=0.9.
tokenizer 含 chat_template，model_generation.py 会自动走 apply_chat_template 渲染 prompt；
Instruct 模型在摘要结束后会输出 EOS，StopOnSubstrings 作为兜底保留但通常不会触发。

Llama-3.1-8B-Instruct 上下文窗口 131072 tokens（带 rope_scaling），
本目录将单次输入上限固定为 8192 tokens 作为本地实验运行控制。
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


# ======================== Meta-Llama-3.1-8B-Instruct 生成配置 ========================
# 官方 generation_config.json (meta-llama/Llama-3.1-8B-Instruct):
#   do_sample=true, temperature=0.6, top_p=0.9
#   eos_token_id=[128001, 128008, 128009]  (多个 EOS token)
# 上下文窗口：131072 tokens（带 rope_scaling）
# Instruct 模型有 chat_template，会正确在摘要结束后输出 EOS，无需依赖 StopOnSubstrings
DEFAULT_GENERATOR = "llama3_8b"
GENERATOR_NAME = "llama3_8b"
GENERATION_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
MAX_GENERATION_INPUT_TOKENS = 8192   # 本地运行时输入上限；低于模型原生上下文窗口
MAX_GENERATION_NEW_TOKENS = 32768    # 本地运行时输出上限；模型遇到 EOS 会提前停止
GENERATION_TEMPERATURE = 0.6         # 官方 generation_config.json
GENERATION_TOP_P = 0.9               # 官方 generation_config.json
GENERATION_TOP_K = 0                 # Llama 不使用 top_k（0 = 禁用）
GENERATION_MIN_P = 0.0
GENERATION_PRESENCE_PENALTY = 0.0
GENERATION_REPETITION_PENALTY = 1.0
GENERATION_DO_SAMPLE = "auto"        # auto: baseline=sampling, CO=beam search

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
