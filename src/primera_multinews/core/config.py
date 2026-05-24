"""
统一配置文件：allenai/PRIMERA-multinews generate-then-optimize 实验。

官方 Hugging Face checkpoint:
  - model id: allenai/PRIMERA-multinews
  - architecture: LEDForConditionalGeneration
  - max_encoder_position_embeddings: 4096
  - generation_config: num_beams=5, no_repeat_ngram_size=3
  - task_specific_params.summarization:
      max_length=142, min_length=56, length_penalty=2.0, early_stopping=true

本目录只承载 PRIMERA-multinews 在 Multi-News 上的 baseline / CO 实验。
"""


# ======================== 生成阶段参数 ========================

NUM_SAMPLES = 0            # 0 = 使用完整测试集，不截断
BEAM_SIZE = 5              # 官方 generation_config 默认 num_beams=5
BUDGET_SENTENCES = 8       # Multi-News 最终摘要默认保留 8 句
MAX_INPUT_TOKENS = 4096    # PRIMERA encoder 上限
MAX_DECODE_STEPS = 142     # summarization task-specific max_length
MIN_DECODE_STEPS = 56      # summarization task-specific min_length
LENGTH_PENALTY = 2.0       # summarization task-specific length_penalty
NO_REPEAT_NGRAM = 3        # generation_config / task-specific 一致
TEMPERATURE = 1.0          # seq2seq beam 解码不使用采样；保留为兼容占位
DOC_SEPARATOR_TOKEN = "<doc-sep>"

# ======================== Batch Size 约束 ========================
# LED 4096-token 编码开销较高，回退值更保守
GENERATION_BATCH_SIZE = 2
UTILITY_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

# ======================== ROUGE 实现 ========================
DEFAULT_ROUGE_IMPL = "hf"

# ======================== 模型路径 ========================
DEFAULT_GENERATOR = "primera_multinews"
GENERATOR_NAME = "primera_multinews"
GENERATOR_DISPLAY_NAME = "PRIMERA-MultiNews"
MODEL_PATH = "allenai/PRIMERA-multinews"

# ======================== FactCC 配置 ========================
FACTCC_MODEL_PATH = "manueldeprada/FactCC"
FACTCC_BATCH_SIZE = 8


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
