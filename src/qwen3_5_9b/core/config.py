"""
统一配置文件：Qwen3.5-9B generate-then-optimize 实验。

本文件集中管理了原来分散在 24 个脚本中的所有超参数，
修改一处即可同步所有实验，避免参数不一致的问题。

运行时说明:
  - 这里保留的是保守默认值，避免全量生成/评估时把机器打满
  - 如需更激进的吞吐，优先在启动脚本或 CLI 显式覆盖 batch
"""


# ======================== 生成阶段参数 ========================

NUM_SAMPLES = 0            # 0 = 使用完整测试集，不截断
BEAM_SIZE = 8              # Beam Search 的束宽（每步保留 top-8 个候选序列）
BUDGET_SENTENCES = 4       # 最终摘要的句子数上限（从候选池中选 4 句拼成摘要）

# ======================== Batch Size 约束 ========================
# 保守回退值；优先在启动脚本或 CLI 显式覆盖
GENERATION_BATCH_SIZE = 8   # 生成阶段 batch size 默认值
UTILITY_BATCH_SIZE = 12    # MiniCheck utility / selector scorer 回退值
EVAL_BATCH_SIZE = 12       # 评估阶段通用 batch size 回退值

# ======================== ROUGE 实现 ========================
DEFAULT_ROUGE_IMPL = "hf"  # 默认使用 HuggingFace evaluate ROUGE (official experiment setting)


# ======================== Qwen3.5-9B 生成配置 ========================
DEFAULT_GENERATOR = "qwen3.5_9B"
GENERATOR_NAME = "qwen3.5_9B"
GENERATION_MODEL_PATH = "Qwen/Qwen3.5-9B"
MAX_GENERATION_INPUT_TOKENS = 8192
MAX_GENERATION_NEW_TOKENS = 32768
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.8
GENERATION_TOP_K = 20
GENERATION_MIN_P = 0.0
GENERATION_PRESENCE_PENALTY = 1.5
GENERATION_REPETITION_PENALTY = 1.0
GENERATION_DO_SAMPLE = "auto"

# ======================== FactCC 配置 ========================
# FactCC 是一个事实一致性分类器，在评估阶段始终使用
FACTCC_MODEL_PATH = "manueldeprada/FactCC"  # HuggingFace 上的 FactCC 模型
FACTCC_BATCH_SIZE = 16                       # FactCC 推理时的批大小


# ======================== 目标变体定义 ========================
# 我们的实验有 4 种"目标变体"，区别在于：
#   - utility 函数：用 ROUGE（覆盖度）还是 MiniCheck（事实一致性）
#   - redundancy 矩阵：是否启用句间冗余惩罚
#
# 每个句子级选择器（ILP/MMR/DPP）都会跑这 4 种变体，
# 所以总共有 5×4 = 20 个句子级实验 + 2 个摘要级实验 = 22 个实验
#
# 【重要】minicheck_only 和 minicheck_redundancy 变体的 utility 函数
# 使用的是 MiniCheck（句子级 P(consistent)），不是 FactCC。
# use_minicheck_utility=True 表示在优化选择阶段加载 MiniCheck 模型。

OBJECTIVE_VARIANTS = {
    # 变体1：只用 ROUGE 作为 utility，不做冗余惩罚
    "rouge_only": {
        "description": "ROUGE only",
        "utility": "ROUGE-1 Recall + ROUGE-2 Recall",       # utility = R1回召 + R2回召
        "redundancy": "disabled (identity matrix)",           # 冗余矩阵是单位矩阵（即不惩罚）
        "use_minicheck_utility": False,                       # 不加载 MiniCheck 模型
        "use_rouge_redundancy": False,                        # 不计算 ROUGE-L 冗余矩阵
    },
    # 变体2：ROUGE utility + ROUGE-L 冗余惩罚
    "rouge_redundancy": {
        "description": "ROUGE + Redundancy",
        "utility": "ROUGE-1 Recall + ROUGE-2 Recall",
        "redundancy": "ROUGE-L F1",                           # 用 ROUGE-L F1 衡量句间冗余
        "use_minicheck_utility": False,
        "use_rouge_redundancy": True,                          # 启用冗余矩阵计算
    },
    # 变体3：MiniCheck 事实一致性作为 utility，不做冗余惩罚
    "minicheck_only": {
        "description": "MiniCheck only",
        "utility": "MiniCheck P(consistent | article, sentence)",  # P(句子与原文一致)
        "redundancy": "disabled (identity matrix)",
        "use_minicheck_utility": True,                         # 加载 MiniCheck 模型计算 utility
        "use_rouge_redundancy": False,
    },
    # 变体4：MiniCheck utility + ROUGE-L 冗余惩罚（最完整的变体）
    "minicheck_redundancy": {
        "description": "MiniCheck + Redundancy",
        "utility": "MiniCheck P(consistent | article, sentence)",
        "redundancy": "ROUGE-L F1",
        "use_minicheck_utility": True,
        "use_rouge_redundancy": True,
    },
}


# ======================== 各选择器的专用超参数 ========================

# --- ILP（整数线性规划）---
ILP_REDUNDANCY_THRESHOLD = 0.6  # 冗余阈值：句对的 ROUGE-L > 0.6 时，ILP 约束它们不能同选
ILP_SCALE = 10000               # utility 放大倍数（PuLP 用整数运算，所以要放大避免精度丢失）

# --- MMR（最大边际相关性）---
MMR_LAMBDA = 0.5  # 平衡系数：λ*相关性 - (1-λ)*冗余，λ=0.5 表示两者权重相等


# ======================== 三指标统一目标函数 ========================
# unified utility = w_rouge * coverage + w_minicheck * faithfulness
# method-specific redundancy terms continue to be handled inside each selector
# 默认 fallback 权重；运行时若保持默认值，会按 method-specific 配置覆盖。
TRI_ROUGE_WEIGHT = 0.01
TRI_MINICHECK_WEIGHT = 0.495
TRI_REDUNDANCY_WEIGHT = 0.495

# 方法特定权重。权重按原始数值使用，不做 sum-normalization。
TRI_METRIC_WEIGHTS_BY_METHOD = {
    "dpp": {"rouge": 0.0100, "minicheck": 0.4950, "redundancy": 0.4950},
    "mmr": {"rouge": 0.1000, "minicheck": 0.2000, "redundancy": 0.7000},
    "ilp": {"rouge": 0.1000, "minicheck": 0.2000, "redundancy": 0.7000},
}
