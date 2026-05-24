"""
共用的特征计算模块：Utility（单句质量分数）和 Redundancy（句间冗余矩阵）。

【核心概念】
在 "Generate-then-Optimize" 流程中，生成阶段产出多个候选摘要后，
需要从中提取句子，计算每句的"好坏"和句间的"重复程度"，
然后交给选择器（ILP/MMR/DPP等）做优化选择。

本模块支持 4 种目标变体，通过 use_minicheck_utility 和 use_rouge_redundancy 两个标志控制：
  - utility 函数：ROUGE（覆盖度）或 MiniCheck（事实一致性）
  - redundancy 矩阵：ROUGE-L F1 矩阵 或 单位矩阵（不惩罚冗余）
"""

from rouge_score import rouge_scorer

from opt_selectors.tri_metric import normalize_tri_metric_weights, tri_metric_utility

# 创建一个全局的 ROUGE 评分器实例（避免重复初始化）
# use_stemmer=True 表示对单词做词干化（如 "running" → "run"），提高匹配率
feature_rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def load_utility_model(device, use_minicheck_utility, batch_size: int = 16):
    """
    按需加载 MiniCheck 模型（仅在 minicheck_only / minicheck_redundancy 变体中使用）。

    参数:
        device: 计算设备（cuda/cpu）
        use_minicheck_utility: 是否使用 MiniCheck 作为 utility 函数

    返回:
        minicheck_scorer — MiniCheck 实例，或 None
    """
    if not use_minicheck_utility:
        return None

    from metrics.minicheck_eval_utils import load_minicheck_model
    print("Loading MiniCheck model for utility computation...")
    return load_minicheck_model(device=device, batch_size=batch_size)


def compute_rouge_utility_scores(unique_sentences, article):
    """coverage utility = ROUGE-1 Recall + ROUGE-2 Recall."""
    utility_scores = []
    for sent in unique_sentences:
        result = feature_rouge_scorer.score(article, sent)
        r1_recall = result["rouge1"].recall
        r2_recall = result["rouge2"].recall
        utility_scores.append(r1_recall + r2_recall)
    return utility_scores


def compute_minicheck_utility_scores(unique_sentences, article, minicheck_scorer):
    """faithfulness utility = MiniCheck P(consistent | article, sentence)."""
    if minicheck_scorer is None:
        raise ValueError("minicheck_scorer is required for MiniCheck-based utility computation.")

    _, raw_probs, _, _ = minicheck_scorer.score(
        docs=[article] * len(unique_sentences),
        claims=list(unique_sentences),
    )
    return [float(prob) for prob in raw_probs]


def compute_utility_scores(unique_sentences, article, use_minicheck_utility,
                           minicheck_scorer=None, segmenter=None, device=None):
    """
    计算每个候选句子的 utility（质量得分）。

    【两种模式】
    1. use_minicheck_utility=True（minicheck 变体）:
       - 用 MiniCheck 模型计算 P(句子与原文一致)

    2. use_minicheck_utility=False（rouge 变体）:
       - 用 ROUGE-1 Recall + ROUGE-2 Recall

    参数:
        unique_sentences: 去重后的候选句子列表
        article: 原始文章文本
        use_minicheck_utility: 是否用 MiniCheck 模型
        minicheck_scorer: MiniCheck 评分器实例
        segmenter: pysbd 句子分割器
        device: 计算设备

    返回:
        utility_scores: 长度为 M 的列表，每个句子一个得分
    """
    if use_minicheck_utility:
        return compute_minicheck_utility_scores(unique_sentences, article, minicheck_scorer)

    return compute_rouge_utility_scores(unique_sentences, article)


def _per_sample_min_max(scores):
    """Min-max normalize a list of scores to [0, 1] within the sample."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def compute_tri_metric_utility_scores(
    unique_sentences,
    article,
    minicheck_scorer,
    w_rouge: float,
    w_minicheck: float,
    w_redundancy: float,
):
    """Compute weighted utility plus the per-sentence component breakdown."""
    effective_weights, _ = normalize_tri_metric_weights(w_rouge, w_minicheck, w_redundancy)
    rouge_scores = compute_rouge_utility_scores(unique_sentences, article)
    minicheck_scores = compute_minicheck_utility_scores(unique_sentences, article, minicheck_scorer)
    calibrated_rouge_scores = _per_sample_min_max(rouge_scores)
    calibrated_minicheck_scores = _per_sample_min_max(minicheck_scores)
    utility_scores = [
        tri_metric_utility(
            rouge_score=calibrated_rouge_score,
            minicheck_score=calibrated_minicheck_score,
            w_rouge=effective_weights["rouge"],
            w_minicheck=effective_weights["minicheck"],
        )
        for calibrated_rouge_score, calibrated_minicheck_score in zip(calibrated_rouge_scores, calibrated_minicheck_scores)
    ]
    metadata = {
        "rouge_scores": rouge_scores,
        "minicheck_scores": minicheck_scores,
        "calibrated_rouge_scores": calibrated_rouge_scores,
        "calibrated_minicheck_scores": calibrated_minicheck_scores,
        "effective_weights": effective_weights,
    }
    return utility_scores, metadata


def compute_redundancy_matrix(unique_sentences, use_rouge_redundancy):
    """
    计算句间冗余矩阵 R[i][j]。

    【两种模式】
    1. use_rouge_redundancy=True: R[i][j] = ROUGE-L F1(句子i, 句子j)
    2. use_rouge_redundancy=False: R 是单位矩阵（不做冗余惩罚）

    参数:
        unique_sentences: 去重后的候选句子列表
        use_rouge_redundancy: 是否计算真正的冗余矩阵

    返回:
        matrix: N×N 的嵌套列表
    """
    N = len(unique_sentences)

    if not use_rouge_redundancy:
        matrix = [[0.0] * N for _ in range(N)]
        for i in range(N):
            matrix[i][i] = 1.0
        return matrix

    matrix = [[0.0] * N for _ in range(N)]
    for i in range(N):
        matrix[i][i] = 1.0
        for j in range(i + 1, N):
            result = feature_rouge_scorer.score(unique_sentences[i], unique_sentences[j])
            f1 = result['rougeL'].fmeasure
            matrix[i][j] = f1
            matrix[j][i] = f1
    return matrix
