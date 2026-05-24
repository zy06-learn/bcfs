"""
MMR（Maximal Marginal Relevance）句子选择器。

MMR(i) = λ × utility(i) - (1-λ) × max_j∈S redundancy(i, j)
"""

from core.config import MMR_LAMBDA
from opt_selectors.tri_metric import redundancy_weight_to_lambda


def mmr_select(
    unique_sentences,
    utility_scores,
    redundancy_matrix,
    budget,
    lambta=None,
    utility_mode="legacy",
    tri_metric_weights=None,
):
    """用 MMR 贪心地逐句选择，平衡信息量和多样性。"""
    if utility_mode == "tri_metric" and lambta is None and tri_metric_weights is not None:
        lambta = redundancy_weight_to_lambda(tri_metric_weights["redundancy"])
    if lambta is None:
        lambta = MMR_LAMBDA

    M = len(unique_sentences)
    if M <= budget:
        return list(range(M))

    selected = []
    remaining = list(range(M))

    for _ in range(budget):
        if not remaining:
            break

        best_idx = None
        best_mmr = float('-inf')

        for idx in remaining:
            relevance = lambta * utility_scores[idx]

            if selected:
                max_red = max(redundancy_matrix[idx][s] for s in selected)
            else:
                max_red = 0.0

            mmr_score = relevance - (1 - lambta) * max_red

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return sorted(selected)
