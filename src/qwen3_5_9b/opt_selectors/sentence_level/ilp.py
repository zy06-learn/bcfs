"""
ILP（整数线性规划）句子选择器。

把"从候选池中选 k 句组成最优摘要"建模为 0-1 整数规划问题。

两条路径:
  - default / minicheck_* 变体: 硬约束 ILP
      max Σ u_i x_i
      s.t. Σ x_i == budget
           x_i + x_j <= 1  当 R[i][j] > threshold
  - tri_metric 变体: soft ILP（pairwise 冗余惩罚）
      max Σ u_i x_i  −  α Σ_{i<j} R[i][j] y_ij
      s.t. 1 <= Σ x_i <= budget
           y_ij <= x_i
           y_ij <= x_j
           y_ij >= x_i + x_j − 1
           x_i, y_ij ∈ {0,1}
      其中 α 由 penalty_scale 控制:
        per_edge     → α = w_redundancy / (budget − 1)   [默认, 已验证]
        per_sentence → α = w_redundancy / budget
        per_pair     → α = w_redundancy / C(budget, 2)
"""

from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, lpSum, PULP_CBC_CMD

from core.config import ILP_REDUNDANCY_THRESHOLD, ILP_SCALE


def _fallback_topk(utility_scores, budget):
    indexed = sorted(enumerate(utility_scores), key=lambda x: x[1], reverse=True)
    return sorted(idx for idx, _ in indexed[:budget])


def _solve_hard_ilp(utility_scores, redundancy_matrix, budget, redundancy_threshold):
    M = len(utility_scores)
    x = [LpVariable(f"x{i}", 0, 1, LpInteger) for i in range(M)]
    prob = LpProblem("ILP_Utility_Redundancy", LpMaximize)
    prob += lpSum(int(utility_scores[i] * ILP_SCALE) * x[i] for i in range(M))
    prob += lpSum(x) == budget

    for i in range(M):
        for j in range(i + 1, M):
            if redundancy_matrix[i][j] > redundancy_threshold:
                prob += x[i] + x[j] <= 1

    prob.solve(PULP_CBC_CMD(msg=0))
    selected = [i for i in range(M) if x[i].varValue is not None and x[i].varValue > 0.5]
    return selected


_PENALTY_SCALE_CHOICES = ("per_edge", "per_sentence", "per_pair")


def _alpha_from_scale(w_redundancy, budget, penalty_scale):
    """Map --ilp-penalty-scale to the pairwise-penalty coefficient alpha."""
    w = float(w_redundancy)
    if penalty_scale == "per_edge":
        return w / max(1, budget - 1)
    if penalty_scale == "per_sentence":
        return w / max(1, budget)
    if penalty_scale == "per_pair":
        pairs = max(1, budget * (budget - 1) // 2)
        return w / pairs
    raise ValueError(f"Unknown penalty_scale={penalty_scale!r}; expected one of {_PENALTY_SCALE_CHOICES}")


def _solve_soft_ilp(utility_scores, redundancy_matrix, budget, w_redundancy, penalty_scale):
    """Soft ILP with pairwise redundancy penalty for tri-metric mode."""
    M = len(utility_scores)

    x = [LpVariable(f"x{i}", 0, 1, LpInteger) for i in range(M)]
    y = {}
    for i in range(M):
        for j in range(i + 1, M):
            y[(i, j)] = LpVariable(f"y_{i}_{j}", 0, 1, LpInteger)

    prob = LpProblem("ILP_Soft_TriMetric", LpMaximize)

    alpha = _alpha_from_scale(w_redundancy, budget, penalty_scale)

    utility_terms = [int(utility_scores[i] * ILP_SCALE) * x[i] for i in range(M)]
    penalty_terms = [
        int(alpha * float(redundancy_matrix[i][j]) * ILP_SCALE) * y[(i, j)]
        for i in range(M)
        for j in range(i + 1, M)
    ]
    prob += lpSum(utility_terms) - lpSum(penalty_terms)

    # budget range: allow fewer than `budget` sentences when extra ones
    # net-negative (utility < pairwise penalty they add). Non-empty floor.
    prob += lpSum(x) <= budget
    prob += lpSum(x) >= 1

    # McCormick linearization of y_ij = x_i AND x_j
    for (i, j), yij in y.items():
        prob += yij <= x[i]
        prob += yij <= x[j]
        prob += yij >= x[i] + x[j] - 1

    prob.solve(PULP_CBC_CMD(msg=0))
    selected = [i for i in range(M) if x[i].varValue is not None and x[i].varValue > 0.5]
    return selected


def ilp_select(
    unique_sentences,
    utility_scores,
    redundancy_matrix,
    budget,
    redundancy_threshold=None,
    utility_mode="legacy",
    tri_metric_weights=None,
    penalty_scale="per_edge",
):
    """用 ILP 在预算约束下选出 utility 最大且低冗余的句子子集。"""
    M = len(unique_sentences)
    if M <= budget:
        return list(range(M))

    if utility_mode == "tri_metric" and tri_metric_weights is not None:
        selected = _solve_soft_ilp(
            utility_scores,
            redundancy_matrix,
            budget,
            float(tri_metric_weights["redundancy"]),
            penalty_scale,
        )
        if not selected:
            return _fallback_topk(utility_scores, budget)
        return sorted(selected)

    threshold = redundancy_threshold if redundancy_threshold is not None else ILP_REDUNDANCY_THRESHOLD
    selected = _solve_hard_ilp(utility_scores, redundancy_matrix, budget, threshold)
    if len(selected) != budget:
        return _fallback_topk(utility_scores, budget)
    return sorted(selected)
