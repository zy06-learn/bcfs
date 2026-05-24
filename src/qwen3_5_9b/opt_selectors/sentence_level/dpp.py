"""
DPP-inspired greedy sentence selector.

Kernel L[i,j] = quality[i] * similarity[i,j] * quality[j]. The current
release uses a determinant-based greedy MAP-style objective, not exact DPP
sampling.
"""

import numpy as np

from opt_selectors.tri_metric import scale_pairwise_matrix


def dpp_select(
    unique_sentences,
    utility_scores,
    redundancy_matrix,
    budget,
    redundancy_weight=None,
    utility_mode="legacy",
    tri_metric_weights=None,
):
    """Use a greedy DPP-inspired determinant objective for quality-diversity selection."""
    M = len(unique_sentences)
    if M <= budget:
        return list(range(M))

    if utility_mode == "tri_metric" and redundancy_weight is None and tri_metric_weights is not None:
        redundancy_weight = tri_metric_weights["redundancy"]
    if redundancy_weight is None:
        redundancy_weight = 1.0

    quality = np.clip(np.array(utility_scores), 0.01, None)
    S = np.clip(np.array(scale_pairwise_matrix(redundancy_matrix, redundancy_weight)), 0, 1)

    Q = np.diag(quality)
    L = Q @ S @ Q + 1e-6 * np.eye(M)

    def greedy_map_dpp(L, k):
        M = L.shape[0]
        selected = []
        candidates = list(range(M))
        
        for _ in range(min(k, M)):
            best_idx = -1
            best_det = -float('inf')
            
            for i in candidates:
                subset = selected + [i]
                sub_L = L[np.ix_(subset, subset)]
                sign, log_det = np.linalg.slogdet(sub_L)
                val = log_det if sign > 0 else -float('inf')
                
                if val > best_det:
                    best_det = val
                    best_idx = i
                    
            if best_idx != -1:
                selected.append(best_idx)
                candidates.remove(best_idx)
            else:
                # Fallback to largest diagonal if determinant fails
                if len(selected) < k:
                    remaining = [(L[c, c], c) for c in candidates]
                    remaining.sort(reverse=True)
                    best_c = remaining[0][1]
                    selected.append(best_c)
                    candidates.remove(best_c)
                else:
                    break
                    
        return selected

    try:
        selected = sorted(greedy_map_dpp(L, budget))
    except Exception:
        selected = sorted(np.argsort(quality)[-budget:].tolist())

    return selected
