"""Helpers for writing experiment outputs to text files."""


DEFAULT_TABLE_METRIC_NAMES = ["rouge", "bertscore"]
DEFAULT_EXTRA_METRIC_NAMES = ["factcc", "minicheck", "alignscore", "factkb"]


def _available_rouge_keys(metrics):
    ordered = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    totals = metrics.get("rouge_totals") or {}
    rouge_scores = metrics.get("rouge_scores", {})
    return [key for key in ordered if key in totals or key in rouge_scores]


def _format_effective_weights(log):
    weights = log.get("effective_weights")
    if not weights:
        return ""

    labels = log.get("effective_weight_labels", {})
    ordered = []
    for key in ("rouge", "minicheck", "redundancy"):
        if key not in weights:
            continue
        label = labels.get(key, key)
        value = weights[key]
        if value is None:
            ordered.append(f"{label}=None")
        else:
            ordered.append(f"{label}={float(value):.4f}")
    return ", ".join(ordered)


def _format_reference_display(reference):
    if isinstance(reference, (list, tuple)):
        if not reference:
            return "[]"
        formatted = ["[Multi-reference set]"]
        for idx, reference_text in enumerate(reference, start=1):
            formatted.append(f"      Ref {idx}: {reference_text}")
        return "\n".join(formatted)

    return str(reference)


def _write_rouge_block(f, metrics, num_samples):
    totals = metrics.get("rouge_totals")
    rouge_impl = metrics.get("rouge_impl", "rouge_score")
    rouge_scores = metrics.get("rouge_scores", {})
    rouge_keys = _available_rouge_keys(metrics)
    if not rouge_keys:
        return

    if totals is None:
        f.write(f"ROUGE Scores (F1 only, impl={rouge_impl}):\n")
        for key in rouge_keys:
            f.write(f"  {key:8s}  F1={rouge_scores[key]:.2f}%\n")
        return

    f.write(f"ROUGE Scores (Precision / Recall / F1, impl={rouge_impl}):\n")
    for key in rouge_keys:
        precision = totals[key][0] / num_samples * 100
        recall = totals[key][1] / num_samples * 100
        f1 = totals[key][2] / num_samples * 100
        f.write(f"  {key:8s}  Precision={precision:.2f}%  Recall={recall:.2f}%  F1={f1:.2f}%\n")


def _write_named_metrics(f, metrics, metric_names, title, num_samples):
    if not metric_names:
        return

    metric_errors = metrics.get("metric_errors", {})
    f.write(f"\n{title}:\n")
    for metric_name in metric_names:
        if metric_name == "rouge":
            _write_rouge_block(f, metrics, num_samples)
        elif metric_name == "bertscore":
            if "bert_P" in metrics:
                f.write("BERTScore (roberta-large):\n")
                f.write(
                    f"  Precision={metrics['bert_P']:.2f}%  "
                    f"Recall={metrics['bert_R']:.2f}%  "
                    f"F1={metrics['bert_F']:.2f}%\n"
                )
            elif "bertscore" in metric_errors:
                f.write(f"BERTScore: unavailable ({metric_errors['bertscore']})\n")
        elif metric_name == "factcc":
            if "factcc" in metrics:
                f.write("FactCC (manueldeprada/FactCC):\n")
                f.write(f"  SentenceAvgCorrect: {metrics['factcc']:.2f}%\n")
            elif "factcc" in metric_errors:
                f.write(f"FactCC: unavailable ({metric_errors['factcc']})\n")
        elif metric_name == "minicheck":
            if "minicheck" in metrics:
                f.write("MiniCheck (EMNLP 2024, MiniCheck-RoBERTa-Large):\n")
                f.write(f"  SummaryAvgConsistent: {metrics['minicheck']:.2f}%\n")
            elif "minicheck" in metric_errors:
                f.write(f"MiniCheck: unavailable ({metric_errors['minicheck']})\n")
        elif metric_name == "alignscore":
            if "alignscore" in metrics:
                f.write("AlignScore (yzha/AlignScore-base, nli_sp):\n")
                f.write(f"  SummaryAvg: {metrics['alignscore']:.2f}%\n")
            elif "alignscore" in metric_errors:
                f.write(f"AlignScore: unavailable ({metric_errors['alignscore']})\n")
        elif metric_name == "factkb":
            if "factkb" in metrics:
                f.write("FactKB (bunsenfeng/FactKB):\n")
                f.write(f"  FactualProb: {metrics['factkb']:.2f}%\n")
            elif "factkb" in metric_errors:
                f.write(f"FactKB: unavailable ({metric_errors['factkb']})\n")


def _write_runtime_block(f, metrics):
    runtime = metrics.get("runtime") or {}
    metric_seconds = metrics.get("metric_seconds") or {}
    if not runtime and not metric_seconds:
        return

    f.write("\nRuntime:\n")
    if runtime.get("started_at"):
        f.write(f"  StartedAt: {runtime['started_at']}\n")
    if runtime.get("finished_at"):
        f.write(f"  FinishedAt: {runtime['finished_at']}\n")
    if "total_seconds" in runtime:
        f.write(f"  MethodTotalSeconds: {float(runtime['total_seconds']):.2f}\n")
    if "generation_seconds" in runtime:
        f.write(f"  GenerationSeconds: {float(runtime['generation_seconds']):.2f}\n")
    if "evaluation_seconds" in runtime:
        f.write(f"  EvaluationSeconds: {float(runtime['evaluation_seconds']):.2f}\n")
    if "resumed" in runtime:
        f.write(
            "  Resume: "
            f"resumed={runtime.get('resumed')}, "
            f"resumed_from={runtime.get('resumed_from', 0)}, "
            f"prior_generation_seconds={float(runtime.get('prior_generation_seconds', 0.0)):.2f}\n"
        )

    if metric_seconds:
        ordered = ["rouge", "bertscore", "factcc", "minicheck", "alignscore", "factkb"]
        seen = set()
        formatted = []
        for name in ordered:
            if name in metric_seconds:
                formatted.append(f"{name}={float(metric_seconds[name]):.2f}s")
                seen.add(name)
        for name in sorted(metric_seconds):
            if name in seen:
                continue
            formatted.append(f"{name}={float(metric_seconds[name]):.2f}s")
        if formatted:
            f.write(f"  MetricSeconds: {', '.join(formatted)}\n")

    co_timing = metrics.get("co_timing_seconds") or {}
    if co_timing:
        f.write("  COTimingSeconds:\n")
        f.write(f"    Samples: {int(co_timing.get('samples', 0))}\n")
        f.write(
            f"    Total: {float(co_timing.get('total', 0.0)):.2f}s "
            f"MeanPerSample: {float(co_timing.get('mean', 0.0)):.6f}s\n"
        )

def _write_pool_block(f, log):
    if "pool" not in log:
        return

    f.write(f"(b) Candidate pool ({len(log['pool'])} sentences):\n")
    utility_scores = log.get("utility_scores", [])
    sentence_costs = log.get("sentence_costs", [])
    sentence_positions = log.get("sentence_positions", [])
    for idx_s, sent in enumerate(log["pool"]):
        extras = []
        if utility_scores:
            extras.append(f"Utility={utility_scores[idx_s]:.4f}")
        if sentence_costs:
            extras.append(f"Cost={sentence_costs[idx_s]}")
        if sentence_positions:
            position = sentence_positions[idx_s]
            extras.append(
                "Doc="
                f"{position['document_id']} Pos={position['sentence_position']} Sid={position['sentence_id']}"
            )
        prefix = " | ".join(extras)
        if prefix:
            f.write(f"    [{idx_s}] {prefix} | {sent}\n")
        else:
            f.write(f"    [{idx_s}] {sent}\n")


def _write_redundancy_block(f, log):
    if "redundancy_shape" not in log:
        return

    f.write(f"(c) Redundancy matrix shape: {log['redundancy_shape']}\n")
    redundancy_matrix = log.get("redundancy_matrix")
    if not redundancy_matrix:
        return

    max_redundancy = 0.0
    max_pair = (0, 0)
    for left in range(len(redundancy_matrix)):
        for right in range(left + 1, len(redundancy_matrix)):
            if redundancy_matrix[left][right] > max_redundancy:
                max_redundancy = redundancy_matrix[left][right]
                max_pair = (left, right)
    f.write(
        "    Highest redundancy pair: "
        f"[{max_pair[0]}] vs [{max_pair[1]}], ROUGE-L F1={max_redundancy:.4f}\n"
    )


def _write_selection_block(f, log):
    if "effective_weights" in log:
        f.write(f"(d0) Effective weights: {_format_effective_weights(log)}\n")
    if "co_timing_seconds" in log:
        timing = log["co_timing_seconds"]
        f.write(
            "(d-timing) CO total seconds: "
            f"{float(timing.get('total', 0.0)):.6f}\n"
        )
    if "selected_indices" in log:
        f.write(f"(d) Selected indices: {log['selected_indices']}")
        if "utility_sum" in log:
            f.write(f", utility sum: {log['utility_sum']:.4f}")
        if "budget_used" in log:
            f.write(f", budget used: {log['budget_used']} {log.get('cost_unit', '')}".rstrip())
        if "objective_score" in log:
            f.write(f", objective: {log['objective_score']:.4f}")
        f.write("\n")

    if "ordering_method" in log:
        f.write(f"(d-order) Ordering method: {log['ordering_method']}\n")
        if "ordered_selected_indices" in log:
            f.write(f"    Ordered selected indices: {log['ordered_selected_indices']}\n")
        for match in log.get("ordering_matches", []):
            source_index = match.get("source_index")
            similarity = match.get("similarity")
            if source_index is None or similarity is None:
                continue
            f.write(
                f"    selected=[{match['selected_index']}] -> source_sent=[{source_index}] "
                f"ROUGE-L={float(similarity):.4f}\n"
            )
            f.write(f"      selected: {match.get('selected_sentence', '')}\n")
            f.write(f"      source:   {match.get('source_sentence', '')}\n")

    if "coverage_score" in log or "diversity_score" in log:
        f.write(
            "    Objective parts: "
            f"coverage={log.get('coverage_score', 0.0):.4f}, "
            f"diversity={log.get('diversity_score', 0.0):.4f}\n"
        )

    if "greedy_selected_indices" in log:
        f.write(f"    Greedy set before singleton check: {log['greedy_selected_indices']}\n")

    if "selection_mode" in log:
        f.write(f"    Final decision: {log['selection_mode']}\n")

    if "best_singleton_index" in log and log["best_singleton_index"] is not None:
        f.write(
            "    Best singleton: "
            f"[{log['best_singleton_index']}] value={log.get('best_singleton_value', 0.0):.4f}\n"
        )

    if "exact_summary_cost" in log:
        f.write(f"    Exact rendered summary cost: {log['exact_summary_cost']} {log.get('cost_unit', '')}\n")


def _write_trace_block(f, log):
    if "selection_trace" not in log:
        return

    f.write("(e) Modified greedy trace:\n")
    for step_id, step in enumerate(log["selection_trace"], start=1):
        status = "accept" if step["accepted"] else "skip"
        f.write(
            f"    step={step_id:02d} cand=[{step['candidate_index']}] {status} "
            f"gain={step['marginal_gain']:.4f} ratio={step['ratio']:.4f} "
            f"cost={step['sentence_cost']} budget={step['cumulative_budget']} "
            f"objective={step['current_objective']:.4f}"
        )
        if "current_coverage" in step or "current_diversity" in step:
            f.write(
                f" coverage={step.get('current_coverage', 0.0):.4f}"
                f" diversity={step.get('current_diversity', 0.0):.4f}"
            )
        f.write("\n")


def _write_top_candidates_block(f, log):
    if "top_candidates" not in log:
        return

    f.write("(f) Top candidate summaries:\n")
    for cand in log["top_candidates"]:
        if "consensus" in cand:
            f.write(
                f"    subset={cand['subset']} | consensus={cand['consensus']:.4f} | "
                f"minicheck={cand['minicheck']:.4f}"
            )
            if "redundancy" in cand:
                f.write(f" | redundancy={cand['redundancy']:.4f}")
            f.write(f" | total={cand['total']:.4f}\n")
        elif "coverage" in cand:
            f.write(
                f"    subset={cand['subset']} | minicheck={cand['minicheck']:.4f} | "
                f"coverage={cand['coverage']:.4f} | redundancy={cand['redundancy']:.4f}"
            )
            if "weighted_total" in cand:
                f.write(f" | total={cand['weighted_total']:.4f}")
            f.write("\n")
        f.write(f"      {cand.get('summary', '')}\n")


def _write_similarity_pairs_block(f, log):
    if "similarity_top_pairs" not in log:
        return

    f.write("(g) Top similarity pairs:\n")
    for pair in log["similarity_top_pairs"]:
        f.write(
            f"    [{pair['left']}] vs [{pair['right']}] score={pair['score']:.4f}\n"
            f"      L: {pair['left_text']}\n"
            f"      R: {pair['right_text']}\n"
        )


def _write_cluster_block(f, log):
    if "cluster_summaries" not in log:
        return

    f.write("(g) Cluster summaries:\n")
    f.write(
        f"    cluster_count={log.get('cluster_count', len(log['cluster_summaries']))}, "
        f"alpha={log.get('alpha', 'n/a')}, lambda={log.get('lambda_weight', 'n/a')}, "
        f"r={log.get('r', 'n/a')}\n"
    )
    f.write(
        f"    use_bigrams={log.get('use_bigrams', 'n/a')}, "
        f"use_stemming={log.get('use_stemming', 'n/a')}, "
        f"cluster_ratio={log.get('cluster_ratio', 'n/a')}\n"
    )
    for cluster in log["cluster_summaries"]:
        f.write(
            f"    cluster={cluster['cluster_id']} size={cluster['size']} "
            f"members={cluster['members']}\n"
        )

    if "singleton_rewards" in log:
        rewards = ", ".join(f"[{idx}]={value:.4f}" for idx, value in enumerate(log["singleton_rewards"]))
        f.write(f"    singleton_rewards: {rewards}\n")


def _save_extended_results(result_file, metrics, generated_summaries, references, config_header, all_sample_logs=None):
    num_samples = len(generated_summaries)
    table_metric_names = metrics.get("table_metric_names")
    extra_metric_names = metrics.get("extra_metric_names")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(config_header)
        if metrics.get("sentence_split_for_rouge"):
            f.write(f"ROUGE sentence split: {metrics['sentence_split_for_rouge']}\n")
        f.write(f"{'=' * 60}\n\n")

        _write_named_metrics(f, metrics, table_metric_names or ["rouge"], "Table Metrics", num_samples)
        _write_named_metrics(f, metrics, extra_metric_names or [], "Extra Metrics", num_samples)
        _write_runtime_block(f, metrics)

        if all_sample_logs:
            f.write(f"\n{'=' * 60}\n")
            f.write("Detailed selection logs (first 10 samples):\n")
            f.write(f"{'=' * 60}\n\n")

            for j in range(min(10, num_samples)):
                log = all_sample_logs[j]
                f.write(f"--- Sample {j + 1} ---\n")

                if "optimization_method" in log:
                    f.write(f"(a) Optimization method: {log['optimization_method']}\n")

                _write_pool_block(f, log)
                _write_redundancy_block(f, log)
                _write_selection_block(f, log)

                if "selected_consensus" in log:
                    f.write(
                        "(d) Selected: "
                        f"consensus={log['selected_consensus']:.4f}, "
                        f"minicheck={log['selected_minicheck']:.4f}, "
                    )
                    if "selected_redundancy" in log:
                        f.write(f"redundancy={log['selected_redundancy']:.4f}, ")
                    f.write(f"total={log['selected_total']:.4f}\n")

                _write_trace_block(f, log)
                _write_top_candidates_block(f, log)
                _write_similarity_pairs_block(f, log)
                _write_cluster_block(f, log)

                f.write(f"(h) Final summary: {generated_summaries[j]}\n")
                f.write(f"    Reference: {_format_reference_display(references[j])}\n\n")

    print(f"\nSaved results to: {result_file}")


def save_results(result_file, metrics, generated_summaries, references, config_header, all_sample_logs=None):
    """Persist aggregate metrics and a small sample of selection logs."""
    table_metric_names = metrics.get("table_metric_names")
    extra_metric_names = metrics.get("extra_metric_names")

    if table_metric_names is None and extra_metric_names is None:
        metrics = dict(metrics)
        metrics["table_metric_names"] = list(DEFAULT_TABLE_METRIC_NAMES)
        metrics["extra_metric_names"] = list(DEFAULT_EXTRA_METRIC_NAMES)

    _save_extended_results(result_file, metrics, generated_summaries, references, config_header, all_sample_logs)
