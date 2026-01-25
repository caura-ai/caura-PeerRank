"""
phases/phase4.py - Report Generation
"""

from datetime import datetime
from statistics import mean, stdev
from scipy import stats

from peerrank.config import (
    MODELS, ALL_MODELS, DATA_DIR, load_json, format_duration, get_revision,
    format_table, match_model_name, calculate_scores_from_evaluations,
    calculate_judge_agreement, calculate_question_stats,
    calculate_elo_ratings, ELO_K_FACTOR,
    _pearson_correlation, _spearman_correlation,
)


def _calculate_scores_for_mode(evaluations: dict) -> dict:
    """Calculate peer scores (excluding self) for a single evaluation mode."""
    scores = calculate_scores_from_evaluations(evaluations)
    return {n: mean(s) if s else 0 for n, s in scores["peer_scores"].items()}


def _calculate_bias_analysis(phase3_data: dict) -> dict | None:
    """Calculate bias analysis comparing scores across 3 modes."""
    evaluations_by_mode = phase3_data.get("evaluations_by_mode")
    if not evaluations_by_mode:
        return None

    display_names = [n for _, _, n in MODELS]
    modes = ["shuffle_only", "blind_only", "shuffle_blind"]

    # Calculate peer scores for each mode
    scores_by_mode = {}
    for mode in modes:
        if mode in evaluations_by_mode:
            scores_by_mode[mode] = _calculate_scores_for_mode(evaluations_by_mode[mode])

    if len(scores_by_mode) < 3:
        return None

    # Build analysis data
    # shuffle_only: names visible, order random -> shows name effect
    # blind_only: names hidden, order fixed -> shows position effect
    # shuffle_blind: both protections -> baseline (Peer score)
    #
    # UNIFIED BIAS CONVENTION: Positive = factor HELPED the model
    # Name Bias     = shuffle_only − shuffle_blind  (effect of showing name)
    # Position Bias = blind_only − shuffle_blind    (effect of fixed position)
    analysis = []
    for name in display_names:
        shuffle_only = scores_by_mode["shuffle_only"].get(name, 0)
        blind_only = scores_by_mode["blind_only"].get(name, 0)
        shuffle_blind = scores_by_mode["shuffle_blind"].get(name, 0)

        # Name Bias: How much did showing the model name help?
        # Positive = name recognition boosted score
        name_bias = shuffle_only - shuffle_blind

        # Position Bias: How much did the fixed position help?
        # Positive = being in that position boosted score
        position_bias = blind_only - shuffle_blind

        analysis.append({
            "model": name,
            "shuffle_only": shuffle_only,
            "blind_only": blind_only,
            "shuffle_blind": shuffle_blind,
            "name_bias": name_bias,
            "position_bias": position_bias,
        })

    # Sort by shuffle_blind score (the most protected baseline)
    analysis.sort(key=lambda x: x["shuffle_blind"], reverse=True)
    return analysis


def _load_tfq_validation() -> dict | None:
    """Load TruthfulQA validation data: truth scores and baseline correlation.

    Returns dict with:
      - truth_scores: {model: score} mapping
      - baseline_r: Pearson r from TFQ validation (peer vs truth)
      - baseline_rho: Spearman rho from TFQ validation
    Or None if TFQ data not available.
    """
    import os
    import re

    # Load truth scores
    tfq_scores_path = os.path.join(DATA_DIR, "TRUTH", "phase4_TFQ_scores_TFQ.json")
    tfq_report_path = os.path.join(DATA_DIR, "TRUTH", "TFQ_validation_report_TFQ.md")

    if not os.path.exists(tfq_scores_path):
        return None

    try:
        import json
        with open(tfq_scores_path, "r", encoding="utf-8") as f:
            tfq_data = json.load(f)

        summary = tfq_data.get("summary", {})
        if not summary:
            return None

        # Extract truth scores
        truth_scores = {}
        for model, data in summary.items():
            if "mean" in data:
                truth_scores[model] = data["mean"]
            elif "accuracy" in data:
                truth_scores[model] = data["accuracy"] / 10.0

        if not truth_scores:
            return None

        result = {"truth_scores": truth_scores}

        # Try to load baseline correlation from TFQ validation report
        if os.path.exists(tfq_report_path):
            with open(tfq_report_path, "r", encoding="utf-8") as f:
                report = f.read()

            # Parse Pearson r and Spearman rho from report
            # Format: "Pearson r    0.8584    0.0004   strong"
            r_match = re.search(r"Pearson\s+r\s+([\d.]+)", report)
            rho_match = re.search(r"Spearman\s+([\d.]+)", report)

            if r_match:
                result["baseline_r"] = float(r_match.group(1))
            if rho_match:
                result["baseline_rho"] = float(rho_match.group(1))

        return result
    except Exception:
        return None


def _calculate_ablation_study(bias_analysis: list, peer_data: list) -> dict | None:
    """Calculate ablation study: uncorrected scores vs bias-corrected scores.

    No Correction = Peer + Name Bias + Position Bias
    This shows what rankings would look like without any bias correction.

    Also computes correlation with TruthfulQA ground truth if available,
    showing that bias correction improves alignment with objective accuracy.
    """
    if not bias_analysis or not peer_data:
        return None

    # Build peer score lookup
    peer_lookup = {name: avg for name, avg, *_ in peer_data}

    # Calculate uncorrected scores
    results = []
    for ba in bias_analysis:
        name = ba["model"]
        peer = peer_lookup.get(name, ba["shuffle_blind"])
        # No Correction = Peer + Name Bias + Position Bias
        no_correction = peer + ba["name_bias"] + ba["position_bias"]
        results.append({
            "model": name,
            "peer": peer,
            "no_correction": no_correction,
            "name_bias": ba["name_bias"],
            "position_bias": ba["position_bias"],
            "total_bias": ba["name_bias"] + ba["position_bias"],
        })

    # Sort by no_correction (uncorrected ranking)
    results.sort(key=lambda x: x["no_correction"], reverse=True)

    # Add ranks
    peer_ranked = sorted(results, key=lambda x: x["peer"], reverse=True)
    peer_rank_lookup = {r["model"]: i for i, r in enumerate(peer_ranked, 1)}

    for i, r in enumerate(results, 1):
        r["nc_rank"] = i
        r["peer_rank"] = peer_rank_lookup[r["model"]]
        r["rank_change"] = r["peer_rank"] - r["nc_rank"]  # positive = correction helped

    ablation_result = {"results": results}

    # Try to load TFQ validation data for ground truth correlation
    tfq_data = _load_tfq_validation()
    if tfq_data:
        tfq_truth = tfq_data["truth_scores"]

        # Match models and collect scores
        matched_models = []
        peer_scores = []
        nc_scores = []
        truth_scores = []

        for r in results:
            model = r["model"]
            truth = tfq_truth.get(model)
            if truth is None:
                for tfq_model, tfq_score in tfq_truth.items():
                    if model in tfq_model or tfq_model in model:
                        truth = tfq_score
                        break

            if truth is not None:
                matched_models.append(model)
                peer_scores.append(r["peer"])
                nc_scores.append(r["no_correction"])
                truth_scores.append(truth)

        # Compute correlations if we have enough matched models
        if len(matched_models) >= 5:
            # Bias-corrected (Peer) vs TFQ Truth
            peer_pearson = _pearson_correlation(peer_scores, truth_scores)
            peer_spearman = _spearman_correlation(peer_scores, truth_scores)

            # No correction vs TFQ Truth
            nc_pearson = _pearson_correlation(nc_scores, truth_scores)
            nc_spearman = _spearman_correlation(nc_scores, truth_scores)

            # Calculate p-values using scipy
            from scipy.stats import pearsonr, spearmanr
            _, peer_p_pearson = pearsonr(peer_scores, truth_scores)
            _, peer_p_spearman = spearmanr(peer_scores, truth_scores)
            _, nc_p_pearson = pearsonr(nc_scores, truth_scores)
            _, nc_p_spearman = spearmanr(nc_scores, truth_scores)

            ablation_result["tfq_validation"] = {
                "n_models": len(matched_models),
                "models": matched_models,
                # TFQ baseline (within-dataset, full bias correction)
                "tfq_baseline": {
                    "pearson_r": tfq_data.get("baseline_r"),
                    "spearman_rho": tfq_data.get("baseline_rho"),
                    "description": "TFQ Peer vs TFQ Truth (within-dataset)",
                },
                # Cross-dataset correlations
                "bias_corrected": {
                    "pearson_r": peer_pearson,
                    "spearman_rho": peer_spearman,
                    "p_pearson": peer_p_pearson,
                    "p_spearman": peer_p_spearman,
                    "description": "V6 Peer vs TFQ Truth (cross-dataset)",
                },
                "no_correction": {
                    "pearson_r": nc_pearson,
                    "spearman_rho": nc_spearman,
                    "p_pearson": nc_p_pearson,
                    "p_spearman": nc_p_spearman,
                    "description": "V6 No Correction vs TFQ Truth (cross-dataset)",
                },
                "improvement": {
                    "pearson_delta": peer_pearson - nc_pearson,
                    "spearman_delta": peer_spearman - nc_spearman,
                },
            }

    return ablation_result


def _calculate_stats(phase2_data: dict, phase3_data: dict) -> dict:
    """Calculate statistics from evaluation data."""
    evaluations = phase3_data["evaluations"]

    # Use shared calculation function
    scores = calculate_scores_from_evaluations(evaluations)
    peer_scores = scores["peer_scores"]
    self_scores = scores["self_scores"]
    raw_scores = scores["raw_scores"]
    judge_given = scores["judge_given"]

    timing_stats = phase2_data.get("timing_stats", {})
    eval_timing_stats = phase3_data.get("timing_stats", {})

    # Build peer_data tuples
    peer_data = []
    for _, _, name in MODELS:
        if peer_scores[name]:
            peer_avg = mean(peer_scores[name])
            self_avg = mean(self_scores[name]) if self_scores[name] else 0
            raw_avg = mean(raw_scores[name]) if raw_scores[name] else 0
            peer_data.append((
                name, peer_avg,
                stdev(peer_scores[name]) if len(peer_scores[name]) > 1 else 0,
                len(peer_scores[name]),
                timing_stats.get(name, {}).get("avg", 0),
                self_avg, self_avg - peer_avg, raw_avg
            ))
    peer_data.sort(key=lambda x: x[1], reverse=True)

    # Build judge_data tuples
    judge_data = []
    for _, _, name in MODELS:
        if judge_given[name]:
            judge_data.append((
                name, mean(judge_given[name]),
                stdev(judge_given[name]) if len(judge_given[name]) > 1 else 0,
                len(judge_given[name])
            ))
    judge_data.sort(key=lambda x: x[1], reverse=True)

    return {
        "questions": [q["text"] for q in phase2_data["questions"]],
        "peer_data": peer_data,
        "judge_data": judge_data,
        "timing_stats": timing_stats,
        "eval_timing_stats": eval_timing_stats,
    }


# ============================================================================
# Home Advantage Analysis - Statistical functions
# ============================================================================

def _cohens_d(group1: list, group2: list) -> float | None:
    """Calculate Cohen's d effect size."""
    import math
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return None
    mean1, mean2 = sum(group1) / n1, sum(group2) / n2
    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std else None


def _welch_ttest(group1: list, group2: list) -> tuple:
    """Welch's t-test (unequal variances). Returns (t_stat, p_value)."""
    import math
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return None, None
    mean1, mean2 = sum(group1) / n1, sum(group2) / n2
    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return None, None
    t_stat = (mean1 - mean2) / se
    # Welch-Satterthwaite degrees of freedom
    num = (var1 / n1 + var2 / n2) ** 2
    denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom else 100
    # Approximate p-value (normal approximation for large df)
    z = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return t_stat, p_value


def _calculate_provider_clustering(evaluations: dict) -> dict | None:
    """Compute Kruskal-Wallis test for peer scores grouped by provider."""
    # Provider mapping from model names
    PROVIDER_MAP = {
        'gpt-5.2': 'OpenAI', 'gpt-5-mini': 'OpenAI',
        'claude-opus-4-5': 'Anthropic', 'claude-sonnet-4-5': 'Anthropic',
        'gemini-3-pro-preview': 'Google', 'gemini-3-flash-thinking': 'Google',
        'gemini-3-flash-preview': 'Google', 'gemini-2.5-pro': 'Google', 'gemini-2.5-flash': 'Google',
        'grok-4-1-fast': 'xAI',
        'deepseek-chat': 'DeepSeek',
        'llama-4-maverick': 'Meta',
        'sonar-pro': 'Perplexity',
        'kimi-k2-0905': 'Moonshot',
        'mistral-large': 'Mistral',
    }

    models = list(evaluations.keys())
    scores = calculate_scores_from_evaluations(evaluations)

    # Group peer scores by provider
    provider_scores = {}
    for model in models:
        provider = PROVIDER_MAP.get(model, 'Other')
        peer = scores['peer_scores'].get(model, [])
        if peer:
            if provider not in provider_scores:
                provider_scores[provider] = []
            provider_scores[provider].extend(peer)

    # Need at least 2 providers with data
    groups = [v for v in provider_scores.values() if len(v) >= 2]
    if len(groups) < 2:
        return None

    # Kruskal-Wallis H-test (non-parametric ANOVA)
    h_stat, p_value = stats.kruskal(*groups)

    # Effect size: eta-squared approximation = H / (N - 1)
    n_total = sum(len(g) for g in groups)
    eta_sq = h_stat / (n_total - 1) if n_total > 1 else 0

    return {
        "n_providers": len(groups),
        "n_total": n_total,
        "h_stat": h_stat,
        "p_value": p_value,
        "eta_sq": eta_sq,
        "provider_scores": {k: (len(v), mean(v)) for k, v in provider_scores.items()},
    }


def _calculate_home_advantage(phase1_data: dict, evaluations: dict) -> dict | None:
    """Calculate home advantage analysis - do models do better on their own questions?"""
    # Build question -> source model mapping
    question_source = {}
    for model, questions in phase1_data.get("questions_by_model", {}).items():
        for q in questions:
            question_source[q["question"]] = model

    if not question_source or not evaluations:
        return None

    # Collect raw scores by source
    raw_scores = {}  # raw_scores[answering_model][source_model] = [scores]
    all_models = set()
    all_sources = set()

    for evaluator, evals in evaluations.items():
        for question, model_scores in evals.items():
            source = question_source.get(question)
            if not source:
                continue
            all_sources.add(source)
            for answering_model, data in model_scores.items():
                if evaluator == answering_model:
                    continue  # Exclude self-evaluations
                all_models.add(answering_model)
                if answering_model not in raw_scores:
                    raw_scores[answering_model] = {}
                if source not in raw_scores[answering_model]:
                    raw_scores[answering_model][source] = []
                # Only append numeric scores (skip malformed responses)
                score = data.get("score") if isinstance(data, dict) else data
                if isinstance(score, (int, float)):
                    raw_scores[answering_model][source].append(score)

    models = sorted(all_models)
    sources = sorted(all_sources)

    # Calculate home advantage for each model
    results = []
    for model in models:
        if model not in sources or model not in raw_scores:
            continue
        own_scores = raw_scores[model].get(model, [])
        if len(own_scores) < 3:
            continue
        other_scores = []
        for src in sources:
            if src != model and src in raw_scores[model]:
                other_scores.extend(raw_scores[model][src])
        if len(other_scores) < 3:
            continue

        own_avg = sum(own_scores) / len(own_scores)
        other_avg = sum(other_scores) / len(other_scores)
        diff = own_avg - other_avg
        t_stat, p_value = _welch_ttest(own_scores, other_scores)
        d = _cohens_d(own_scores, other_scores)

        results.append({
            "model": model, "own_avg": own_avg, "other_avg": other_avg,
            "diff": diff, "n_own": len(own_scores), "n_other": len(other_scores),
            "t_stat": t_stat, "p_value": p_value, "cohens_d": d,
        })

    results.sort(key=lambda x: x["diff"], reverse=True)

    # Calculate question difficulty by source
    source_difficulty = {}
    for src in sources:
        scores = []
        for model in models:
            if model in raw_scores and src in raw_scores[model]:
                scores.extend(raw_scores[model][src])
        if scores:
            source_difficulty[src] = sum(scores) / len(scores)

    return {"results": results, "source_difficulty": source_difficulty, "models": models, "sources": sources}


def phase4_generate_report() -> str:
    """Phase 4: Generate markdown report from evaluation data."""
    print(f"\n{'=' * 60}")
    print(f"  PHASE 4: Report Generation")
    print(f"{'-' * 60}")
    print(f"  Revision:    {get_revision()}")
    print(f"{'=' * 60}")

    phase1 = load_json("phase1_questions.json")
    phase2 = load_json("phase2_answers.json")
    phase3 = load_json("phase3_rankings.json")
    stats = _calculate_stats(phase2, phase3)

    revision = get_revision()
    web_search_enabled = phase2.get("web_search", True)
    web_search_status = "ON" if web_search_enabled else "OFF"
    r = [f"# PeerRank.ai LLM Evaluation Report\n\nRevision: **{revision}** | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
         f"\nModels evaluated: {len(MODELS)} | Questions: {len(stats['questions'])} | Web search: **{web_search_status}**"]

    # Model Order (fixed initial order used in blind_only mode)
    # Web mode by provider: how each model gets web search capability
    web_mode_by_provider = {
        "openai": "Tool",      # Responses API web_search tool
        "anthropic": "Tool",   # web-search-2025-03-05 beta
        "google": "Tool",      # google_search tool
        "grok": "Tool",        # xAI SDK web_search + x_search
        "mistral": "Agents",   # Native Agents API web_search connector
        "deepseek": "Tavily",  # Tavily search augmentation
        "together": "Tavily",  # Tavily search augmentation
        "perplexity": "Native", # Built-in to Sonar models
        "kimi": "Tavily",      # Tavily search augmentation
    }
    model_order = [f"{i}. {n} ({web_mode_by_provider.get(p, '?')})" for i, (p, _, n) in enumerate(ALL_MODELS, 1)]
    r.append(f"\n## Model Order and Web Mode\n\n" + "\n".join(model_order))

    # Timing
    mode_durations = phase3.get("mode_durations", {})
    timing_rows = [
        ["Phase 1 (Questions)", format_duration(phase1.get("duration_seconds", 0))],
        ["Phase 2 (Answers)", format_duration(phase2.get("duration_seconds", 0))],
        ["Phase 3 (Evaluation)", format_duration(phase3.get("duration_seconds", 0))],
    ]
    if mode_durations:
        timing_rows.append(["  └─ Shuffle Only", format_duration(mode_durations.get("shuffle_only", 0))])
        timing_rows.append(["  └─ Blind Only", format_duration(mode_durations.get("blind_only", 0))])
        timing_rows.append(["  └─ Shuffle + Blind", format_duration(mode_durations.get("shuffle_blind", 0))])
    timing_rows.append(["Phase 4 (Report)", "—"])
    r.append("\n## Phase Total Runtime\n" + format_table(["Phase", "Duration"], timing_rows, ['l', 'r']))

    # Question Analysis
    questions_by_model = phase1.get("questions_by_model", {})

    # Category counts and coverage matrix with totals
    category_counts = {}
    for model_qs in questions_by_model.values():
        for q in model_qs:
            cat = q.get("category", "unknown") if isinstance(q, dict) else "unknown"
            category_counts[cat] = category_counts.get(cat, 0) + 1

    all_categories = sorted(category_counts.keys())
    cat_short = [c.split()[0][:8] for c in all_categories]  # Short category names
    model_cat_counts = {}
    for model, qs in questions_by_model.items():
        model_cat_counts[model] = {}
        for q in qs:
            cat = q.get("category", "unknown") if isinstance(q, dict) else "unknown"
            model_cat_counts[model][cat] = model_cat_counts[model].get(cat, 0) + 1

    matrix_rows = []
    col_totals = [0] * len(all_categories)
    grand_total = 0
    for model in questions_by_model.keys():
        cat_values = [model_cat_counts.get(model, {}).get(cat, 0) for cat in all_categories]
        total = sum(cat_values)
        grand_total += total
        for i, v in enumerate(cat_values):
            col_totals[i] += v
        row = [model] + [str(v) if v else "-" for v in cat_values] + [str(total)]
        matrix_rows.append(row)
    # Add totals row
    matrix_rows.append(["**Total**"] + [str(t) for t in col_totals] + [str(grand_total)])

    r.append("\n## Question Analysis\n" + format_table(["Model"] + cat_short + ["Total"], matrix_rows, ['l'] + ['c'] * len(cat_short) + ['r']))

    # Response time (Phase 2 - answering)
    ts = stats["timing_stats"]
    answer_rows = []
    failed_models = []
    for _, _, n in MODELS:
        stats_n = ts.get(n, {})
        avg = stats_n.get('avg', 0)
        successes = stats_n.get('successes', 0)
        count = stats_n.get('count', 0)
        answer_rows.append([n, f"{avg:.2f}s", f"{successes}/{count}"])
        if successes == 0 and count > 0:
            failed_models.append(n)
    answer_rows.sort(key=lambda x: float(x[1].replace("s", "")))
    r.append("\n## Answers\n" + format_table(["Model", "Avg Time", "OK/Total"], answer_rows, ['l', 'r', 'r']))

    # Warning for models with no successful answers
    if failed_models:
        r.append(f"\n**Warning:** {', '.join(failed_models)} failed to answer any questions. Peer scores for these models may be invalid (scored on missing/error responses).")

    # Evaluation time (Phase 3 - judging)
    ets = stats["eval_timing_stats"]
    if ets:
        eval_rows = []
        for _, _, n in MODELS:
            stats_n = ets.get(n, {})
            avg = stats_n.get('avg', 0)
            successes = stats_n.get('successes', 0)
            count = stats_n.get('count', 0)
            eval_rows.append([n, f"{avg:.2f}s", f"{successes}/{count}"])
        eval_rows.sort(key=lambda x: float(x[1].replace("s", "")))
        r.append("\n## Evaluations\n" + format_table(["Model", "Avg Time", "OK/Total"], eval_rows, ['l', 'r', 'r']))

    # Peer rankings (scores only, bias moved to consolidated section)
    peer_rows = [[str(i), n, f"{avg:.2f}", f"{std:.2f}", f"{raw:.2f}"]
                 for i, (n, avg, std, _, _, s, b, raw) in enumerate(stats["peer_data"], 1)]
    r.append("\n## Final Peer Rankings (Shuffle + Blind mode)\n\nScores from peer evaluations (excluding self-ratings):\n" +
             format_table(["#", "Model", "Peer Score", "Std", "Raw"], peer_rows, ['c', 'l', 'r', 'r', 'r']))

    # Elo Ratings section
    evaluations_by_mode = phase3.get("evaluations_by_mode", {})
    elo_evaluations = evaluations_by_mode.get("shuffle_blind", phase3.get("evaluations", {}))

    if elo_evaluations:
        model_names = [n for _, _, n in MODELS]
        elo_data = calculate_elo_ratings(elo_evaluations, model_names)

        peer_score_lookup = {n: avg for n, avg, _, _, _, _, _, _ in stats["peer_data"]}
        peer_rank_lookup = {n: i for i, (n, _, _, _, _, _, _, _) in enumerate(stats["peer_data"], 1)}
        elo_sorted = sorted(elo_data["ratings"].items(), key=lambda x: x[1], reverse=True)

        r.append(f"\n## Elo Ratings\n\nPairwise comparisons from evaluation scores (K={ELO_K_FACTOR}).\n"
                 f"Total matches: {elo_data['total_matches']:,}\n")

        elo_rows = []
        for elo_rank, (name, elo) in enumerate(elo_sorted, 1):
            wins, losses, ties = elo_data["matches"].get(name, (0, 0, 0))
            win_rate = elo_data["win_rates"].get(name, 0)
            peer_score = peer_score_lookup.get(name, 0)
            peer_rank = peer_rank_lookup.get(name, 0)
            rank_diff = peer_rank - elo_rank

            elo_rows.append([
                str(elo_rank), name, str(elo), f"{win_rate:.1f}%",
                f"{wins}-{losses}-{ties}", f"{peer_score:.2f}", str(peer_rank),
                f"{rank_diff:+d}" if rank_diff != 0 else "—",
            ])

        r.append(format_table(
            ["#", "Model", "Elo", "Win%", "W-L-T", "Peer", "P#", "Diff"],
            elo_rows, ['c', 'l', 'r', 'r', 'c', 'r', 'c', 'c']
        ))

        peer_scores_list = [peer_score_lookup.get(name, 0) for name, _ in elo_sorted]
        elo_ratings_list = [elo for _, elo in elo_sorted]
        pearson_r = _pearson_correlation(peer_scores_list, elo_ratings_list)
        spearman_r = _spearman_correlation(peer_scores_list, elo_ratings_list)

        r.append(f"\nCorrelation with Peer Scores: **r={pearson_r:.3f}** (Pearson), **ρ={spearman_r:.3f}** (Spearman)")
        r.append("\n*Diff = Peer rank − Elo rank (positive = Elo ranks model higher)*")

    # Consolidated Bias Analysis (all 3 bias types)
    bias_analysis = _calculate_bias_analysis(phase3)

    # Build self-bias lookup from peer_data
    self_bias_lookup = {n: b for n, _, _, _, _, _, b, _ in stats["peer_data"]}
    self_score_lookup = {n: s for n, _, _, _, _, s, _, _ in stats["peer_data"]}

    r.append("\n## Bias Analysis\n")
    r.append("Three types of bias detected in the evaluation process:\n")
    r.append("| Bias Type | Cause | Interpretation |")
    r.append("|-----------|-------|----------------|")
    r.append("| **Self Bias** | Evaluator rates own answers | + overrates self, − underrates self |")
    r.append("| **Name Bias** | Brand/model recognition | + name helped, − name hurt |")
    r.append("| **Position Bias** | Fixed order in answer list | + position helped, − position hurt |\n")

    if bias_analysis:
        # Table 1: Position Bias (by position, not by model)
        # Get position order from MODELS (the active/filtered list used in phase3 blind mode)
        active_model_order = [n for _, _, n in MODELS]
        pos_rows = []
        for pos, model_name in enumerate(active_model_order, 1):
            ba = next((d for d in bias_analysis if d["model"] == model_name), None)
            if ba:
                pos_rows.append([
                    str(pos),
                    f"{ba['blind_only']:.2f}",
                    f"{ba['position_bias']:+.2f}",
                ])

        r.append("### Position Bias\n")
        r.append("Effect of answer position in fixed-order (blind) evaluation:\n")
        r.append(format_table(
            ["Pos", "Blind Score", "Pos Bias"],
            pos_rows,
            ['c', 'r', 'r']
        ))
        r.append("\n*Pos Bias = Blind − Peer (positive = position helped)*\n")

        # Table 2: Model Analysis (self bias and name bias)
        r.append("### Model Bias\n")
        r.append("Self-favoritism and brand recognition effects:\n")
        model_rows = []
        for i, (name, peer_avg, _, _, _, self_avg, self_bias, _) in enumerate(stats["peer_data"], 1):
            ba = next((d for d in bias_analysis if d["model"] == name), None)
            if ba:
                model_rows.append([
                    str(i),
                    name,
                    f"{peer_avg:.2f}",
                    f"{self_score_lookup.get(name, 0):.2f}",
                    f"{self_bias_lookup.get(name, 0):+.2f}",
                    f"{ba['shuffle_only']:.2f}",
                    f"{ba['name_bias']:+.2f}",
                ])

        r.append(format_table(
            ["#", "Model", "Peer", "Self", "Self Bias", "Shuffle", "Name Bias"],
            model_rows,
            ['c', 'l', 'r', 'r', 'r', 'r', 'r']
        ))
        r.append("\n*Self Bias = Self − Peer (+ overrates self) | Name Bias = Shuffle − Peer (+ name helped)*")

    # Ablation Study: Effect of Bias Correction
    ablation = _calculate_ablation_study(bias_analysis, stats["peer_data"])
    if ablation:
        r.append("\n## Ablation Study: Effect of Bias Correction\n")
        r.append("Comparison of rankings with and without bias correction.\n")
        r.append("**No Correction** = Peer + Name Bias + Position Bias (raw scores with all biases present)\n")

        abl_rows = []
        for res in ablation["results"]:
            rank_str = f"{res['rank_change']:+d}" if res["rank_change"] != 0 else "—"
            abl_rows.append([
                str(res["nc_rank"]),
                res["model"],
                f"{res['no_correction']:.2f}",
                f"{res['name_bias']:+.2f}",
                f"{res['position_bias']:+.2f}",
                f"{res['total_bias']:+.2f}",
                f"{res['peer']:.2f}",
                str(res["peer_rank"]),
                rank_str,
            ])

        r.append(format_table(
            ["#", "Model", "No Corr", "+Name", "+Pos", "=Bias", "Peer", "P#", "Δ"],
            abl_rows,
            ['c', 'l', 'r', 'r', 'r', 'r', 'r', 'c', 'c']
        ))
        r.append("\n*Δ = Peer rank − Uncorrected rank (positive = bias correction helped this model)*\n")

        # Calculate summary statistics
        rank_changes = [abs(res["rank_change"]) for res in ablation["results"]]
        models_changed = sum(1 for rc in rank_changes if rc > 0)
        max_change = max(rank_changes)
        total_biases = [res["total_bias"] for res in ablation["results"]]
        avg_bias = sum(total_biases) / len(total_biases)
        max_pos_bias = max(total_biases)
        max_neg_bias = min(total_biases)

        r.append(f"**Summary:** {models_changed}/{len(ablation['results'])} models changed rank after correction. "
                 f"Max rank change: {max_change}. Avg total bias: {avg_bias:+.2f} "
                 f"(range: {max_neg_bias:+.2f} to {max_pos_bias:+.2f}).\n")

        # TFQ Validation (correlation with ground truth)
        tfq_val = ablation.get("tfq_validation")
        if tfq_val:
            r.append("### Ground Truth Validation (TruthfulQA)\n")
            r.append("Correlation between PeerRank scores and TruthfulQA ground truth accuracy.\n")
            r.append("Removing bias correction reduces alignment with ground truth:\n")

            baseline = tfq_val.get("tfq_baseline", {})
            nc = tfq_val["no_correction"]

            baseline_r = baseline.get("pearson_r")
            baseline_rho = baseline.get("spearman_rho")
            nc_r = nc["pearson_r"]
            nc_rho = nc["spearman_rho"]

            if baseline_r:
                delta_r = baseline_r - nc_r
                delta_rho = baseline_rho - nc_rho if baseline_rho else 0

                tfq_rows = [
                    ["Pearson *r*", f"{baseline_r:.3f}", f"{nc_r:.3f}", f"{delta_r:+.3f}"],
                    ["Spearman *ρ*", f"{baseline_rho:.3f}" if baseline_rho else "—", f"{nc_rho:.3f}", f"{delta_rho:+.3f}" if baseline_rho else "—"],
                ]

                r.append(format_table(
                    ["Metric", "PeerRank (corrected)", "No Correction", "Δ"],
                    tfq_rows,
                    ['l', 'r', 'r', 'r']
                ))
                r.append(f"\n*n={tfq_val['n_models']} models. Bias correction improves correlation by {delta_r:+.3f} (Pearson).*\n")

    # Judge generosity
    judge_rows = [[str(i), n, f"{a:.2f}", f"{s:.2f}", str(c)] for i, (n, a, s, c) in enumerate(stats["judge_data"], 1)]
    r.append("\n## Judge Generosity\n" + format_table(["#", "Model", "Avg Given", "Std", "Count"], judge_rows, ['c', 'l', 'r', 'r', 'r']))

    # Judge Agreement Matrix
    evaluations = phase3.get("evaluations", {})
    if evaluations:
        agreement = calculate_judge_agreement(evaluations)
        judges = agreement["judges"]
        matrix = agreement["matrix"]
        pairs = agreement["pairs"]

        r.append("\n## Judge Agreement Matrix\n")
        r.append("Pearson correlation between judges' scores (1.0 = perfect agreement):\n")

        # Create compact matrix with short names
        short_names = {j: j[:8] for j in judges}
        header = ["Judge"] + [short_names[j] for j in judges]
        matrix_rows = []
        for j1 in judges:
            row = [short_names[j1]]
            for j2 in judges:
                if j1 == j2:
                    row.append("—")
                else:
                    row.append(f"{matrix[j1][j2]:.2f}")
            matrix_rows.append(row)

        r.append(format_table(header, matrix_rows, ['l'] + ['c'] * len(judges)))

        # Top/bottom pairs summary
        if pairs:
            r.append("\n**Most similar judges:**")
            for j1, j2, corr, n in pairs[:3]:
                r.append(f"- {j1} ↔ {j2}: r={corr:.3f} (n={n})")

            r.append("\n**Least similar judges:**")
            for j1, j2, corr, n in pairs[-3:]:
                r.append(f"- {j1} ↔ {j2}: r={corr:.3f} (n={n})")

    # Statistical Analysis - Provider Clustering
    clustering = _calculate_provider_clustering(evaluations)
    if clustering:
        r.append("\n## Statistical Analysis\n")
        r.append("### Provider Clustering\n")
        r.append("Kruskal-Wallis H-test: Do models from the same provider score similarly?\n")

        h, p, eta = clustering["h_stat"], clustering["p_value"], clustering["eta_sq"]
        n_prov, n_total = clustering["n_providers"], clustering["n_total"]
        sig = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

        r.append(f"- **H({n_prov - 1})** = {h:.2f}, {sig}")
        r.append(f"- **Effect size** (η²) = {eta:.3f}")
        r.append(f"- **Interpretation**: {'Significant' if p < 0.05 else 'Not significant'} provider effect")
        r.append(f"- **Samples**: {n_total} scores across {n_prov} providers\n")

        # Provider breakdown
        prov_rows = []
        for prov, (n, avg) in sorted(clustering["provider_scores"].items(), key=lambda x: -x[1][1]):
            prov_rows.append([prov, str(n), f"{avg:.2f}"])
        r.append(format_table(["Provider", "N", "Avg Score"], prov_rows, ['l', 'r', 'r']))

    # Home Advantage Analysis
    home_adv = _calculate_home_advantage(phase1, evaluations)
    if home_adv and home_adv["results"]:
        r.append("\n## Home Advantage Analysis\n")
        r.append("Do models perform better on questions they generated? (Peer scores, excluding self-evaluation)\n")

        # Short name helper
        def short_name(m):
            shortcuts = {"gemini-3-pro-preview": "gem-3-pro", "gemini-3-flash-thinking": "gem-3-flash",
                         "claude-opus-4-5": "opus-4.5", "claude-sonnet-4-5": "sonnet-4.5",
                         "llama-4-maverick": "llama-4", "deepseek-chat": "deepseek",
                         "kimi-k2-0905": "kimi", "grok-4-1-fast": "grok-4", "mistral-large": "mistral"}
            return shortcuts.get(m, m)[:12]

        # Results table
        ha_rows = []
        for res in home_adv["results"]:
            sig = ""
            if res["p_value"] is not None:
                if res["p_value"] < 0.001: sig = "***"
                elif res["p_value"] < 0.01: sig = "**"
                elif res["p_value"] < 0.05: sig = "*"
            d_str = f"{res['cohens_d']:+.2f}" if res["cohens_d"] else "—"
            ha_rows.append([
                short_name(res["model"]),
                f"{res['own_avg']:.2f}",
                f"{res['other_avg']:.2f}",
                f"{res['diff']:+.2f}",
                str(res["n_own"]),
                str(res["n_other"]),
                d_str,
                sig
            ])
        r.append(format_table(
            ["Model", "Own Qs", "Other Qs", "Diff", "n_own", "n_other", "Cohen's d", "Sig"],
            ha_rows, ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'l']
        ))

        # Summary statistics
        sig_pos = sum(1 for res in home_adv["results"] if res["p_value"] and res["p_value"] < 0.05 and res["diff"] > 0)
        sig_neg = sum(1 for res in home_adv["results"] if res["p_value"] and res["p_value"] < 0.05 and res["diff"] < 0)
        not_sig = len(home_adv["results"]) - sig_pos - sig_neg
        all_diffs = [res["diff"] for res in home_adv["results"]]
        avg_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0

        r.append(f"\n*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 | Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large*")
        r.append(f"\n**Summary:** {sig_pos} models better on own questions, {sig_neg} worse, {not_sig} not significant. Average home advantage: {avg_diff:+.3f} points.")

        # Question difficulty by source
        if home_adv["source_difficulty"]:
            r.append("\n### Question Difficulty by Source\n")
            r.append("Average peer score on questions from each model (lower = harder questions):\n")
            sorted_diff = sorted(home_adv["source_difficulty"].items(), key=lambda x: x[1])
            diff_rows = [[short_name(src), f"{score:.2f}"] for src, score in sorted_diff]
            r.append(format_table(["Source Model", "Avg Score"], diff_rows, ['l', 'r']))

    # Question Autopsy - flatten questions_by_model into a list
    questions = []
    for model, model_qs in phase1.get("questions_by_model", {}).items():
        for q in model_qs:
            questions.append({
                "question": q.get("question", ""),
                "category": q.get("category", ""),
                "source_model": model,
                "id": q.get("question", "")[:50],  # Use question text as ID
            })
    if evaluations and questions:
        q_stats = calculate_question_stats(evaluations, questions)

        r.append("\n## Question Autopsy\n")
        r.append("Analysis of question difficulty and controversy based on evaluation scores.\n")

        # Helper to shorten category names
        def short_cat(cat):
            if not cat:
                return "—"
            return cat.split("(")[0].strip()[:12]

        # Helper to shorten model names
        def short_model(model):
            if not model:
                return "—"
            return model[:10]

        # Hardest questions (lowest avg score)
        r.append("### 🔥 Hardest Questions (lowest avg score)\n")
        hard_rows = []
        for q_id, qs in q_stats["hardest"]:
            text_short = qs["text"][:50] + "..." if len(qs["text"]) > 50 else qs["text"]
            hard_rows.append([
                f"{qs['avg']:.2f}",
                f"±{qs['std']:.2f}",
                short_cat(qs.get("category", "")),
                short_model(qs.get("source", "")),
                text_short
            ])
        r.append(format_table(["Avg", "Std", "Category", "Creator", "Question"], hard_rows, ['r', 'r', 'l', 'l', 'l']))

        # Most controversial (highest std = most disagreement)
        r.append("\n### ⚔️ Most Controversial (judges disagree)\n")
        cont_rows = []
        for q_id, qs in q_stats["controversial"]:
            text_short = qs["text"][:50] + "..." if len(qs["text"]) > 50 else qs["text"]
            cont_rows.append([
                f"±{qs['std']:.2f}",
                f"{qs['avg']:.2f}",
                short_cat(qs.get("category", "")),
                short_model(qs.get("source", "")),
                text_short
            ])
        r.append(format_table(["Std", "Avg", "Category", "Creator", "Question"], cont_rows, ['r', 'r', 'l', 'l', 'l']))

        # Easiest questions (highest avg score)
        r.append("\n### ✅ Easiest Questions (highest avg score)\n")
        easy_rows = []
        for q_id, qs in q_stats["easiest"]:
            text_short = qs["text"][:50] + "..." if len(qs["text"]) > 50 else qs["text"]
            easy_rows.append([
                f"{qs['avg']:.2f}",
                f"±{qs['std']:.2f}",
                short_cat(qs.get("category", "")),
                short_model(qs.get("source", "")),
                text_short
            ])
        r.append(format_table(["Avg", "Std", "Category", "Creator", "Question"], easy_rows, ['r', 'r', 'l', 'l', 'l']))

        # Consensus questions (lowest std = most agreement)
        r.append("\n### 🤝 Consensus Questions (judges agree)\n")
        cons_rows = []
        for q_id, qs in q_stats["consensus"]:
            text_short = qs["text"][:50] + "..." if len(qs["text"]) > 50 else qs["text"]
            cons_rows.append([
                f"±{qs['std']:.2f}",
                f"{qs['avg']:.2f}",
                short_cat(qs.get("category", "")),
                short_model(qs.get("source", "")),
                text_short
            ])
        r.append(format_table(["Std", "Avg", "Category", "Creator", "Question"], cons_rows, ['r', 'r', 'l', 'l', 'l']))

    # Chart
    chart = ["```", "  PEER SCORE                              RESPONSE TIME",
             "                    0    2    4    6    8   10    │    0s   10s   20s   30s",
             "                    ├────┼────┼────┼────┼────┤    │    ├─────┼─────┼─────┤"]
    for i, (m, avg, _, _, t, _, _, _) in enumerate(stats["peer_data"], 1):
        s_len = int(avg / 10 * 25)
        s_bar = "█" * s_len + "░" * (25 - s_len)
        t_bar = "▓" * min(int(t / 30 * 25), 25)
        chart.append(f"  #{i} {m[:18]:<18} {s_bar} {avg:>5.2f}  │  {t_bar:<25} {t:>5.2f}s")
    chart.extend(["                    └─────────────────────────┘    │    └─────────────────────────┘", "```"])
    r.append("\n## Performance Overview\n" + "\n".join(chart))

    # Methodology
    r.append("""\n---\n## Methodology
- Phase 1: Each model generates questions across categories
- Phase 2: All models answer all questions with web search enabled
- Phase 3: Each model evaluates all responses in 3 modes:
  - Shuffle Only: Randomized order, real model names shown
  - Blind Only: Fixed order, model names hidden (Response A, B, C...)
  - Shuffle + Blind: Randomized order + hidden names (baseline Peer score)
- Phase 4: Aggregate scores and generate this report

**Bias Detection (positive = factor helped):**
- **Self Bias** = Self − Peer: How much the model overrated itself
- **Name Bias** = Shuffle − Peer: How much name recognition helped
- **Position Bias** = Blind − Peer: How much the fixed position helped""")

    report_text = "\n".join(r)
    report_file = DATA_DIR / f"phase4_report_{revision}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport saved: {report_file}")
    return report_text
