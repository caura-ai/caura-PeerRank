"""
phases/phase4.py - Report Generation
"""

from datetime import datetime
from statistics import mean, stdev

from config import (
    MODELS, ALL_MODELS, DATA_DIR, load_json, format_duration, get_revision,
    format_table, match_model_name, calculate_scores_from_evaluations,
    calculate_judge_agreement, calculate_question_stats,
    get_phase4_elo, calculate_elo_ratings, ELO_K_FACTOR,
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
    if get_phase4_elo():
        # Use shuffle_blind evaluations (same as peer scores) if available, else fallback
        evaluations_by_mode = phase3.get("evaluations_by_mode", {})
        elo_evaluations = evaluations_by_mode.get("shuffle_blind", phase3.get("evaluations", {}))

        if elo_evaluations:
            model_names = [n for _, _, n in MODELS]
            elo_data = calculate_elo_ratings(elo_evaluations, model_names)

            # Build peer score lookup for comparison
            peer_score_lookup = {n: avg for n, avg, _, _, _, _, _, _ in stats["peer_data"]}
            peer_rank_lookup = {n: i for i, (n, _, _, _, _, _, _, _) in enumerate(stats["peer_data"], 1)}

            # Sort by Elo rating
            elo_sorted = sorted(elo_data["ratings"].items(), key=lambda x: x[1], reverse=True)

            r.append(f"\n## Elo Ratings\n\nPairwise comparisons from evaluation scores (K={ELO_K_FACTOR}).\n"
                     f"Total matches: {elo_data['total_matches']:,}\n")

            elo_rows = []
            for elo_rank, (name, elo) in enumerate(elo_sorted, 1):
                wins, losses, ties = elo_data["matches"].get(name, (0, 0, 0))
                win_rate = elo_data["win_rates"].get(name, 0)
                peer_score = peer_score_lookup.get(name, 0)
                peer_rank = peer_rank_lookup.get(name, 0)
                rank_diff = peer_rank - elo_rank  # positive = Elo ranks higher

                elo_rows.append([
                    str(elo_rank),
                    name,
                    str(elo),
                    f"{win_rate:.1f}%",
                    f"{wins}-{losses}-{ties}",
                    f"{peer_score:.2f}",
                    str(peer_rank),
                    f"{rank_diff:+d}" if rank_diff != 0 else "—",
                ])

            r.append(format_table(
                ["#", "Model", "Elo", "Win%", "W-L-T", "Peer", "P#", "Diff"],
                elo_rows,
                ['c', 'l', 'r', 'r', 'c', 'r', 'c', 'c']
            ))

            # Calculate correlation between peer scores and Elo ratings
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
