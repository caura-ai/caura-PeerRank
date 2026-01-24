"""
phase5.py - Final Analysis by Judge LLM
"""

import asyncio
import time
from datetime import datetime

from statistics import mean

from peerrank.config import (
    DATA_DIR, MODELS, get_revision, format_duration, load_json, format_table,
    get_phase5_judge, MAX_TOKENS_EVAL, EFFICIENCY_QUALITY_EXPONENT,
    calculate_scores_from_evaluations,
)
from peerrank.providers import call_llm


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def _calc_avg_estimated_input(phase2: dict, model_name: str) -> int:
    """Calculate estimated avg input tokens for a model from question lengths."""
    questions = phase2.get("questions", [])
    web_search_enabled = phase2.get("web_search", False)
    if not questions:
        return 0

    # Web search context adds ~2000 tokens on average
    search_context_tokens = 2000 if web_search_enabled else 0

    estimates = []
    for q in questions:
        answers = q.get("answers", {})
        if model_name in answers:
            # Base: question text + typical system prompt (~50 tokens) + search context
            q_tokens = _estimate_tokens(q.get("text", "")) + 50 + search_context_tokens
            estimates.append(q_tokens)

    return int(mean(estimates)) if estimates else 0


def _generate_cost_analysis(phase2: dict, phase3: dict) -> str:
    """Generate cost analysis sections from phase2 and phase3 data."""
    cost_stats = phase2.get("cost_stats", {})
    total_cost = phase2.get("total_cost", 0)

    if not cost_stats:
        return ""

    sections = []

    # Calculate peer scores for Performance vs Cost
    evaluations = phase3.get("evaluations", {})
    scores = calculate_scores_from_evaluations(evaluations)
    peer_scores = scores["peer_scores"]

    # Build peer_data for scoring
    peer_data = []
    for _, _, name in MODELS:
        if peer_scores[name]:
            peer_avg = mean(peer_scores[name])
            peer_data.append((name, peer_avg))
    peer_data.sort(key=lambda x: x[1], reverse=True)

    # Answering API Cost Analysis
    cost_rows = []
    for _, _, n in MODELS:
        if n in cost_stats:
            cs = cost_stats[n]
            total = cs.get("total", 0)
            calls = cs.get("calls", 0)
            input_tokens = cs.get("input_tokens", 0)
            output_tokens = cs.get("output_tokens", 0)
            avg_cost = total / calls if calls > 0 else 0
            cost_rows.append([
                n,
                f"${total:.4f}",
                f"{input_tokens:,}",
                f"{output_tokens:,}",
                f"${avg_cost:.4f}"
            ])
    cost_rows.sort(key=lambda x: float(x[1].replace("$", "")))
    sections.append(f"## Answering API Cost Analysis\n\nTotal Phase 2 Cost: **${total_cost:.4f}**\n\n" +
                    format_table(["Model", "Total Cost", "Total Input Tokens", "Total Output Tokens", "Avg Cost/Q"],
                                 cost_rows, ['l', 'r', 'r', 'r', 'r']))

    # Performance vs. Cost Analysis
    perf_cost_rows = []
    for i, (n, avg) in enumerate(peer_data, 1):
        cs = cost_stats.get(n, {})
        total_model_cost = cs.get("total", 0)
        calls = cs.get("calls", 0)
        input_tokens = cs.get("input_tokens", 0)
        output_tokens = cs.get("output_tokens", 0)

        avg_cost = total_model_cost / calls if calls > 0 else 0
        avg_input_reported = input_tokens / calls if calls > 0 else 0
        avg_input_estimated = _calc_avg_estimated_input(phase2, n)
        avg_output = output_tokens / calls if calls > 0 else 0
        avg_cost_cents = avg_cost * 100  # Convert to cents

        # Efficiency metric: (Peer Score ^ exponent) / Cost in cents
        efficiency = (avg ** EFFICIENCY_QUALITY_EXPONENT) / avg_cost_cents if avg_cost_cents > 0 else 0

        perf_cost_rows.append([
            str(i),
            n,
            f"{avg:.2f}",
            f"{int(avg_input_reported):,}",
            f"{int(avg_input_estimated):,}",
            f"{int(avg_output):,}",
            f"${avg_cost:.4f}",
            f"{efficiency:.2f}"
        ])

    # Sort by efficiency rank for display
    perf_cost_rows.sort(key=lambda x: float(x[7]), reverse=True)
    for eff_rank, row in enumerate(perf_cost_rows, 1):
        row[0] = str(eff_rank)

    # Format exponent for display
    if EFFICIENCY_QUALITY_EXPONENT == 2.0:
        exp_display = "²"
    elif EFFICIENCY_QUALITY_EXPONENT == 1.5:
        exp_display = "¹·⁵"
    else:
        exp_str = str(EFFICIENCY_QUALITY_EXPONENT).replace(".", "·")
        exp_display = "".join(["⁰¹²³⁴⁵⁶⁷⁸⁹"[int(c)] if c.isdigit() else c for c in exp_str])

    sections.append(f"\n## Performance vs. Cost\n\nEfficiency analysis (score-weighted value, ranked by Points{exp_display}/¢):\n" +
                    format_table(["Rank", "Model", "Score", "Reported In", "Est. In", "Output", "Cost/Ans", f"Pts{exp_display}/¢"],
                                 perf_cost_rows, ['c', 'l', 'r', 'r', 'r', 'r', 'r', 'r']))

    return "\n".join(sections)

ANALYSIS_PROMPT = """You are an expert analyst reviewing an LLM peer evaluation report.

Analyze this PeerRank evaluation report and provide insights:

{report}

---

Provide a comprehensive analysis covering:

## 1. Overall Peer Ranking Assessment
Summarize the peer evaluation of participating LLMs based on peer scores, response times, and consistency.

## 2. Top Performers
Which models stood out and why? Consider both score and efficiency.

## 3. Outliers & Anomalies
Identify any unusual patterns:
- Models with unexpectedly high/low scores
- Large gaps between self-rating and peer-rating (self-bias)
- Unusual position or name bias effects
- Response time outliers

## 4. Interesting Findings
What patterns or insights are worth highlighting?
- Correlation between speed and peer scores
- Judge generosity patterns (harsh vs lenient evaluators)
- Category performance differences

## 5. Potential Concerns
Any red flags or issues worth investigating?
- Possible evaluation gaming
- Suspiciously high self-ratings
- Models that may have struggled

## 6. Recommendations
Based on this evaluation, which models would you recommend for:
- General use (best overall)
- Speed-critical applications
- High-stakes/accuracy-critical tasks

## 7. Media Headlines

Generate 5 attention-grabbing headlines for tech news coverage. Be specific with numbers and model names. Examples of tone:
- "GPT-5.2 Crowned King by AI Peers, But Claude Shows Surprising Self-Doubt"
- "Speed vs Smarts: Llama-4 Answers 3x Faster But Scores 30% Lower"
- "AI Models Rate Themselves 15% Higher Than Peers - Except One"

Make them punchy, factual, and newsworthy. Include the most surprising or counterintuitive findings.

Keep the analysis data-driven, citing specific numbers from the report."""


async def phase5_final_analysis() -> str:
    """Phase 5: Judge LLM analyzes the Phase 4 report."""
    judge_provider, judge_model, judge_name = get_phase5_judge()
    revision = get_revision()

    print(f"\n{'=' * 60}")
    print(f"  PHASE 5: Final Analysis")
    print(f"{'-' * 60}")
    print(f"  Revision:    {revision}")
    print(f"  Judge:       {judge_name}")
    print(f"{'=' * 60}")

    # Load Phase 4 report
    report_file = DATA_DIR / f"phase4_report_{revision}.md"
    if not report_file.exists():
        raise FileNotFoundError(f"Phase 4 report not found: {report_file}")

    with open(report_file, "r", encoding="utf-8") as f:
        report_content = f.read()

    # Load phase2 and phase3 for cost analysis
    phase2 = load_json("phase2_answers.json")
    phase3 = load_json("phase3_rankings.json")
    cost_analysis = _generate_cost_analysis(phase2, phase3)

    print(f"\n  Loaded report: {len(report_content)} chars")
    print(f"  Sending to {judge_name} for analysis...", flush=True)

    phase_start = time.time()

    # Call judge LLM
    prompt = ANALYSIS_PROMPT.format(report=report_content)
    analysis, duration, _, _, _ = await call_llm(
        judge_provider, judge_model, prompt,
        max_tokens=MAX_TOKENS_EVAL,
        use_web_search=False
    )

    print(f"  Analysis received in {format_duration(duration)}")

    # Build output
    output = f"""# PeerRank.ai Final Analysis

**{revision}** | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Judge: **{judge_name}**

---

{cost_analysis}

---

{analysis}

---

*Analysis generated by {judge_name} in {format_duration(duration)}*
"""

    # Save analysis
    analysis_file = DATA_DIR / f"phase5_analysis_{revision}.md"
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write(output)

    total_duration = time.time() - phase_start
    print(f"\n{'=' * 60}")
    print(f"  Phase 5 complete in {format_duration(total_duration)}")
    print(f"  Analysis saved: {analysis_file}")
    print(f"{'=' * 60}")

    return output
