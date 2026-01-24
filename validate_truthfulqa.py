"""
TruthfulQA Validation - Correlate peer rankings with ground truth

Usage:
    python truth_validation.py                    # Interactive menu
    python truth_validation.py --all              # Run all phases
    python truth_validation.py --phase 1-5       # Run specific phase
    python truth_validation.py --num-questions 50
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from statistics import mean
import random

from peerrank.config import (
    MODELS, DATA_DIR, format_duration, CATEGORIES,
    set_revision, calculate_scores_from_evaluations,
    PROVIDER_CONCURRENCY,
)
from peerrank.providers import call_llm, clear_clients

# =============================================================================
# CONFIGURATION
# =============================================================================

TRUTH_DIR = DATA_DIR / "TRUTH"  # All TruthfulQA files go here
VALIDATION_REVISION = "TFQ"
NUM_QUESTIONS = 10

# Category mapping (TruthfulQA -> PeerRank)
CATEGORY_MAP = {
    # reasoning/logic
    "Logical Falsehood": "reasoning/logic",
    "Statistics": "reasoning/logic",
    "Indexical Error: Identity": "reasoning/logic",
    "Indexical Error: Location": "reasoning/logic",
    "Indexical Error: Time": "reasoning/logic",
    "Indexical Error: Other": "reasoning/logic",
    "Confusion: People": "reasoning/logic",
    "Confusion: Places": "reasoning/logic",
    "Confusion: Other": "reasoning/logic",
    # creative/open-ended
    "Subjective": "creative/open-ended",
    "Proverbs": "creative/open-ended",
    # practical how-to
    "Nutrition": "practical how-to",
    "Health": "practical how-to",
    # current events
    "Politics": "current events (needs recent info)",
    "Misinformation": "current events (needs recent info)",
}  # All others default to "factual knowledge"


def set_num_questions(n: int):
    global NUM_QUESTIONS
    NUM_QUESTIONS = n


# =============================================================================
# FILE I/O
# =============================================================================

def load_validation_json(filename: str) -> dict:
    base, ext = filename.rsplit(".", 1)
    filepath = TRUTH_DIR / f"{base}_{VALIDATION_REVISION}.{ext}"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath.name} not found")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_validation_json(filename: str, data: dict):
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    base, ext = filename.rsplit(".", 1)
    filepath = TRUTH_DIR / f"{base}_{VALIDATION_REVISION}.{ext}"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath.name}")


def progress_bar(completed: int, total: int, width: int = 40) -> str:
    pct = completed * 100 // total
    filled = pct * width // 100
    bar = "=" * filled + ">" + "." * (width - filled - 1) if filled < width else "=" * width
    return f"[{bar}] {pct:3}% ({completed}/{total})"


# =============================================================================
# PHASE 1: Generate Questions
# =============================================================================

def phase1_generate(num_questions: int = 10):
    """Generate MC questions from TruthfulQA with ground truth."""
    from datasets import load_dataset

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Generate TruthfulQA Questions")
    print(f"{'=' * 60}")

    # Load both splits - MC has answers, generation has categories
    mc_dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    gen_dataset = load_dataset("truthful_qa", "generation", split="validation")
    print(f"  Dataset: {len(mc_dataset)} questions available")

    # Build question -> TruthfulQA category mapping from generation split
    question_to_tfq_cat = {row["question"]: row.get("category", "") for row in gen_dataset}

    # Group by category
    active_cats = set(CATEGORIES)
    by_cat = {c: [] for c in active_cats}

    for row in mc_dataset:
        tfq_cat = question_to_tfq_cat.get(row["question"], "")
        cat = CATEGORY_MAP.get(tfq_cat, "factual knowledge")
        if cat not in by_cat:
            continue
        mc1 = row.get("mc1_targets", {})
        choices, labels = mc1.get("choices", []), mc1.get("labels", [])
        if not choices or not labels or 1 not in labels:
            continue
        correct_idx = labels.index(1)
        by_cat[cat].append({
            "category": cat,
            "question": row["question"],
            "choices": choices,
            "correct_index": correct_idx,
        })

    # Sample evenly across categories
    questions = []
    cats_with_data = [c for c in active_cats if by_cat.get(c)]
    per_cat = num_questions // len(cats_with_data) if cats_with_data else 0
    remainder = num_questions % len(cats_with_data) if cats_with_data else 0

    for i, cat in enumerate(cats_with_data):
        take = min(per_cat + (1 if i < remainder else 0), len(by_cat[cat]))
        questions.extend(random.sample(by_cat[cat], take))
    random.shuffle(questions)

    # Save phase1 questions
    output = {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "questions_by_model": {"TRUTHFUL_QA": [
            {"category": q["category"], "question": q["question"], "choices": q["choices"]}
            for q in questions
        ]}
    }
    save_validation_json("phase1_questions.json", output)

    # Save ground truth
    ground_truth = {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "questions": questions
    }
    save_validation_json("phase1_ground_truth.json", ground_truth)

    # Log category statistics
    print(f"\n  Generated {len(questions)} questions")
    print(f"  {'-' * 40}")
    print(f"  {'Category':<35} {'Selected':>8} {'Available':>10}")
    print(f"  {'-' * 40}")

    # Count selected questions per category
    selected_counts = {}
    for q in questions:
        cat = q["category"]
        selected_counts[cat] = selected_counts.get(cat, 0) + 1

    for cat in sorted(cats_with_data, key=lambda c: c[:20]):
        selected = selected_counts.get(cat, 0)
        available = len(by_cat.get(cat, []))
        short_cat = cat[:33] + ".." if len(cat) > 35 else cat
        print(f"  {short_cat:<35} {selected:>8} {available:>10}")

    print(f"  {'-' * 40}")
    total_available = sum(len(v) for v in by_cat.values())
    print(f"  {'TOTAL':<35} {len(questions):>8} {total_available:>10}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 2: Answer Questions (MC + Explanation)
# =============================================================================

async def phase2_answer():
    """Models answer MC questions with explanations."""
    set_revision(VALIDATION_REVISION)

    print(f"\n{'=' * 60}")
    print("  PHASE 2: Answer Questions (MC + Explanation)")
    print(f"{'=' * 60}")

    phase_start = time.time()
    questions = load_validation_json("phase1_questions.json")["questions_by_model"]["TRUTHFUL_QA"]
    model_names = [n for _, _, n in MODELS]
    total = len(questions) * len(model_names)
    completed = 0
    lock = asyncio.Lock()

    print(f"  Models: {len(model_names)} | Questions: {len(questions)} | Total: {total}")

    async def answer_one(provider, model_id, model_name, question, shuffled_choices, correct_letter, semaphore):
        nonlocal completed
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        choices_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(shuffled_choices))

        prompt = f"""Answer this multiple-choice question.

Output format (strict):
First line: exactly one letter (A/B/C/...) and nothing else.
Second line: 2-3 sentence explanation.

Question: {question["question"]}

{choices_text}"""

        try:
            async with semaphore:
                response, duration, in_tok, out_tok, _ = await call_llm(
                    provider, model_id, prompt, max_tokens=16000, timeout=180, temperature=0
                )
            answer_letter = next((c for c in response.upper() if c in letters[:len(question["choices"])]), "?")
            result = {"text": response.strip(), "answer_letter": answer_letter,
                      "duration": duration, "input_tokens": in_tok, "output_tokens": out_tok}
        except Exception as e:
            result = {"text": f"Error: {e}", "answer_letter": "?", "duration": 0, "input_tokens": 0, "output_tokens": 0}

        async with lock:
            completed += 1
            print(f"\r  {progress_bar(completed, total)}    ", end="", flush=True)
        return model_name, result

    # Process all questions
    output_questions = []
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p, _, _ in MODELS}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for question in questions:
        # Shuffle choices to avoid position bias (correct answer is always first in TruthfulQA)
        original_choices = question["choices"]
        correct_answer = original_choices[0]  # TruthfulQA: correct is always index 0

        # Create shuffled order with deterministic seed per question
        indices = list(range(len(original_choices)))
        rng = random.Random(hash(question["question"]))
        rng.shuffle(indices)

        shuffled_choices = [original_choices[i] for i in indices]
        correct_letter = letters[shuffled_choices.index(correct_answer)]

        tasks = [answer_one(p, m, n, question, shuffled_choices, correct_letter, semaphores[p]) for p, m, n in MODELS]
        results = await asyncio.gather(*tasks)
        output_questions.append({
            "text": question["question"],
            "category": question["category"],
            "choices": shuffled_choices,  # Store shuffled order
            "correct_letter": correct_letter,  # Track where correct answer ended up
            "answers": {name: result for name, result in results}
        })

    print()
    save_validation_json("phase2_answers.json", {
        "revision": VALIDATION_REVISION, "timestamp": datetime.now().isoformat(),
        "phase": 2, "duration_seconds": round(time.time() - phase_start, 2),
        "questions": output_questions
    })
    print(f"  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 3: Peer Evaluation (shuffle_blind only - bias analysis not needed)
# =============================================================================

async def phase3_evaluate():
    """Run peer evaluation (shuffle_blind mode only).

    For validation, we only need peer scores to correlate with accuracy.
    Full bias analysis (3 modes) is unnecessary and triples the cost.
    """
    set_revision(VALIDATION_REVISION)
    from peerrank_phase3 import _run_evaluation_pass
    from peerrank.config import get_bias_test_config, calculate_timing_stats

    phase_start = time.time()
    seed = get_bias_test_config()["seed"]
    questions = load_validation_json("phase2_answers.json")["questions"]

    print(f"\n{'=' * 60}")
    print("  PHASE 3: Peer Evaluation")
    print(f"{'=' * 60}")

    evaluations, timing = await _run_evaluation_pass(questions, shuffle=True, blind=True, seed=seed, mode_name="shuffle_blind")

    save_validation_json("phase3_rankings.json", {
        "revision": VALIDATION_REVISION, "timestamp": datetime.now().isoformat(),
        "phase": 3, "duration_seconds": round(time.time() - phase_start, 2),
        "evaluations_by_mode": {"shuffle_blind": evaluations},
        "evaluations": evaluations,
        "timing_stats": calculate_timing_stats(timing),
    })
    print(f"  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 4: Ground Truth Scoring (MC Accuracy)
# =============================================================================

def phase4_ground_truth_score():
    """Score MC accuracy (objective, no LLM judge needed)."""
    print(f"\n{'=' * 60}")
    print("  PHASE 4: Ground Truth Accuracy")
    print(f"{'=' * 60}")

    phase2 = load_validation_json("phase2_answers.json")
    model_names = [n for _, _, n in MODELS]
    scores = {n: [] for n in model_names}

    # Fallback: load ground truth for old data without shuffled choices
    gt_map = None
    if phase2["questions"] and "correct_letter" not in phase2["questions"][0]:
        ground_truth = load_validation_json("phase1_ground_truth.json")
        gt_map = {gt["question"]: gt["correct_index"] for gt in ground_truth["questions"]}
        print("  Note: Using legacy ground truth (choices not shuffled)")

    for q in phase2["questions"]:
        # Use correct_letter stored in phase2 (accounts for shuffled choices)
        correct_letter = q.get("correct_letter")
        if not correct_letter and gt_map:
            correct_idx = gt_map.get(q["text"])
            if correct_idx is None:
                continue
            correct_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[correct_idx]

        for model in model_names:
            ans = q["answers"].get(model, {})
            answer_letter = ans.get("answer_letter", "?") if isinstance(ans, dict) else "?"
            scores[model].append(1 if answer_letter == correct_letter else 0)

    # Calculate accuracy
    summary = {}
    for model, score_list in scores.items():
        correct = sum(score_list)
        total = len(score_list)
        summary[model] = {
            "accuracy": round(100 * correct / total, 1) if total else 0,
            "correct": correct, "total": total,
            "mean": round(10 * correct / total, 2) if total else 0,  # 0-10 scale
        }

    # Print rankings
    ranked = sorted(summary.items(), key=lambda x: (-x[1]["accuracy"], x[0]))
    print(f"\n  {'Model':<25} {'Accuracy':>10}")
    print(f"  {'-' * 35}")
    for model, stats in ranked:
        print(f"  {model:<25} {stats['accuracy']:>8.1f}% ({stats['correct']}/{stats['total']})")

    save_validation_json("phase4_TFQ_scores.json", {
        "revision": VALIDATION_REVISION, "timestamp": datetime.now().isoformat(),
        "phase": 4, "judge_model": "N/A (direct accuracy)", "summary": summary
    })
    print(f"\n{'=' * 60}")


# =============================================================================
# PHASE 5: Correlation Analysis
# =============================================================================

def phase5_correlation_analysis():
    """Correlate peer scores with ground truth accuracy."""
    print(f"\n{'=' * 60}")
    print("  PHASE 5: Correlation Analysis")
    print(f"{'=' * 60}")

    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("  Error: scipy required. pip install scipy")
        return None

    truth_data = load_validation_json("phase4_TFQ_scores.json")

    try:
        phase3_data = load_validation_json("phase3_rankings.json")
    except FileNotFoundError:
        print("  No Phase 3 data. Run Phase 3 first.")
        return None

    # Get peer scores
    evaluations = phase3_data.get("evaluations_by_mode", {}).get("shuffle_blind", phase3_data.get("evaluations", {}))
    model_names = [n for _, _, n in MODELS]
    scores_result = calculate_scores_from_evaluations(evaluations, model_names)

    peer_means = {m: mean(s) for m, s in scores_result["peer_scores"].items() if s}
    truth_means = {m: stats["mean"] for m, stats in truth_data["summary"].items() if stats["mean"] > 0}

    common = sorted(set(peer_means) & set(truth_means))
    if len(common) < 3:
        print(f"  Need 3+ models with both scores. Found: {len(common)}")
        return None

    peer_arr = [peer_means[m] for m in common]
    truth_arr = [truth_means[m] for m in common]

    # Check for zero variance (all scores identical)
    if len(set(truth_arr)) == 1:
        print(f"\n  WARNING: All truth scores identical ({truth_arr[0]:.1f})")
        print(f"  Cannot compute correlation. Try more questions for variance.")
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1
    else:
        pearson_r, pearson_p = pearsonr(peer_arr, truth_arr)
        spearman_r, spearman_p = spearmanr(peer_arr, truth_arr)

    print(f"\n  Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"  Spearman:   {spearman_r:.4f} (p={spearman_p:.4f})")

    # Build comparison with proper tie handling
    peer_ranked = sorted(common, key=lambda m: -peer_means[m])
    truth_ranked = sorted(common, key=lambda m: -truth_means[m])

    # Handle ties in truth ranking
    truth_ranks = {}
    i = 0
    while i < len(truth_ranked):
        # Find all models with same score
        score = truth_means[truth_ranked[i]]
        tied = [truth_ranked[i]]
        j = i + 1
        while j < len(truth_ranked) and truth_means[truth_ranked[j]] == score:
            tied.append(truth_ranked[j])
            j += 1
        # Assign average rank to tied models
        avg_rank = (i + 1 + j) / 2
        for m in tied:
            truth_ranks[m] = avg_rank if len(tied) > 1 else i + 1
        i = j

    peer_ranks = {m: i + 1 for i, m in enumerate(peer_ranked)}

    # Build report
    comparison = []
    for m in common:
        comparison.append({
            "model": m,
            "peer_score": round(peer_means[m], 2),
            "truth_score": round(truth_means[m], 2),
            "peer_rank": peer_ranks[m],
            "truth_rank": truth_ranks[m],
            "rank_diff": peer_ranks[m] - truth_ranks[m],
        })
    comparison.sort(key=lambda x: x["peer_rank"])

    # Print comparison table
    print(f"\n  {'Model':<22} {'Peer':>6} {'Truth':>6} {'P.Rank':>7} {'T.Rank':>7}")
    print(f"  {'-' * 50}")
    for row in comparison:
        tr = f"{row['truth_rank']:.1f}" if row['truth_rank'] != int(row['truth_rank']) else f"{int(row['truth_rank'])}"
        print(f"  {row['model']:<22} {row['peer_score']:>6.2f} {row['truth_score']:>6.2f} {row['peer_rank']:>7} {tr:>7}")

    # Interpret
    def interpret(r):
        ar = abs(r)
        if ar >= 0.8: return "strong"
        if ar >= 0.6: return "moderate"
        if ar >= 0.4: return "weak"
        return "none"

    interp = interpret(pearson_r)

    # Save report with fixed-width columns
    num_q = truth_data['summary'][common[0]]['total']
    report = f"""# TruthfulQA Validation Report

Revision: {VALIDATION_REVISION}
Models:   {len(common)}
Questions: {num_q}

## Correlation

  Metric       Value    p-value   Interpretation
  ----------   ------   -------   --------------
  Pearson r    {pearson_r:>6.4f}   {pearson_p:>7.4f}   {interp}
  Spearman     {spearman_r:>6.4f}   {spearman_p:>7.4f}   {interpret(spearman_r)}

## Score Comparison

  Rank  Model                      Peer   Truth  T.Rank
  ----  -------------------------  -----  -----  ------
"""
    for row in comparison:
        tr = f"{row['truth_rank']:.1f}" if row['truth_rank'] != int(row['truth_rank']) else str(int(row['truth_rank']))
        report += f"  {row['peer_rank']:>4}  {row['model']:<25}  {row['peer_score']:>5.2f}  {row['truth_score']:>5.2f}  {tr:>6}\n"

    report += f"\n## Conclusion\n\n"
    if pearson_r >= 0.7 and pearson_p < 0.05:
        report += f"Peer evaluation **strongly correlates** with ground truth (r={pearson_r:.3f})."
    elif pearson_r >= 0.5 and pearson_p < 0.05:
        report += f"Peer evaluation shows **moderate correlation** with ground truth (r={pearson_r:.3f})."
    elif len(set(truth_arr)) == 1:
        report += f"**Cannot determine correlation** - all models achieved identical accuracy. Use more questions."
    else:
        report += f"Peer evaluation shows **weak/no correlation** with ground truth (r={pearson_r:.3f})."

    report_file = TRUTH_DIR / f"TFQ_validation_report_{VALIDATION_REVISION}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {report_file.name}")

    save_validation_json("TFQ_analysis.json", {
        "revision": VALIDATION_REVISION, "timestamp": datetime.now().isoformat(),
        "correlation": {"pearson_r": round(pearson_r, 4), "pearson_p": round(pearson_p, 4),
                        "spearman_r": round(spearman_r, 4), "spearman_p": round(spearman_p, 4)},
        "comparison": comparison,
    })

    print(f"\n{'=' * 60}")
    return {"pearson_r": pearson_r, "spearman_r": spearman_r}


# =============================================================================
# RUN ALL PHASES
# =============================================================================

async def run_all_phases(num_questions: int = 10):
    """Run complete validation workflow."""
    print(f"\n{'#' * 60}")
    print(f"  TruthfulQA VALIDATION")
    print(f"{'#' * 60}")
    print(f"  Models: {len(MODELS)} | Questions: {num_questions}")
    print(f"{'#' * 60}\n")

    start = time.time()
    phase1_generate(num_questions)
    await phase2_answer()
    await phase3_evaluate()
    phase4_ground_truth_score()
    phase5_correlation_analysis()

    print(f"\n{'#' * 60}")
    print(f"  COMPLETE in {format_duration(time.time() - start)}")
    print(f"{'#' * 60}\n")


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def get_last_completed_phase() -> int:
    for phase, fn in [(5, "TFQ_analysis"), (4, "phase4_TFQ_scores"),
                      (3, "phase3_rankings"), (2, "phase2_answers"), (1, "phase1_questions")]:
        if (TRUTH_DIR / f"{fn}_{VALIDATION_REVISION}.json").exists():
            return phase
    return 0


def show_menu():
    print("\n" + "=" * 50)
    print("  TruthfulQA Validation")
    print("=" * 50)
    print(f"  Progress: Phase {get_last_completed_phase()}/5")
    print(f"  Models: {len(MODELS)} | Questions: {NUM_QUESTIONS}")
    print(f"""
  [1-5] Run Phase 1-5
  [A] Run ALL    [N] Set questions ({NUM_QUESTIONS})
  [R] View report [Q] Quit
""")
    return input("  > ").strip().upper()


def interactive_menu():
    while True:
        choice = show_menu()
        if choice == "1": phase1_generate(NUM_QUESTIONS)
        elif choice == "2": clear_clients(); asyncio.run(phase2_answer())
        elif choice == "3": clear_clients(); asyncio.run(phase3_evaluate())
        elif choice == "4": phase4_ground_truth_score()
        elif choice == "5": phase5_correlation_analysis()
        elif choice == "A": clear_clients(); asyncio.run(run_all_phases(NUM_QUESTIONS))
        elif choice == "N":
            try:
                n = int(input("  Questions (1-500): "))
                if 1 <= n <= 500: set_num_questions(n)
            except ValueError: pass
        elif choice == "R":
            rf = TRUTH_DIR / f"TFQ_validation_report_{VALIDATION_REVISION}.md"
            if rf.exists(): print(rf.read_text())
            else: print("  No report yet")
        elif choice == "Q": break


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Validation")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.num_questions: set_num_questions(args.num_questions)

    if args.phase == 1: phase1_generate(NUM_QUESTIONS)
    elif args.phase == 2: clear_clients(); asyncio.run(phase2_answer())
    elif args.phase == 3: clear_clients(); asyncio.run(phase3_evaluate())
    elif args.phase == 4: phase4_ground_truth_score()
    elif args.phase == 5: phase5_correlation_analysis()
    elif args.all: clear_clients(); asyncio.run(run_all_phases(NUM_QUESTIONS))
    else: interactive_menu()


if __name__ == "__main__":
    main()
