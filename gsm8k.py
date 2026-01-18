"""
GSM8K Validation - Correlate peer rankings with math accuracy

Grade School Math 8K (GSM8K) tests models on multi-step arithmetic reasoning.
Unlike TruthfulQA (multiple choice), GSM8K uses open-ended problems with
numerical answers extracted via the #### pattern.

Usage:
    python gsm8k.py                           # Interactive menu
    python gsm8k.py --all                     # Run all phases
    python gsm8k.py --phase 1-5               # Run specific phase
    python gsm8k.py --num-questions 50        # Set question count
    python gsm8k.py --difficulty easy,medium  # Filter by difficulty
    python gsm8k.py --difficulty hard         # Only hard questions
"""

import argparse
import asyncio
import json
import re
import time
from datetime import datetime
from statistics import mean
import random

from config import (
    MODELS, DATA_DIR, format_duration,
    set_revision, calculate_scores_from_evaluations,
    PROVIDER_CONCURRENCY, calculate_timing_stats, get_bias_test_config,
)
from providers import call_llm

# =============================================================================
# CONFIGURATION
# =============================================================================

GSM8K_DIR = DATA_DIR / "GSM8K"  # All GSM8K files go here
VALIDATION_REVISION = "GSM8K"
NUM_QUESTIONS = 50

# Difficulty categories based on solution step count
DIFFICULTY_THRESHOLDS = {
    "easy": (1, 3),      # 1-3 steps
    "medium": (4, 5),    # 4-5 steps
    "hard": (6, 100),    # 6+ steps
}

# Active difficulty levels (default: all)
ALL_DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTIES = ALL_DIFFICULTIES.copy()


def set_num_questions(n: int):
    global NUM_QUESTIONS
    NUM_QUESTIONS = n


def set_difficulties(levels: list[str]):
    """Set which difficulty levels to include (easy, medium, hard)."""
    global DIFFICULTIES
    valid = [d for d in levels if d in ALL_DIFFICULTIES]
    if valid:
        DIFFICULTIES.clear()
        DIFFICULTIES.extend(valid)
    return DIFFICULTIES


def get_difficulties_display() -> str:
    """Get display string for current difficulty settings."""
    if set(DIFFICULTIES) == set(ALL_DIFFICULTIES):
        return "all"
    return "+".join(DIFFICULTIES)


# =============================================================================
# FILE I/O
# =============================================================================

def load_validation_json(filename: str) -> dict:
    base, ext = filename.rsplit(".", 1)
    filepath = GSM8K_DIR / f"{base}_{VALIDATION_REVISION}.{ext}"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath.name} not found")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_validation_json(filename: str, data: dict):
    GSM8K_DIR.mkdir(parents=True, exist_ok=True)
    base, ext = filename.rsplit(".", 1)
    filepath = GSM8K_DIR / f"{base}_{VALIDATION_REVISION}.{ext}"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath.name}")


def progress_bar(completed: int, total: int, width: int = 40) -> str:
    pct = completed * 100 // total
    filled = pct * width // 100
    bar = "=" * filled + ">" + "." * (width - filled - 1) if filled < width else "=" * width
    return f"[{bar}] {pct:3}% ({completed}/{total})"


# =============================================================================
# ANSWER EXTRACTION & NORMALIZATION
# =============================================================================

def extract_gold_answer(solution: str) -> float | None:
    """Extract gold answer from GSM8K solution text (#### number format)."""
    pattern = r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)"
    matches = re.findall(pattern, solution)
    if matches:
        return normalize_number(matches[-1])
    return None


def extract_model_answer(response: str) -> float | None:
    """
    Extract numerical answer from model response.

    Tries multiple patterns in order of preference:
    1. #### number (explicit format we requested)
    2. Final answer is/= number
    3. Therefore/So/Thus the answer is number
    4. Last number in response
    """
    if not response:
        return None

    # Pattern 1: #### number (preferred - what we ask for)
    pattern1 = r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)"
    matches = re.findall(pattern1, response)
    if matches:
        return normalize_number(matches[-1])

    # Pattern 2: "final answer is/=" followed by number
    pattern2 = r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)"
    matches = re.findall(pattern2, response, re.IGNORECASE)
    if matches:
        return normalize_number(matches[-1])

    # Pattern 3: "Therefore/So/Thus" statements
    pattern3 = r"(?:therefore|so|thus|hence)[^.]*?(\d+(?:,\d{3})*(?:\.\d+)?)"
    matches = re.findall(pattern3, response, re.IGNORECASE)
    if matches:
        return normalize_number(matches[-1])

    # Pattern 4: Boxed answer (LaTeX style)
    pattern4 = r"\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}"
    matches = re.findall(pattern4, response)
    if matches:
        return normalize_number(matches[-1])

    # Pattern 5: Last standalone number in response (fallback)
    # Match numbers that appear at end of sentences or lines
    pattern5 = r"(?:^|[^\d])(-?\d+(?:,\d{3})*(?:\.\d+)?)(?:\s*\.?\s*$|[^\d])"
    matches = re.findall(pattern5, response)
    if matches:
        # Filter out very small numbers that are likely part of reasoning
        candidates = [normalize_number(m) for m in matches if normalize_number(m) is not None]
        if candidates:
            return candidates[-1]

    return None


def normalize_number(num_str: str) -> float | None:
    """Normalize number string to float, handling commas and common formats."""
    if not num_str:
        return None
    try:
        # Remove commas, dollar signs, and whitespace
        cleaned = num_str.replace(",", "").replace("$", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def numbers_match(a: float | None, b: float | None, tolerance: float = 1e-6) -> bool:
    """Check if two numbers match within tolerance."""
    if a is None or b is None:
        return False
    # Exact match for integers
    if a == b:
        return True
    # Tolerance-based match for floats
    return abs(a - b) < tolerance


def count_solution_steps(solution: str) -> int:
    """Count reasoning steps in a GSM8K solution."""
    # Each step typically ends with a line or is separated by periods
    # Count lines that contain calculations or logical steps
    lines = [l.strip() for l in solution.split('\n') if l.strip()]
    step_indicators = ['=', '+', '-', '*', '/', 'so', 'therefore', 'thus', 'hence']
    steps = 0
    for line in lines:
        line_lower = line.lower()
        if any(ind in line_lower for ind in step_indicators):
            steps += 1
    return max(steps, 1)  # At least 1 step


def categorize_difficulty(solution: str) -> str:
    """Categorize question difficulty based on solution complexity."""
    steps = count_solution_steps(solution)
    for difficulty, (low, high) in DIFFICULTY_THRESHOLDS.items():
        if low <= steps <= high:
            return difficulty
    return "hard"  # Default for very complex problems


# =============================================================================
# PHASE 1: Load Questions from GSM8K
# =============================================================================

def phase1_generate(num_questions: int = 50):
    """Load questions from GSM8K dataset with stratified sampling by difficulty."""
    from datasets import load_dataset

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Load GSM8K Questions")
    print(f"{'=' * 60}")

    # Load GSM8K test split
    dataset = load_dataset("gsm8k", "main", split="test")
    print(f"  Dataset: {len(dataset)} questions available")

    # Categorize by difficulty
    by_difficulty = {"easy": [], "medium": [], "hard": []}

    for row in dataset:
        question = row["question"]
        solution = row["answer"]
        gold_answer = extract_gold_answer(solution)

        if gold_answer is None:
            continue  # Skip if we can't extract the answer

        difficulty = categorize_difficulty(solution)
        by_difficulty[difficulty].append({
            "question": question,
            "solution": solution,
            "gold_answer": gold_answer,
            "difficulty": difficulty,
            "steps": count_solution_steps(solution),
        })

    # Stratified sampling - try to balance across active difficulties
    questions = []
    difficulties_with_data = [d for d in DIFFICULTIES if by_difficulty.get(d)]
    per_difficulty = num_questions // len(difficulties_with_data) if difficulties_with_data else 0
    remainder = num_questions % len(difficulties_with_data) if difficulties_with_data else 0

    for i, difficulty in enumerate(difficulties_with_data):
        take = min(per_difficulty + (1 if i < remainder else 0), len(by_difficulty[difficulty]))
        questions.extend(random.sample(by_difficulty[difficulty], take))

    random.shuffle(questions)

    # Save phase1 questions (without gold answers for model answering)
    output = {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "questions_by_model": {"GSM8K": [
            {"question": q["question"], "difficulty": q["difficulty"]}
            for q in questions
        ]}
    }
    save_validation_json("phase1_questions.json", output)

    # Save ground truth (with gold answers)
    ground_truth = {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "questions": questions
    }
    save_validation_json("phase1_ground_truth.json", ground_truth)

    # Log difficulty statistics
    print(f"\n  Generated {len(questions)} questions")
    print(f"  {'-' * 40}")
    print(f"  {'Difficulty':<15} {'Selected':>10} {'Available':>12}")
    print(f"  {'-' * 40}")

    selected_counts = {}
    for q in questions:
        d = q["difficulty"]
        selected_counts[d] = selected_counts.get(d, 0) + 1

    for difficulty in ["easy", "medium", "hard"]:
        selected = selected_counts.get(difficulty, 0)
        available = len(by_difficulty.get(difficulty, []))
        print(f"  {difficulty:<15} {selected:>10} {available:>12}")

    print(f"  {'-' * 40}")
    total_available = sum(len(v) for v in by_difficulty.values())
    print(f"  {'TOTAL':<15} {len(questions):>10} {total_available:>12}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 2: Answer Questions (Chain-of-Thought Math)
# =============================================================================

async def phase2_answer():
    """Models answer math questions with chain-of-thought reasoning."""
    set_revision(VALIDATION_REVISION)

    print(f"\n{'=' * 60}")
    print("  PHASE 2: Answer Questions (Math with CoT)")
    print(f"{'=' * 60}")

    phase_start = time.time()
    questions = load_validation_json("phase1_questions.json")["questions_by_model"]["GSM8K"]
    model_names = [n for _, _, n in MODELS]
    total = len(questions) * len(model_names)
    completed = 0
    lock = asyncio.Lock()

    print(f"  Models: {len(model_names)} | Questions: {len(questions)} | Total: {total}")

    async def answer_one(provider, model_id, model_name, question, semaphore):
        nonlocal completed

        prompt = f"""Solve this math problem step by step.

Show your reasoning clearly, then provide your final numerical answer.

IMPORTANT: End your response with the final answer in this exact format:
#### <number>

For example, if the answer is 42, end with:
#### 42

Question: {question["question"]}"""

        try:
            async with semaphore:
                response, duration, in_tok, out_tok = await call_llm(
                    provider, model_id, prompt, max_tokens=2048, timeout=90,
                    temperature=0, use_web_search=False
                )

            extracted = extract_model_answer(response)
            result = {
                "text": response.strip(),
                "extracted_answer": extracted,
                "duration": duration,
                "input_tokens": in_tok,
                "output_tokens": out_tok
            }
        except Exception as e:
            result = {
                "text": f"Error: {e}",
                "extracted_answer": None,
                "duration": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }

        async with lock:
            completed += 1
            print(f"\r  {progress_bar(completed, total)}    ", end="", flush=True)
        return model_name, result

    # Process all questions
    output_questions = []
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p, _, _ in MODELS}

    for question in questions:
        tasks = [answer_one(p, m, n, question, semaphores[p]) for p, m, n in MODELS]
        results = await asyncio.gather(*tasks)
        output_questions.append({
            "text": question["question"],
            "difficulty": question.get("difficulty", "unknown"),
            "answers": {name: result for name, result in results}
        })

    print()
    save_validation_json("phase2_answers.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "duration_seconds": round(time.time() - phase_start, 2),
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
    from phase3 import _run_evaluation_pass

    phase_start = time.time()
    seed = get_bias_test_config()["seed"]
    questions = load_validation_json("phase2_answers.json")["questions"]

    print(f"\n{'=' * 60}")
    print("  PHASE 3: Peer Evaluation")
    print(f"{'=' * 60}")

    evaluations, timing = await _run_evaluation_pass(
        questions, shuffle=True, blind=True, seed=seed, mode_name="shuffle_blind"
    )

    save_validation_json("phase3_rankings.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 3,
        "duration_seconds": round(time.time() - phase_start, 2),
        "evaluations_by_mode": {"shuffle_blind": evaluations},
        "evaluations": evaluations,
        "timing_stats": calculate_timing_stats(timing),
    })
    print(f"  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 4: Ground Truth Scoring (Math Accuracy)
# =============================================================================

def phase4_ground_truth_score():
    """Score math accuracy (exact numerical match against gold answers)."""
    print(f"\n{'=' * 60}")
    print("  PHASE 4: Ground Truth Accuracy")
    print(f"{'=' * 60}")

    phase2 = load_validation_json("phase2_answers.json")
    ground_truth = load_validation_json("phase1_ground_truth.json")

    # Build question -> gold answer mapping
    gt_map = {gt["question"]: gt["gold_answer"] for gt in ground_truth["questions"]}

    model_names = [n for _, _, n in MODELS]
    scores = {n: [] for n in model_names}

    for q in phase2["questions"]:
        gold_answer = gt_map.get(q["text"])
        if gold_answer is None:
            continue

        for model in model_names:
            ans = q["answers"].get(model, {})
            extracted = ans.get("extracted_answer") if isinstance(ans, dict) else None
            is_correct = numbers_match(extracted, gold_answer)
            scores[model].append(1 if is_correct else 0)

    # Calculate accuracy
    summary = {}
    for model, score_list in scores.items():
        correct = sum(score_list)
        total = len(score_list)
        summary[model] = {
            "accuracy": round(100 * correct / total, 1) if total else 0,
            "correct": correct,
            "total": total,
            "mean": round(10 * correct / total, 2) if total else 0,  # 0-10 scale for correlation
        }

    # Print rankings
    ranked = sorted(summary.items(), key=lambda x: (-x[1]["accuracy"], x[0]))
    print(f"\n  {'Model':<25} {'Accuracy':>10}")
    print(f"  {'-' * 35}")
    for model, stats in ranked:
        print(f"  {model:<25} {stats['accuracy']:>8.1f}% ({stats['correct']}/{stats['total']})")

    save_validation_json("phase4_GSM8K_scores.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 4,
        "judge_model": "N/A (exact match)",
        "summary": summary
    })
    print(f"\n{'=' * 60}")


# =============================================================================
# PHASE 5: Correlation Analysis
# =============================================================================

def phase5_correlation_analysis():
    """Correlate peer scores with ground truth math accuracy."""
    print(f"\n{'=' * 60}")
    print("  PHASE 5: Correlation Analysis")
    print(f"{'=' * 60}")

    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("  Error: scipy required. pip install scipy")
        return None

    truth_data = load_validation_json("phase4_GSM8K_scores.json")

    try:
        phase3_data = load_validation_json("phase3_rankings.json")
    except FileNotFoundError:
        print("  No Phase 3 data. Run Phase 3 first.")
        return None

    # Get peer scores
    evaluations = phase3_data.get("evaluations_by_mode", {}).get(
        "shuffle_blind", phase3_data.get("evaluations", {})
    )
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

    # Check for zero variance
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
        score = truth_means[truth_ranked[i]]
        tied = [truth_ranked[i]]
        j = i + 1
        while j < len(truth_ranked) and truth_means[truth_ranked[j]] == score:
            tied.append(truth_ranked[j])
            j += 1
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

    # Interpret correlation
    def interpret(r):
        ar = abs(r)
        if ar >= 0.8: return "strong"
        if ar >= 0.6: return "moderate"
        if ar >= 0.4: return "weak"
        return "none"

    interp = interpret(pearson_r)

    # Save markdown report
    num_q = truth_data['summary'][common[0]]['total']
    report = f"""# GSM8K Validation Report

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
        report += f"Peer evaluation **strongly correlates** with math accuracy (r={pearson_r:.3f})."
    elif pearson_r >= 0.5 and pearson_p < 0.05:
        report += f"Peer evaluation shows **moderate correlation** with math accuracy (r={pearson_r:.3f})."
    elif len(set(truth_arr)) == 1:
        report += f"**Cannot determine correlation** - all models achieved identical accuracy. Use more questions."
    else:
        report += f"Peer evaluation shows **weak/no correlation** with math accuracy (r={pearson_r:.3f})."

    report_file = GSM8K_DIR / f"GSM8K_validation_report_{VALIDATION_REVISION}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {report_file.name}")

    save_validation_json("GSM8K_analysis.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "correlation": {
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4)
        },
        "comparison": comparison,
    })

    print(f"\n{'=' * 60}")
    return {"pearson_r": pearson_r, "spearman_r": spearman_r}


# =============================================================================
# RUN ALL PHASES
# =============================================================================

async def run_all_phases(num_questions: int = 50):
    """Run complete validation workflow."""
    print(f"\n{'#' * 60}")
    print(f"  GSM8K VALIDATION")
    print(f"{'#' * 60}")
    print(f"  Models: {len(MODELS)} | Questions: {num_questions}")
    print(f"  Difficulty: {get_difficulties_display()}")
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
    for phase, fn in [(5, "GSM8K_analysis"), (4, "phase4_GSM8K_scores"),
                      (3, "phase3_rankings"), (2, "phase2_answers"), (1, "phase1_questions")]:
        if (GSM8K_DIR / f"{fn}_{VALIDATION_REVISION}.json").exists():
            return phase
    return 0


def show_menu():
    print("\n" + "=" * 50)
    print("  GSM8K Validation (Math Accuracy)")
    print("=" * 50)
    print(f"  Progress: Phase {get_last_completed_phase()}/5")
    print(f"  Models: {len(MODELS)} | Questions: {NUM_QUESTIONS}")
    print(f"  Difficulty: {get_difficulties_display()}")
    print(f"""
  [1-5] Run Phase 1-5
  [A] Run ALL    [N] Set questions ({NUM_QUESTIONS})
  [D] Difficulty [R] View report [Q] Quit
""")
    return input("  > ").strip().upper()


def interactive_menu():
    while True:
        choice = show_menu()
        if choice == "1": phase1_generate(NUM_QUESTIONS)
        elif choice == "2": asyncio.run(phase2_answer())
        elif choice == "3": asyncio.run(phase3_evaluate())
        elif choice == "4": phase4_ground_truth_score()
        elif choice == "5": phase5_correlation_analysis()
        elif choice == "A": asyncio.run(run_all_phases(NUM_QUESTIONS))
        elif choice == "N":
            try:
                n = int(input("  Questions (1-500): "))
                if 1 <= n <= 500: set_num_questions(n)
            except ValueError: pass
        elif choice == "D":
            print(f"  Current: {get_difficulties_display()}")
            print("  Options: easy, medium, hard (comma-separated)")
            print("  Example: easy,medium or hard or easy,hard")
            user_input = input("  > ").strip().lower()
            if user_input:
                levels = [d.strip() for d in user_input.split(",")]
                result = set_difficulties(levels)
                print(f"  Set to: {get_difficulties_display()}")
        elif choice == "R":
            rf = GSM8K_DIR / f"GSM8K_validation_report_{VALIDATION_REVISION}.md"
            if rf.exists(): print(rf.read_text())
            else: print("  No report yet")
        elif choice == "Q": break


def main():
    parser = argparse.ArgumentParser(description="GSM8K Validation (Math Accuracy)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Difficulty levels: easy,medium,hard (comma-separated)")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.num_questions: set_num_questions(args.num_questions)
    if args.difficulty:
        levels = [d.strip() for d in args.difficulty.split(",")]
        set_difficulties(levels)

    if args.phase == 1: phase1_generate(NUM_QUESTIONS)
    elif args.phase == 2: asyncio.run(phase2_answer())
    elif args.phase == 3: asyncio.run(phase3_evaluate())
    elif args.phase == 4: phase4_ground_truth_score()
    elif args.phase == 5: phase5_correlation_analysis()
    elif args.all: asyncio.run(run_all_phases(NUM_QUESTIONS))
    else: interactive_menu()


if __name__ == "__main__":
    main()
