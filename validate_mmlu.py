"""
MMLU Validation - Comprehensive LLM evaluation on Massive Multitask Language Understanding

Features:
    - Choose participating LLMs from 12 available models
    - Choose judge LLM for evaluation
    - Choose number of questions (1-14042)
    - Choose from 57 subject categories
    - 5-phase pipeline with correlation analysis

Usage:
    python validate_mmlu.py                    # Interactive menu
    python validate_mmlu.py --all              # Run all phases
    python validate_mmlu.py --phase 1-5        # Run specific phase
    python validate_mmlu.py --num-questions 50
    python validate_mmlu.py --subjects math,physics,chemistry
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from statistics import mean, stdev
from math import sqrt
import random

from dotenv import load_dotenv

from peerrank.config import (
    MODELS as ALL_MODELS, DATA_DIR, format_duration,
    set_revision, calculate_scores_from_evaluations,
    PROVIDER_CONCURRENCY, calculate_timing_stats, get_bias_test_config,
    extract_json, TOKEN_COSTS,
)
from peerrank.providers import call_llm, clear_clients

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

MMLU_DIR = DATA_DIR / "MMLU"
VALIDATION_REVISION = "MMLU"
NUM_QUESTIONS = 50
LETTERS = ["A", "B", "C", "D"]

# Active models (modifiable via menu)
MODELS = list(ALL_MODELS)
EXCLUDED_MODELS = set()

# Judge model
JUDGE_PROVIDER = "openai"
JUDGE_MODEL_ID = "gpt-5.2"
JUDGE_NAME = "gpt-5.2"

# Active subjects (empty = all)
SUBJECTS = []

# All MMLU subjects (57 total)
ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions"
]

# Subject categories for grouping
SUBJECT_CATEGORIES = {
    "STEM": ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
             "college_computer_science", "college_mathematics", "college_physics", "computer_security",
             "conceptual_physics", "electrical_engineering", "elementary_mathematics", "machine_learning",
             "high_school_biology", "high_school_chemistry", "high_school_computer_science",
             "high_school_mathematics", "high_school_physics", "high_school_statistics", "virology"],
    "Humanities": ["high_school_european_history", "high_school_us_history", "high_school_world_history",
                   "philosophy", "prehistory", "world_religions", "logical_fallacies", "formal_logic"],
    "Social Sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics",
                        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
                        "human_sexuality", "public_relations", "security_studies", "sociology", "us_foreign_policy"],
    "Professional": ["business_ethics", "clinical_knowledge", "college_medicine", "management", "marketing",
                     "medical_genetics", "professional_accounting", "professional_law", "professional_medicine",
                     "professional_psychology", "international_law", "jurisprudence"],
    "Other": ["global_facts", "human_aging", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition"]
}


def set_num_questions(n: int):
    global NUM_QUESTIONS
    NUM_QUESTIONS = n


def set_subjects(subjects: list[str]):
    global SUBJECTS
    SUBJECTS.clear()
    SUBJECTS.extend([s.strip().lower().replace(" ", "_") for s in subjects if s.strip()])
    return SUBJECTS


def set_judge(provider: str, model_id: str, name: str):
    global JUDGE_PROVIDER, JUDGE_MODEL_ID, JUDGE_NAME
    JUDGE_PROVIDER = provider
    JUDGE_MODEL_ID = model_id
    JUDGE_NAME = name


def toggle_model(model_name: str) -> bool:
    """Toggle a model's inclusion. Returns new state (True=included)."""
    global MODELS, EXCLUDED_MODELS
    if model_name in EXCLUDED_MODELS:
        EXCLUDED_MODELS.discard(model_name)
        MODELS = [(p, m, n) for p, m, n in ALL_MODELS if n not in EXCLUDED_MODELS]
        return True
    else:
        EXCLUDED_MODELS.add(model_name)
        MODELS = [(p, m, n) for p, m, n in ALL_MODELS if n not in EXCLUDED_MODELS]
        return False


def get_subjects_display() -> str:
    if not SUBJECTS:
        return "all (57)"
    return f"{len(SUBJECTS)} selected"


def get_models_display() -> str:
    return f"{len(MODELS)}/{len(ALL_MODELS)}"


# =============================================================================
# FILE I/O
# =============================================================================

def load_validation_json(filename: str) -> dict:
    base, ext = filename.rsplit(".", 1)
    filepath = MMLU_DIR / f"{base}_{VALIDATION_REVISION}.{ext}"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath.name} not found")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_validation_json(filename: str, data: dict):
    MMLU_DIR.mkdir(parents=True, exist_ok=True)
    base, ext = filename.rsplit(".", 1)
    filepath = MMLU_DIR / f"{base}_{VALIDATION_REVISION}.{ext}"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath.name}")


def progress_bar(completed: int, total: int, width: int = 40) -> str:
    if total == 0:
        return "[" + "." * width + "] 0%"
    pct = completed * 100 // total
    filled = pct * width // 100
    bar = "=" * filled + ">" + "." * (width - filled - 1) if filled < width else "=" * width
    return f"[{bar}] {pct:3}% ({completed}/{total})"


# =============================================================================
# CONFIDENCE INTERVAL HELPERS
# =============================================================================

def correlation_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import norm
    import math
    if abs(r) >= 1.0 or n < 4:
        return (r, r)
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    r_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
    r_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
    return (round(r_lo, 4), round(r_hi, 4))


def wilson_ci(correct: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import norm
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (round(max(0, center - margin), 4), round(min(1, center + margin), 4))


# =============================================================================
# PHASE 1: Load Questions from MMLU
# =============================================================================

def phase1_generate(num_questions: int = 50):
    """Load questions from MMLU dataset with stratified sampling by subject."""
    from datasets import load_dataset

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Load MMLU Questions")
    print(f"{'=' * 60}")

    # Load MMLU dataset
    print("  Loading dataset from HuggingFace...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    print(f"  Dataset: {len(dataset)} questions available")

    # Group by subject
    by_subject = {}
    for row in dataset:
        subject = row.get("subject", "unknown")

        # Filter by selected subjects if any
        if SUBJECTS and subject not in SUBJECTS:
            continue

        if subject not in by_subject:
            by_subject[subject] = []
        by_subject[subject].append({
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],  # 0-3 index
            "subject": subject,
        })

    print(f"  Subjects with data: {len(by_subject)}")

    # Stratified sampling
    questions = []
    subjects_with_data = list(by_subject.keys())

    if not subjects_with_data:
        print("  ERROR: No questions match the selected subjects")
        return

    total_available = sum(len(v) for v in by_subject.values())
    target = min(num_questions, total_available)

    # If fewer questions than subjects, randomly select which subjects to sample from
    if target < len(subjects_with_data):
        # Randomly pick 'target' subjects and take 1 question from each
        selected_subjects = random.sample(subjects_with_data, target)
        for subject in selected_subjects:
            questions.extend(random.sample(by_subject[subject], 1))
    else:
        # Distribute evenly across subjects
        per_subject = target // len(subjects_with_data)
        remainder = target % len(subjects_with_data)

        # Shuffle subjects so remainder distribution is random
        shuffled_subjects = subjects_with_data.copy()
        random.shuffle(shuffled_subjects)

        for i, subject in enumerate(shuffled_subjects):
            take = min(per_subject + (1 if i < remainder else 0), len(by_subject[subject]))
            if take > 0:
                questions.extend(random.sample(by_subject[subject], take))

    random.shuffle(questions)
    questions = questions[:num_questions]

    # Save phase1 questions
    output = {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "num_models": len(MODELS),
        "model_names": [n for _, _, n in MODELS],
        "questions_by_model": {"MMLU": [
            {"question": q["question"], "choices": q["choices"], "subject": q["subject"]}
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

    # Log subject statistics
    print(f"\n  Selected {len(questions)} questions from {len(set(q['subject'] for q in questions))} subjects")
    print(f"  {'-' * 55}")
    print(f"  {'Subject':<35} {'Selected':>8} {'Available':>10}")
    print(f"  {'-' * 55}")

    selected_counts = {}
    for q in questions:
        s = q["subject"]
        selected_counts[s] = selected_counts.get(s, 0) + 1

    # Show subjects with selections first, then top available
    subjects_with_selections = sorted(
        [(s, selected_counts.get(s, 0), len(by_subject[s])) for s in by_subject],
        key=lambda x: (-x[1], -x[2])  # Sort by selected desc, then available desc
    )

    shown = 0
    for subject, selected, available in subjects_with_selections:
        if shown >= 20 and selected == 0:
            break  # Stop after showing 20, but always show selected subjects
        short_sub = subject[:33] + ".." if len(subject) > 35 else subject
        marker = "*" if selected > 0 else " "
        print(f"  {marker}{short_sub:<34} {selected:>8} {available:>10}")
        shown += 1

    remaining = len(by_subject) - shown
    if remaining > 0:
        print(f"  ... and {remaining} more subjects (0 selected)")

    print(f"  {'-' * 55}")
    print(f"  {'TOTAL':<35} {len(questions):>8} {total_available:>10}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 2: Answer Questions (Multiple Choice)
# =============================================================================

async def phase2_answer():
    """Models answer MMLU multiple-choice questions."""
    set_revision(VALIDATION_REVISION)

    print(f"\n{'=' * 60}")
    print("  PHASE 2: Answer Questions (Multiple Choice)")
    print(f"{'=' * 60}")

    phase_start = time.time()
    questions = load_validation_json("phase1_questions.json")["questions_by_model"]["MMLU"]
    model_names = [n for _, _, n in MODELS]

    # Check for existing progress
    output_questions = []
    start_idx = 0
    try:
        existing = load_validation_json("phase2_answers.json")
        if existing and "questions" in existing:
            saved_count = len(existing["questions"])
            if saved_count < len(questions) and saved_count > 0:
                if existing["questions"][0]["text"] == questions[0]["question"]:
                    output_questions = existing["questions"]
                    start_idx = saved_count
                    print(f"  Resuming from question {start_idx + 1}/{len(questions)}")
    except Exception:
        pass

    total = len(questions) * len(model_names)
    completed = start_idx * len(model_names)
    lock = asyncio.Lock()

    print(f"  Models: {len(model_names)} | Questions: {len(questions)} | Total: {total}")

    SAVE_INTERVAL = 10

    async def answer_one(provider, model_id, model_name, question, q_idx, semaphore):
        nonlocal completed

        choices_text = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(question["choices"]))

        prompt = f"""Answer this multiple-choice question with ONLY the letter (A, B, C, or D).
No explanation. Just the letter.

Question: {question["question"]}

{choices_text}

Answer:"""

        start_time = time.time()
        try:
            async with semaphore:
                response, duration, in_tok, out_tok, _ = await call_llm(
                    provider, model_id, prompt, max_tokens=8192, timeout=60,
                    temperature=0, use_web_search=False
                )

            # Extract letter from response - try multiple patterns
            answer_text = response.strip()
            answer_letter = "?"

            # Pattern 1: Single letter response (most reliable)
            if len(answer_text) <= 3 and answer_text.upper().strip(".") in LETTERS:
                answer_letter = answer_text.upper().strip(".")
            else:
                # Pattern 2: Letter at end of response (common for reasoning models)
                import re
                # Match letter at end: "...the answer is C" or just "C" on last line
                end_match = re.search(r'\b([A-D])\s*\.?\s*$', answer_text, re.IGNORECASE)
                if end_match:
                    answer_letter = end_match.group(1).upper()
                else:
                    # Pattern 3: "Answer: X" or "Answer is X" pattern
                    answer_match = re.search(r'(?:answer|choice)[:\s]+([A-D])\b', answer_text, re.IGNORECASE)
                    if answer_match:
                        answer_letter = answer_match.group(1).upper()
                    else:
                        # Fallback: first letter found (original behavior)
                        answer_letter = next((c for c in answer_text.upper() if c in LETTERS), "?")

            result = {
                "text": response.strip(),
                "answer_letter": answer_letter,
                "duration": duration,
                "input_tokens": in_tok,
                "output_tokens": out_tok
            }
        except Exception as e:
            result = {
                "text": f"Error: {e}",
                "answer_letter": "?",
                "duration": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }

        elapsed = time.time() - start_time
        async with lock:
            completed += 1
            print(f"\r  {progress_bar(completed, total)} | Q{q_idx+1} {model_name[:15]:<15} ({elapsed:.1f}s)  ", end="", flush=True)
        return model_name, result

    # Process questions
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p, _, _ in MODELS}

    for q_idx, question in enumerate(questions[start_idx:], start=start_idx):
        tasks = [answer_one(p, m, n, question, q_idx, semaphores[p]) for p, m, n in MODELS]
        results = await asyncio.gather(*tasks)

        correct_letter = LETTERS[question.get("answer", 0)] if "answer" not in question else "?"
        # Load ground truth for correct answer
        try:
            gt = load_validation_json("phase1_ground_truth.json")
            for gtq in gt["questions"]:
                if gtq["question"] == question["question"]:
                    correct_letter = LETTERS[gtq["answer"]]
                    break
        except:
            pass

        output_questions.append({
            "text": question["question"],
            "choices": question["choices"],
            "subject": question.get("subject", "unknown"),
            "correct_letter": correct_letter,
            "answers": {name: result for name, result in results}
        })

        # Log answers
        print(f"\n  Q{q_idx+1} [{question.get('subject', 'unknown')[:20]}] Correct={correct_letter}")
        for model_name, result in results:
            status = "✓" if result["answer_letter"] == correct_letter else "✗"
            print(f"    {model_name:<20}: {result['answer_letter']} {status}")

        # Incremental save
        if (q_idx + 1) % SAVE_INTERVAL == 0 or q_idx == len(questions) - 1:
            save_validation_json("phase2_answers.json", {
                "revision": VALIDATION_REVISION,
                "timestamp": datetime.now().isoformat(),
                "phase": 2,
                "duration_seconds": round(time.time() - phase_start, 2),
                "questions": output_questions
            })

    print(f"\n  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 3: Peer Evaluation
# =============================================================================

EVAL_PROMPT = """You are evaluating multiple-choice answers. Score each response 1-10.

Scoring:
- 10: Correct letter chosen
- 5: Wrong letter but shows reasoning attempt
- 1: Wrong letter, no reasoning

Question: {question}

Choices:
{choices}

Correct Answer: {correct_letter}

Responses:
{responses}

Output ONLY a JSON object with scores for each response label:
{{"Response A": {{"score": N, "reason": "..."}}, "Response B": {{"score": N, "reason": "..."}}, ...}}
"""

async def phase3_evaluate():
    """Peer evaluation of MMLU answers."""
    from peerrank.config import MAX_TOKENS_EVAL, TEMPERATURE_EVAL

    set_revision(VALIDATION_REVISION)

    print(f"\n{'=' * 60}")
    print("  PHASE 3: Peer Evaluation")
    print(f"{'=' * 60}")

    phase_start = time.time()
    seed = get_bias_test_config()["seed"]
    questions = load_validation_json("phase2_answers.json")["questions"]
    model_names = [n for _, _, n in MODELS]

    evaluations = {n: {} for n in model_names}
    timing = {n: [] for n in model_names}

    total = len(MODELS) * len(questions)
    completed = 0
    lock = asyncio.Lock()

    print(f"  Evaluators: {len(MODELS)} | Questions: {len(questions)} | Total: {total}")

    async def evaluate(provider, model_id, name, q, q_idx, semaphore):
        nonlocal completed

        # Format responses with labels
        answers = q["answers"]
        response_items = list(answers.items())

        # Shuffle for blind evaluation
        rng = random.Random(seed + q_idx if seed else None)
        rng.shuffle(response_items)

        label_to_model = {}
        responses_text = ""
        for idx, (model_name, ans) in enumerate(response_items):
            label = f"Response {chr(65 + idx)}"
            label_to_model[label] = model_name
            answer_text = ans.get("text", "No response")[:100]
            responses_text += f"\n{label}: {answer_text}"

        choices_text = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(q["choices"]))

        prompt = EVAL_PROMPT.format(
            question=q["text"],
            choices=choices_text,
            correct_letter=q.get("correct_letter", "?"),
            responses=responses_text
        )

        start_time = time.time()
        try:
            async with semaphore:
                response, duration, _, _, _ = await call_llm(
                    provider, model_id, prompt,
                    max_tokens=32000, use_web_search=False, temperature=0
                )
            scores = extract_json(response)
            if scores and isinstance(scores, dict):
                remapped = {}
                for label, score_data in scores.items():
                    model_name = label_to_model.get(label)
                    if model_name and isinstance(score_data, dict):
                        remapped[model_name] = score_data
                result = (name, q_idx, remapped, duration)
            else:
                result = (name, q_idx, {}, duration)
        except Exception as e:
            result = (name, q_idx, {}, 0)

        elapsed = time.time() - start_time
        async with lock:
            completed += 1
            print(f"\r  {progress_bar(completed, total)} | Q{q_idx+1} {name[:15]:<15} ({elapsed:.1f}s)  ", end="", flush=True)
        return result

    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p, _, _ in MODELS}

    for q_idx, q in enumerate(questions):
        tasks = [evaluate(p, m, n, q, q_idx, semaphores[p]) for p, m, n in MODELS]
        results = await asyncio.gather(*tasks)

        for name, idx, scores, duration in results:
            evaluations[name][str(idx)] = scores
            timing[name].append(duration)

    save_validation_json("phase3_rankings.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 3,
        "duration_seconds": round(time.time() - phase_start, 2),
        "evaluations_by_mode": {"shuffle_blind": evaluations},
        "timing_stats": calculate_timing_stats(timing),
    })

    print(f"\n  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 4: Ground Truth Scoring
# =============================================================================

def phase4_ground_truth_score():
    """Score accuracy against ground truth (direct letter matching)."""
    print(f"\n{'=' * 60}")
    print("  PHASE 4: Ground Truth Accuracy")
    print(f"{'=' * 60}")

    phase2 = load_validation_json("phase2_answers.json")
    model_names = [n for _, _, n in MODELS]
    scores = {n: [] for n in model_names}

    for q in phase2["questions"]:
        correct_letter = q.get("correct_letter", "?")

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
            "correct": correct,
            "total": total,
            "mean": round(10 * correct / total, 2) if total else 0,
        }

    # Print rankings
    ranked = sorted(summary.items(), key=lambda x: (-x[1]["accuracy"], x[0]))
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'-' * 45}")
    for model, stats in ranked:
        print(f"  {model:<25} {stats['accuracy']:>8.1f}% {stats['correct']:>6}/{stats['total']}")

    save_validation_json("phase4_MMLU_scores.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 4,
        "judge_model": "N/A (direct match)",
        "summary": summary
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

    truth_data = load_validation_json("phase4_MMLU_scores.json")

    try:
        phase3_data = load_validation_json("phase3_rankings.json")
    except FileNotFoundError:
        print("  No Phase 3 data. Run Phase 3 first.")
        return None

    evaluations = phase3_data.get("evaluations_by_mode", {}).get(
        "shuffle_blind", phase3_data.get("evaluations", {})
    )
    model_names = [n for _, _, n in MODELS]
    scores_result = calculate_scores_from_evaluations(evaluations, model_names)

    peer_means = {m: mean(s) for m, s in scores_result["peer_scores"].items() if s}
    truth_means = {m: stats["mean"] for m, stats in truth_data["summary"].items() if stats["total"] > 0}

    common = sorted(set(peer_means) & set(truth_means))
    if len(common) < 3:
        print(f"  Need 3+ models. Found: {len(common)}")
        return None

    peer_arr = [peer_means[m] for m in common]
    truth_arr = [truth_means[m] for m in common]

    if len(set(truth_arr)) == 1:
        print(f"\n  WARNING: All truth scores identical")
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1
        pearson_ci = (0, 0)
    else:
        pearson_r, pearson_p = pearsonr(peer_arr, truth_arr)
        spearman_r, spearman_p = spearmanr(peer_arr, truth_arr)
        pearson_ci = correlation_ci(pearson_r, len(common))

    print(f"\n  Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f}) 95% CI [{pearson_ci[0]:.3f}, {pearson_ci[1]:.3f}]")
    print(f"  Spearman:   {spearman_r:.4f} (p={spearman_p:.4f})")

    # Build comparison
    peer_ranked = sorted(common, key=lambda m: -peer_means[m])
    truth_ranked = sorted(common, key=lambda m: -truth_means[m])
    peer_ranks = {m: i + 1 for i, m in enumerate(peer_ranked)}
    truth_ranks = {m: i + 1 for i, m in enumerate(truth_ranked)}

    comparison = []
    for m in common:
        comparison.append({
            "model": m,
            "peer_score": round(peer_means[m], 2),
            "truth_score": round(truth_means[m], 2),
            "peer_rank": peer_ranks[m],
            "truth_rank": truth_ranks[m],
            "accuracy": truth_data["summary"][m]["accuracy"],
        })
    comparison.sort(key=lambda x: x["peer_rank"])

    print(f"\n  {'Model':<22} {'Peer':>6} {'Truth':>6} {'Acc%':>8}")
    print(f"  {'-' * 45}")
    for row in comparison:
        print(f"  {row['model']:<22} {row['peer_score']:>6.2f} {row['truth_score']:>6.2f} {row['accuracy']:>7.1f}%")

    # Save report
    num_q = truth_data['summary'][common[0]]['total']
    report = f"""# MMLU Validation Report

Revision: {VALIDATION_REVISION}
Models: {len(common)}
Questions: {num_q}

## Correlation

| Metric | Value | p-value | 95% CI |
|--------|-------|---------|--------|
| Pearson r | {pearson_r:.4f} | {pearson_p:.4f} | [{pearson_ci[0]:.3f}, {pearson_ci[1]:.3f}] |
| Spearman | {spearman_r:.4f} | {spearman_p:.4f} | - |

## Model Rankings

| Rank | Model | Peer Score | Truth Score | Accuracy |
|------|-------|------------|-------------|----------|
"""
    for row in comparison:
        report += f"| {row['peer_rank']} | {row['model']} | {row['peer_score']:.2f} | {row['truth_score']:.2f} | {row['accuracy']:.1f}% |\n"

    report_file = MMLU_DIR / f"MMLU_validation_report_{VALIDATION_REVISION}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {report_file.name}")

    save_validation_json("MMLU_analysis.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "correlation": {
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "pearson_ci_95": list(pearson_ci),
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
    print(f"\n{'#' * 60}")
    print(f"  MMLU VALIDATION")
    print(f"{'#' * 60}")
    print(f"  Models: {len(MODELS)} | Questions: {num_questions}")
    print(f"  Subjects: {get_subjects_display()}")
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
    for phase, fn in [(5, "MMLU_analysis"), (4, "phase4_MMLU_scores"),
                      (3, "phase3_rankings"), (2, "phase2_answers"), (1, "phase1_questions")]:
        if (MMLU_DIR / f"{fn}_{VALIDATION_REVISION}.json").exists():
            return phase
    return 0


def show_menu():
    print("\n" + "=" * 60)
    print("  MMLU Validation (Massive Multitask Language Understanding)")
    print("=" * 60)
    print(f"  Progress: Phase {get_last_completed_phase()}/5")
    print(f"  Models: {get_models_display()} | Questions: {NUM_QUESTIONS}")
    print(f"  Subjects: {get_subjects_display()}")
    print(f"  Judge: {JUDGE_NAME}")
    print()
    print("  --- Run ---")
    print("  [1-5] Run Phase    [A] All    [R] View Report")
    print()
    print("  --- Setup ---")
    print("  [M] Models         [N] Questions    [S] Subjects")
    print("  [J] Judge          [C] Categories   [Q] Quit")
    print()
    return input("  > ").strip().upper()


def select_models_menu():
    """Interactive model selection."""
    while True:
        print("\n  --- Model Selection ---")
        for i, (p, m, n) in enumerate(ALL_MODELS, 1):
            status = "✓" if n not in EXCLUDED_MODELS else " "
            print(f"  [{status}] {i:2}. {n:<25} ({p})")
        print(f"\n  Selected: {len(MODELS)}/{len(ALL_MODELS)}")
        print("  Enter number to toggle, 'all' to select all, 'none' to clear, or 'done'")

        choice = input("  > ").strip().lower()
        if choice == "done" or choice == "":
            break
        elif choice == "all":
            EXCLUDED_MODELS.clear()
            MODELS.clear()
            MODELS.extend(ALL_MODELS)
        elif choice == "none":
            EXCLUDED_MODELS.clear()
            for _, _, n in ALL_MODELS:
                EXCLUDED_MODELS.add(n)
            MODELS.clear()
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(ALL_MODELS):
                    _, _, name = ALL_MODELS[idx]
                    toggle_model(name)
            except ValueError:
                pass


def select_judge_menu():
    """Interactive judge selection."""
    print("\n  --- Judge Selection ---")
    for i, (p, m, n) in enumerate(ALL_MODELS, 1):
        current = " (current)" if n == JUDGE_NAME else ""
        print(f"  {i:2}. {n:<25} ({p}){current}")

    choice = input("\n  Enter number: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(ALL_MODELS):
            p, m, n = ALL_MODELS[idx]
            set_judge(p, m, n)
            print(f"  Judge set to: {n}")
    except ValueError:
        pass


def select_subjects_menu():
    """Interactive subject selection."""
    print("\n  --- Subject Categories ---")
    print("  [1] STEM (20 subjects)")
    print("  [2] Humanities (8 subjects)")
    print("  [3] Social Sciences (11 subjects)")
    print("  [4] Professional (12 subjects)")
    print("  [5] Other (6 subjects)")
    print("  [A] All subjects")
    print("  [L] List all 57 subjects")
    print("  [X] Custom (comma-separated)")

    choice = input("\n  > ").strip().upper()

    if choice == "1":
        set_subjects(SUBJECT_CATEGORIES["STEM"])
    elif choice == "2":
        set_subjects(SUBJECT_CATEGORIES["Humanities"])
    elif choice == "3":
        set_subjects(SUBJECT_CATEGORIES["Social Sciences"])
    elif choice == "4":
        set_subjects(SUBJECT_CATEGORIES["Professional"])
    elif choice == "5":
        set_subjects(SUBJECT_CATEGORIES["Other"])
    elif choice == "A":
        set_subjects([])
    elif choice == "L":
        print("\n  All MMLU subjects:")
        for i, s in enumerate(ALL_SUBJECTS, 1):
            print(f"  {i:2}. {s}")
        input("\n  Press Enter to continue...")
    elif choice == "X":
        custom = input("  Enter subjects (comma-separated): ")
        subjects = [s.strip() for s in custom.split(",")]
        set_subjects(subjects)

    print(f"  Subjects: {get_subjects_display()}")


def interactive_menu():
    while True:
        choice = show_menu()

        if choice == "1":
            phase1_generate(NUM_QUESTIONS)
        elif choice == "2":
            clear_clients()
            asyncio.run(phase2_answer())
        elif choice == "3":
            clear_clients()
            asyncio.run(phase3_evaluate())
        elif choice == "4":
            phase4_ground_truth_score()
        elif choice == "5":
            phase5_correlation_analysis()
        elif choice == "A":
            clear_clients()
            asyncio.run(run_all_phases(NUM_QUESTIONS))
        elif choice == "M":
            select_models_menu()
        elif choice == "N":
            try:
                n = int(input("  Questions (1-14042): "))
                if 1 <= n <= 14042:
                    set_num_questions(n)
            except ValueError:
                pass
        elif choice == "S" or choice == "C":
            select_subjects_menu()
        elif choice == "J":
            select_judge_menu()
        elif choice == "R":
            rf = MMLU_DIR / f"MMLU_validation_report_{VALIDATION_REVISION}.md"
            if rf.exists():
                print(rf.read_text(encoding="utf-8"))
            else:
                print("  No report yet")
        elif choice == "Q":
            break


def main():
    parser = argparse.ArgumentParser(description="MMLU Validation")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.num_questions:
        set_num_questions(args.num_questions)
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
        set_subjects(subjects)

    if args.phase == 1:
        phase1_generate(NUM_QUESTIONS)
    elif args.phase == 2:
        clear_clients()
        asyncio.run(phase2_answer())
    elif args.phase == 3:
        clear_clients()
        asyncio.run(phase3_evaluate())
    elif args.phase == 4:
        phase4_ground_truth_score()
    elif args.phase == 5:
        phase5_correlation_analysis()
    elif args.all:
        clear_clients()
        asyncio.run(run_all_phases(NUM_QUESTIONS))
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
