"""
HLE Validation - Correlate peer rankings with Humanity's Last Exam accuracy

Humanity's Last Exam (HLE) is a frontier benchmark with 2,500 expert-level questions.
FIXES APPLIED:
  1. Phase 2 now allows Chain-of-Thought (CoT) reasoning (critical for HLE difficulty).
  2. Phase 2 enforces <answer> XML tags for final output extraction.
  3. Phase 4 Judge prompt updated to handle verbose reasoning and locate the final answer.

Usage:
    python validate_hle.py                       # Interactive menu
    python validate_hle.py --all                 # Run all phases
    python validate_hle.py --phase 2             # Re-run answering (recommended after fix)
    python validate_hle.py --num-questions 50    # Set question count
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
    extract_json,
)
from peerrank.providers import call_llm, clear_clients

load_dotenv()

# Exclude kimi (unreliable for HLE context)
MODELS = [(p, m, n) for p, m, n in ALL_MODELS if "kimi" not in n.lower()]

# =============================================================================
# PROMPTS
# =============================================================================

# Phase 2: Answering Prompt (Fixed to allow CoT)
HLE_ANSWER_PROMPT = """You are an expert answering a PhD-level academic question.

CRITICAL INSTRUCTIONS:
1. These questions are extremely difficult. Do NOT guess.
2. You MUST think step-by-step. Show your work, calculations, or code logic.
3. After your reasoning, output the final answer strictly inside <answer> tags.

Question:
{question}

Output Format:
[Step-by-step reasoning...]
<answer>
[Final Answer Only]
</answer>
"""

# Phase 3: Peer Eval Prompt
HLE_EVAL_PROMPT = """You are grading expert-level academic responses. These are graduate/PhD-level questions.

Scoring rubric (1-10 integer):
- 10: Correct answer with thorough, accurate reasoning.
- 8-9: Correct answer, reasoning mostly sound with minor gaps.
- 6-7: Partially correct OR correct answer but incomplete reasoning.
- 4-5: Wrong answer but demonstrates relevant knowledge.
- 1-3: Wrong answer, fundamentally flawed or irrelevant.

CRITICAL RULES:
1. These are EXPERT-LEVEL questions - correctness matters most.
2. Look for the final answer inside <answer> tags if present.
3. Do NOT penalize long reasoning; it is required for these problems.
4. If you're uncertain, give credit for demonstrating domain expertise.

Question:
{question}

Responses:
{responses}

Output format (STRICT):
- Return ONLY a single JSON object.
- Each entry: {{"score": <1-10>, "reason": "<brief justification>", "flags": []}}

Example: {{"{label_example}": {{"score": 9, "reason": "Correct calculation inside tags", "flags": []}}}}
"""

# Phase 4: Ground Truth Judge Prompt (Fixed to handle CoT)
JUDGE_PROMPT = """You are an expert academic judge. Determine if the model's answer is correct.

Question: {question}

Correct Answer: {correct_answer}

Model Response: {model_response}

Evaluation Instructions:
1. The model response likely contains step-by-step reasoning.
2. Look for the final conclusion, often wrapped in <answer>...</answer> tags.
3. Compare the semantic meaning of that final conclusion to the Correct Answer.
4. For math/science: Allow equivalent forms (e.g., 1/2 == 0.5) and minor rounding.
5. For code: The logic must be valid to produce the correct result.

Output ONLY a JSON object:
{{"correct": true/false, "reason": "<brief explanation>"}}
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

HLE_DIR = DATA_DIR / "HLE"  # All HLE files go here
VALIDATION_REVISION = "HLE"
NUM_QUESTIONS = 50

# Active subjects (empty = all)
SUBJECTS = []


def set_num_questions(n: int):
    global NUM_QUESTIONS
    NUM_QUESTIONS = n


def set_subjects(subjects: list[str]):
    """Set which subjects to include (empty = all)."""
    global SUBJECTS
    SUBJECTS.clear()
    SUBJECTS.extend([s.strip().lower() for s in subjects if s.strip()])
    return SUBJECTS


def get_subjects_display() -> str:
    """Get display string for current subject settings."""
    if not SUBJECTS:
        return "all"
    return "+".join(SUBJECTS[:3]) + ("..." if len(SUBJECTS) > 3 else "")


# =============================================================================
# FILE I/O
# =============================================================================

def load_validation_json(filename: str) -> dict:
    base, ext = filename.rsplit(".", 1)
    filepath = HLE_DIR / f"{base}_{VALIDATION_REVISION}.{ext}"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath.name} not found")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_validation_json(filename: str, data: dict):
    HLE_DIR.mkdir(parents=True, exist_ok=True)
    base, ext = filename.rsplit(".", 1)
    filepath = HLE_DIR / f"{base}_{VALIDATION_REVISION}.{ext}"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath.name}")


def progress_bar(completed: int, total: int, width: int = 40) -> str:
    if total == 0: return "[ ] 0%"
    pct = completed * 100 // total
    filled = pct * width // 100
    bar = "=" * filled + ">" + "." * (width - filled - 1) if filled < width else "=" * width
    return f"[{bar}] {pct:3}% ({completed}/{total})"


# =============================================================================
# CONFIDENCE INTERVAL HELPERS
# =============================================================================

def correlation_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute CI for Pearson r using Fisher z-transformation."""
    from scipy.stats import norm
    import math

    if abs(r) >= 1.0 or n < 4:
        return (r, r)

    z = 0.5 * math.log((1 + r) / (1 - r))  # Fisher z
    se = 1 / math.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)

    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    r_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
    r_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)

    return (round(r_lo, 4), round(r_hi, 4))


def wilson_ci(correct: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (accuracy)."""
    from scipy.stats import norm

    if total == 0:
        return (0.0, 0.0)

    p = correct / total
    z = norm.ppf(1 - alpha / 2)

    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom

    return (round(max(0, center - margin), 4), round(min(1, center + margin), 4))


def peer_score_ci(scores: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """CI for mean peer score using t-distribution."""
    from scipy.stats import t

    if len(scores) < 2:
        m = scores[0] if scores else 0
        return (m, m)

    n = len(scores)
    m = mean(scores)
    se = stdev(scores) / sqrt(n)
    t_crit = t.ppf(1 - alpha / 2, n - 1)

    return (round(m - t_crit * se, 2), round(m + t_crit * se, 2))


# =============================================================================
# PHASE 1: Load Questions from HLE
# =============================================================================

def phase1_generate(num_questions: int = 50):
    """Load questions from HLE dataset (requires HF_TOKEN)."""
    from datasets import load_dataset

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Load HLE Questions")
    print(f"{'=' * 60}")

    # Check for HF token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("\n  ERROR: HLE is a gated dataset. Set HF_TOKEN in .env file.")
        print("  1. Accept terms at: https://huggingface.co/datasets/cais/hle")
        print("  2. Get token from: https://huggingface.co/settings/tokens")
        print("  3. Add to .env: HF_TOKEN=your_token_here")
        return

    # Load HLE dataset
    try:
        dataset = load_dataset("cais/hle", split="test", token=token)
    except Exception as e:
        if "gated" in str(e).lower():
            print("\n  ERROR: Access denied. Request access at:")
            print("  https://huggingface.co/datasets/cais/hle")
            return
        raise

    print(f"  Dataset: {len(dataset)} questions available")
    print(f"  Columns: {dataset.column_names}")

    # Filter to text-only questions (skip multimodal for now)
    text_only = []
    by_subject = {}

    for row in dataset:
        # Skip questions with images (multimodal)
        if row.get("image"):
            continue

        question = row.get("question", "")
        answer = row.get("answer", "")
        subject = row.get("subject", "unknown")

        if not question or not answer:
            continue

        # Filter by subject if specified
        if SUBJECTS and subject.lower() not in SUBJECTS:
            continue

        item = {
            "id": row.get("id", ""),
            "question": question,
            "answer": answer,
            "subject": subject,
        }
        text_only.append(item)

        if subject not in by_subject:
            by_subject[subject] = []
        by_subject[subject].append(item)

    print(f"  Text-only questions: {len(text_only)}")
    print(f"  Subjects: {len(by_subject)}")

    # Sample questions (stratified by subject if possible)
    questions = []
    if len(text_only) <= num_questions:
        questions = text_only
    else:
        # Stratified sampling across subjects
        subjects_with_data = list(by_subject.keys())
        per_subject = num_questions // len(subjects_with_data) if subjects_with_data else 0
        remainder = num_questions % len(subjects_with_data) if subjects_with_data else 0

        for i, subject in enumerate(subjects_with_data):
            take = min(per_subject + (1 if i < remainder else 0), len(by_subject[subject]))
            if take > 0:
                questions.extend(random.sample(by_subject[subject], take))

        # If we don't have enough, fill randomly
        if len(questions) < num_questions:
            remaining = [q for q in text_only if q not in questions]
            need = min(num_questions - len(questions), len(remaining))
            questions.extend(random.sample(remaining, need))

    random.shuffle(questions)
    questions = questions[:num_questions]

    # Save phase1 questions (without answers for model answering)
    output = {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "questions_by_model": {"HLE": [
            {"id": q["id"], "question": q["question"], "subject": q["subject"]}
            for q in questions
        ]}
    }
    save_validation_json("phase1_questions.json", output)

    # Save ground truth (with answers)
    ground_truth = {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "questions": questions
    }
    save_validation_json("phase1_ground_truth.json", ground_truth)

    # Log subject statistics
    print(f"\n  Selected {len(questions)} questions")
    print(f"  {'-' * 50}")
    print(f"  {'Subject':<30} {'Selected':>10} {'Available':>10}")
    print(f"  {'-' * 50}")

    selected_counts = {}
    for q in questions:
        s = q["subject"]
        selected_counts[s] = selected_counts.get(s, 0) + 1

    # Show top subjects
    top_subjects = sorted(by_subject.items(), key=lambda x: -len(x[1]))[:10]
    for subject, items in top_subjects:
        selected = selected_counts.get(subject, 0)
        short_sub = subject[:28] + ".." if len(subject) > 30 else subject
        print(f"  {short_sub:<30} {selected:>10} {len(items):>10}")

    if len(by_subject) > 10:
        print(f"  ... and {len(by_subject) - 10} more subjects")

    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':<30} {len(questions):>10} {len(text_only):>10}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 2: Answer Questions
# =============================================================================

async def phase2_answer():
    """Models answer HLE questions using Chain-of-Thought."""
    set_revision(VALIDATION_REVISION)

    print(f"\n{'=' * 60}")
    print("  PHASE 2: Answer Questions (CoT + XML Tags)")
    print(f"{'=' * 60}")

    phase_start = time.time()
    questions = load_validation_json("phase1_questions.json")["questions_by_model"]["HLE"]
    model_names = [n for _, _, n in MODELS]

    # Check for existing progress to resume
    output_questions = []
    start_idx = 0
    try:
        existing = load_validation_json("phase2_answers.json")
        if existing and "questions" in existing:
            saved_count = len(existing["questions"])
            if saved_count < len(questions):
                if saved_count > 0 and existing["questions"][0]["text"] == questions[0]["question"]:
                    output_questions = existing["questions"]
                    start_idx = saved_count
                    print(f"  Resuming from question {start_idx + 1}/{len(questions)}")
    except Exception:
        pass

    total = len(questions) * len(model_names)
    completed = start_idx * len(model_names)
    lock = asyncio.Lock()

    print(f"  Models: {len(model_names)} | Questions: {len(questions)} | Total: {total}")

    SAVE_INTERVAL = 5

    async def answer_one(provider, model_id, model_name, question, q_idx, semaphore):
        nonlocal completed

        prompt = HLE_ANSWER_PROMPT.format(question=question["question"])

        start_time = time.time()
        try:
            async with semaphore:
                response, duration, in_tok, out_tok, _ = await call_llm(
                    provider, model_id, prompt, max_tokens=8000, timeout=300,
                    temperature=0.2, use_web_search=False
                )

            result = {
                "text": response.strip(),
                "duration": duration,
                "input_tokens": in_tok,
                "output_tokens": out_tok
            }
        except Exception as e:
            result = {
                "text": f"Error: {e}",
                "duration": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }

        elapsed = time.time() - start_time
        async with lock:
            completed += 1
            print(f"\r  {progress_bar(completed, total)} | Q{q_idx+1} {model_name[:15]:<15} ({elapsed:.1f}s)    ", end="", flush=True)
        return model_name, result

    # Process remaining questions
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p, _, _ in MODELS}

    for q_idx, question in enumerate(questions[start_idx:], start=start_idx):
        tasks = [answer_one(p, m, n, question, q_idx, semaphores[p]) for p, m, n in MODELS]
        results = await asyncio.gather(*tasks)
        output_questions.append({
            "text": question["question"],
            "id": question.get("id", ""),
            "subject": question.get("subject", "unknown"),
            "answers": {name: result for name, result in results}
        })

        # Log question and all answers
        print(f"\n\n  {'─' * 70}")
        print(f"  Q{q_idx+1}: {question['question'][:100]}{'...' if len(question['question']) > 100 else ''}")
        print(f"  Subject: {question.get('subject', 'unknown')}")
        print(f"  {'─' * 70}")
        for model_name, result in results:
            # Show just the tag content if possible, else the first 150 chars
            txt = result["text"]
            if "<answer>" in txt:
                display = "TAGGED: " + txt.split("<answer>")[-1].split("</answer>")[0][:100]
            else:
                display = txt[:150].replace("\n", " ")
            print(f"  {model_name:<20}: {display}")

        # Incremental save
        if (q_idx + 1) % SAVE_INTERVAL == 0 or q_idx == len(questions) - 1:
            save_validation_json("phase2_answers.json", {
                "revision": VALIDATION_REVISION,
                "timestamp": datetime.now().isoformat(),
                "phase": 2,
                "duration_seconds": round(time.time() - phase_start, 2),
                "questions": output_questions
            })
            print(f"\n  [Saved progress: {len(output_questions)}/{len(questions)} questions]")

    print()
    print(f"  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 3: Peer Evaluation
# =============================================================================

def _format_hle_responses(question: dict, shuffle: bool, blind: bool, seed: int | None) -> tuple[str, dict]:
    """Format responses for HLE evaluation."""
    from peerrank_phase3 import format_responses_for_eval
    return format_responses_for_eval(question, shuffle, blind, seed)


async def phase3_evaluate():
    """Run peer evaluation for HLE questions."""
    from peerrank.config import MAX_TOKENS_EVAL, TEMPERATURE_EVAL

    set_revision(VALIDATION_REVISION)

    phase_start = time.time()
    seed = get_bias_test_config()["seed"]
    questions = load_validation_json("phase2_answers.json")["questions"]

    print(f"\n{'=' * 60}")
    print("  PHASE 3: Peer Evaluation")
    print(f"{'=' * 60}")

    # Check for existing progress to resume
    evaluations = {n: {} for _, _, n in MODELS}
    timing = {n: [] for _, _, n in MODELS}
    start_idx = 0

    try:
        existing = load_validation_json("phase3_rankings.json")
        if existing and "evaluations_by_mode" in existing and "_progress_idx" in existing:
            saved_idx = existing["_progress_idx"]
            if saved_idx < len(questions):
                evaluations = existing["evaluations_by_mode"].get("shuffle_blind", {})
                timing = {n: [] for n in evaluations}
                start_idx = saved_idx
                print(f"  Resuming from question {start_idx + 1}/{len(questions)}")
    except Exception:
        pass

    total = len(MODELS) * len(questions)
    completed = start_idx * len(MODELS)
    lock = asyncio.Lock()
    SAVE_INTERVAL = 10

    print(f"  Models: {len(MODELS)} | Questions: {len(questions)} | Total: {total}")

    async def evaluate(provider, model_id, name, q, q_idx, semaphore):
        nonlocal completed

        responses_text, label_to_model = _format_hle_responses(q, shuffle=True, blind=True, seed=seed)
        label_example = "Response A"

        prompt = HLE_EVAL_PROMPT.format(
            question=q["text"],
            responses=responses_text,
            label_example=label_example
        )

        start_time = time.time()
        try:
            async with semaphore:
                response, duration, _, _, _ = await call_llm(
                    provider, model_id, prompt,
                    max_tokens=MAX_TOKENS_EVAL,
                    use_web_search=False,
                    temperature=TEMPERATURE_EVAL
                )
            scores = extract_json(response)
            if scores and isinstance(scores, dict):
                remapped = {}
                for label, score_data in scores.items():
                    model_name = label_to_model.get(label)
                    if model_name:
                        remapped[model_name] = score_data
                    else:
                        for full_label, model in label_to_model.items():
                            if label in full_label or full_label in label:
                                remapped[model] = score_data
                                break
                result = (name, q_idx, remapped, duration)
            else:
                result = (name, q_idx, {}, duration)
        except Exception as e:
            print(f"\n      [ERROR] {name}: {e}", flush=True)
            result = (name, q_idx, {}, 0)

        elapsed = time.time() - start_time
        async with lock:
            completed += 1
            print(f"\r  {progress_bar(completed, total)} | Q{q_idx+1} {name[:15]:<15} ({elapsed:.1f}s)    ", end="", flush=True)

        return result

    # Process remaining questions
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p, _, _ in MODELS}

    for q_idx, q in enumerate(questions[start_idx:], start=start_idx):
        tasks = [evaluate(p, m, n, q, q_idx, semaphores[p]) for p, m, n in MODELS]
        results = await asyncio.gather(*tasks)

        for name, idx, scores, duration in results:
            evaluations[name][str(idx)] = scores
            timing[name].append(duration)

        # Incremental save
        if (q_idx + 1) % SAVE_INTERVAL == 0 or q_idx == len(questions) - 1:
            save_validation_json("phase3_rankings.json", {
                "revision": VALIDATION_REVISION,
                "timestamp": datetime.now().isoformat(),
                "phase": 3,
                "duration_seconds": round(time.time() - phase_start, 2),
                "evaluations_by_mode": {"shuffle_blind": evaluations},
                "timing_stats": calculate_timing_stats(timing),
                "_progress_idx": q_idx + 1,
            })
            print(f"\n  [Saved progress: {q_idx + 1}/{len(questions)} questions]")

    print()
    print(f"  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 4: Ground Truth Scoring (LLM Judge)
# =============================================================================

async def phase4_ground_truth_score():
    """Score accuracy using LLM judge (semantic equivalence check)."""
    print(f"\n{'=' * 60}")
    print("  PHASE 4: Ground Truth Accuracy (LLM Judge)")
    print(f"{'=' * 60}")

    phase_start = time.time()
    phase2 = load_validation_json("phase2_answers.json")
    ground_truth = load_validation_json("phase1_ground_truth.json")

    # Build question -> correct answer mapping
    gt_map = {gt["question"]: gt["answer"] for gt in ground_truth["questions"]}

    model_names = [n for _, _, n in MODELS]
    scores = {n: [] for n in model_names}

    # Use GPT-5.2 as judge
    judge_provider = "openai"
    judge_model = "gpt-5.2"
    print(f"  Judge: {judge_model}")

    total = len(phase2["questions"]) * len(model_names)
    completed = 0
    lock = asyncio.Lock()

    async def judge_one(question_text, correct_answer, model_name, model_response, semaphore):
        nonlocal completed

        prompt = JUDGE_PROMPT.format(
            question=question_text,
            correct_answer=correct_answer,
            model_response=model_response
        )

        try:
            async with semaphore:
                response, _, _, _, _ = await call_llm(
                    judge_provider, judge_model, prompt,
                    max_tokens=1000, timeout=60, temperature=0
                )
            result = extract_json(response)
            is_correct = result.get("correct", False) if result else False
        except Exception:
            is_correct = False

        async with lock:
            completed += 1
            print(f"\r  {progress_bar(completed, total)}    ", end="", flush=True)

        return model_name, 1 if is_correct else 0

    semaphore = asyncio.Semaphore(8)

    for q in phase2["questions"]:
        correct_answer = gt_map.get(q["text"])
        if correct_answer is None:
            continue

        tasks = []
        for model in model_names:
            ans = q["answers"].get(model, {})
            model_response = ans.get("text", "") if isinstance(ans, dict) else ""
            
            # Skip scoring if the answer is completely missing or an error
            if not model_response or model_response.startswith("Error:") or "FinishReason" in model_response:
                scores[model].append(0)
            else:
                tasks.append(judge_one(q["text"], correct_answer, model, model_response, semaphore))

        if tasks:
            results = await asyncio.gather(*tasks)
            for model_name, score in results:
                scores[model_name].append(score)

    print()

    # Calculate accuracy
    summary = {}
    for model, score_list in scores.items():
        correct = sum(score_list)
        total = len(score_list)
        summary[model] = {
            "accuracy": round(100 * correct / total, 1) if total else 0,
            "correct": correct,
            "total": total,
            "mean": round(10 * correct / total, 2) if total else 0,  # 0-10 scale
        }

    # Print rankings
    ranked = sorted(summary.items(), key=lambda x: (-x[1]["accuracy"], x[0]))
    print(f"\n  {'Model':<25} {'Accuracy':>10}")
    print(f"  {'-' * 35}")
    for model, stats in ranked:
        print(f"  {model:<25} {stats['accuracy']:>8.1f}% ({stats['correct']}/{stats['total']})")

    save_validation_json("phase4_HLE_scores.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 4,
        "duration_seconds": round(time.time() - phase_start, 2),
        "judge_model": judge_model,
        "summary": summary
    })
    print(f"\n  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 5: Correlation Analysis
# =============================================================================

def phase5_correlation_analysis():
    """Correlate peer scores with ground truth HLE accuracy."""
    print(f"\n{'=' * 60}")
    print("  PHASE 5: Correlation Analysis")
    print(f"{'=' * 60}")

    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("  Error: scipy required. pip install scipy")
        return None

    truth_data = load_validation_json("phase4_HLE_scores.json")

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
    truth_means = {m: stats["mean"] for m, stats in truth_data["summary"].items() if stats["total"] > 0}

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

    # Build report with CIs
    comparison = []
    for m in common:
        peer_scores_list = scores_result["peer_scores"].get(m, [])
        p_ci = peer_score_ci(peer_scores_list) if len(peer_scores_list) >= 2 else (peer_means[m], peer_means[m])

        truth_stats = truth_data["summary"].get(m, {})
        correct = truth_stats.get("correct", 0)
        total = truth_stats.get("total", 0)
        t_ci_pct = wilson_ci(correct, total)
        t_ci = (round(t_ci_pct[0] * 10, 2), round(t_ci_pct[1] * 10, 2))

        comparison.append({
            "model": m,
            "peer_score": round(peer_means[m], 2),
            "peer_ci": p_ci,
            "truth_score": round(truth_means[m], 2),
            "truth_ci": t_ci,
            "peer_rank": peer_ranks[m],
            "truth_rank": truth_ranks[m],
            "rank_diff": peer_ranks[m] - truth_ranks[m],
            "accuracy": truth_stats.get("accuracy", 0),
        })
    comparison.sort(key=lambda x: x["peer_rank"])

    # Print comparison table
    print(f"\n  {'Model':<22} {'Peer':>6} {'95% CI':>14} {'Truth':>6} {'95% CI':>14}")
    print(f"  {'-' * 64}")
    for row in comparison:
        p_ci_str = f"[{row['peer_ci'][0]:.2f},{row['peer_ci'][1]:.2f}]"
        t_ci_str = f"[{row['truth_ci'][0]:.2f},{row['truth_ci'][1]:.2f}]"
        print(f"  {row['model']:<22} {row['peer_score']:>6.2f} {p_ci_str:>14} {row['truth_score']:>6.2f} {t_ci_str:>14}")

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
    report = f"""# HLE Validation Report

Revision: {VALIDATION_REVISION}
Models:   {len(common)}
Questions: {num_q}
Judge: {truth_data.get('judge_model', 'gpt-5.2')}

## Correlation

  Metric       Value    95% CI              p-value   Interpretation
  ----------   ------   -----------------   -------   --------------
  Pearson r    {pearson_r:>6.4f}   [{pearson_ci[0]:.3f}, {pearson_ci[1]:.3f}]      {pearson_p:>7.4f}   {interp}
  Spearman     {spearman_r:>6.4f}   -                   {spearman_p:>7.4f}   {interpret(spearman_r)}

## Score Comparison

  Rank  Model                      Peer   Peer 95% CI       Truth  Truth 95% CI    Acc%
  ----  -------------------------  -----  ----------------  -----  ----------------  ----
"""
    for row in comparison:
        p_ci = f"[{row['peer_ci'][0]:.2f}, {row['peer_ci'][1]:.2f}]"
        t_ci = f"[{row['truth_ci'][0]:.2f}, {row['truth_ci'][1]:.2f}]"
        report += f"  {row['peer_rank']:>4}  {row['model']:<25}  {row['peer_score']:>5.2f}  {p_ci:<16}  {row['truth_score']:>5.2f}  {t_ci:<16}  {row['accuracy']:.1f}\n"

    report += f"\n## Conclusion\n\n"
    if pearson_r >= 0.7 and pearson_p < 0.05:
        report += f"Peer evaluation **strongly correlates** with HLE accuracy (r={pearson_r:.3f})."
    elif pearson_r >= 0.5 and pearson_p < 0.05:
        report += f"Peer evaluation shows **moderate correlation** with HLE accuracy (r={pearson_r:.3f})."
    elif len(set(truth_arr)) == 1:
        report += f"**Cannot determine correlation** - all models achieved identical accuracy. Use more questions."
    else:
        report += f"Peer evaluation shows **weak/no correlation** with HLE accuracy (r={pearson_r:.3f})."

    report_file = HLE_DIR / f"HLE_validation_report_{VALIDATION_REVISION}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {report_file.name}")

    save_validation_json("HLE_analysis.json", {
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
    """Run complete validation workflow."""
    print(f"\n{'#' * 60}")
    print(f"  HLE VALIDATION (Humanity's Last Exam)")
    print(f"{'#' * 60}")
    print(f"  Models: {len(MODELS)} | Questions: {num_questions}")
    print(f"  Subjects: {get_subjects_display()}")
    print(f"{'#' * 60}\n")

    start = time.time()
    phase1_generate(num_questions)
    await phase2_answer()
    await phase3_evaluate()
    await phase4_ground_truth_score()
    phase5_correlation_analysis()

    print(f"\n{'#' * 60}")
    print(f"  COMPLETE in {format_duration(time.time() - start)}")
    print(f"{'#' * 60}\n")


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def get_last_completed_phase() -> int:
    for phase, fn in [(5, "HLE_analysis"), (4, "phase4_HLE_scores"),
                      (3, "phase3_rankings"), (2, "phase2_answers"), (1, "phase1_questions")]:
        if (HLE_DIR / f"{fn}_{VALIDATION_REVISION}.json").exists():
            return phase
    return 0


def show_menu():
    print("\n" + "=" * 50)
    print("  HLE Validation (Humanity's Last Exam)")
    print("=" * 50)
    print(f"  Progress: Phase {get_last_completed_phase()}/5")
    print(f"  Models: {len(MODELS)} | Questions: {NUM_QUESTIONS}")
    print(f"  Subjects: {get_subjects_display()}")
    print(f"""
  [1-5] Run Phase 1-5
  [A] Run ALL    [N] Set questions ({NUM_QUESTIONS})
  [S] Subjects   [R] View report   [Q] Quit
""")
    return input("  > ").strip().upper()


def interactive_menu():
    while True:
        choice = show_menu()
        if choice == "1": phase1_generate(NUM_QUESTIONS)
        elif choice == "2": clear_clients(); asyncio.run(phase2_answer())
        elif choice == "3": clear_clients(); asyncio.run(phase3_evaluate())
        elif choice == "4": clear_clients(); asyncio.run(phase4_ground_truth_score())
        elif choice == "5": phase5_correlation_analysis()
        elif choice == "A": clear_clients(); asyncio.run(run_all_phases(NUM_QUESTIONS))
        elif choice == "N":
            try:
                n = int(input("  Questions (1-2500): "))
                if 1 <= n <= 2500: set_num_questions(n)
            except ValueError: pass
        elif choice == "S":
            print(f"  Current: {get_subjects_display()}")
            print("  Enter comma-separated subjects (empty = all)")
            user_input = input("  > ").strip()
            if user_input:
                subjects = [s.strip() for s in user_input.split(",")]
                result = set_subjects(subjects)
                print(f"  Set to: {get_subjects_display()}")
            else:
                set_subjects([])
                print("  Set to: all")
        elif choice == "R":
            rf = HLE_DIR / f"HLE_validation_report_{VALIDATION_REVISION}.md"
            if rf.exists(): print(rf.read_text(encoding="utf-8"))
            else: print("  No report yet")
        elif choice == "Q": break


def main():
    parser = argparse.ArgumentParser(description="HLE Validation (Humanity's Last Exam)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--subjects", type=str, default=None,
                        help="Subject filter (comma-separated)")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.num_questions: set_num_questions(args.num_questions)
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
        set_subjects(subjects)

    if args.phase == 1: phase1_generate(NUM_QUESTIONS)
    elif args.phase == 2: clear_clients(); asyncio.run(phase2_answer())
    elif args.phase == 3: clear_clients(); asyncio.run(phase3_evaluate())
    elif args.phase == 4: clear_clients(); asyncio.run(phase4_ground_truth_score())
    elif args.phase == 5: phase5_correlation_analysis()
    elif args.all: clear_clients(); asyncio.run(run_all_phases(NUM_QUESTIONS))
    else: interactive_menu()


if __name__ == "__main__":
    main()