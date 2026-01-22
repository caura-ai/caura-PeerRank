"""
phase3.py - Cross-Evaluation Rankings
"""

import asyncio
import random
import time
from datetime import datetime
from statistics import mean

from config import (
    MODELS, MAX_TOKENS_EVAL, TEMPERATURE_EVAL, BIAS_MODES, PROVIDER_CONCURRENCY,
    extract_json, save_json, load_json, format_duration,
    get_revision, calculate_timing_stats, get_bias_test_config,
)
from providers import call_llm

EVAL_PROMPT = """You are grading responses for a benchmark. Score EACH response independently.

DO NOT try to identify authorship. Ignore writing style and focus on quality.

Scoring rubric (overall 1-10 integer):
- 10: Correct + complete + well-justified; directly answers; no hallucinations.
- 7-9: Mostly correct; minor omissions/imprecision; reasoning mostly sound.
- 4-6: Mixed/partial correctness, unclear reasoning, or misses key constraints.
- 1-3: Mostly incorrect, misleading, evasive, or hallucinated/unsupported.

Priority rules:
- Prioritize correctness/faithfulness over eloquence.
- Penalize confident-sounding unsupported specifics (made-up numbers, names, dates, citations).
- Citations [1][2], source mentions, or "I searched…" phrasing are NEUTRAL - do not reward or penalize. Judge factual accuracy only.
- If the question is subjective/creative: score instruction-following, coherence, and usefulness; do not mark “incorrect” unless it violates constraints or is nonsensical.

Calibration / score discipline:
- Use the full range when justified; avoid clustering 7-8.
- If there are 3+ responses and quality differs, use at least 3 distinct scores.

Question:
{question}

Responses:
{responses}

Output format (STRICT):
- Return ONLY a single JSON object (no markdown, no extra text).
- You MUST include an entry for EVERY label present in Responses, exactly once.
- Each entry MUST contain keys: "score", "reason", "flags" (no other keys).
- "score" MUST be an integer 1–10.
- "reason" MUST be 8–20 words and cite a specific strength or flaw.
- "flags" MUST be an array using only:
  "hallucination", "unsupported_specifics", "evasive", "incorrect",
  "good_uncertainty", "clear_correct"
  Use [] if none apply.

Example:
{{"{label_example}": {{"score": 8, "reason": "Correct core claim, minor omission on edge case; clear and grounded.", "flags": ["clear_correct"]}}}}
"""

# Labels for blind evaluation (A-Z)
BLIND_LABELS = [chr(ord('A') + i) for i in range(26)]


def format_responses_for_eval(question: dict, shuffle: bool, blind: bool, seed: int | None) -> tuple[str, dict]:
    """
    Format responses for evaluation with optional shuffling and blinding.

    Returns:
        (formatted_text, label_to_model_map)

    label_to_model_map maps the label used in prompt -> actual model name
    For non-blind: {"gpt-5.2": "gpt-5.2", ...}
    For blind: {"Response A": "gpt-5.2", "Response B": "claude-opus-4-5", ...}
    """
    model_names = [m[2] for m in MODELS]
    answers = question.get("answers", {})

    # Create list of (model_name, answer) pairs - extract text from answer object
    pairs = [(m, answers.get(m, {}).get("text", "N/A")) for m in model_names]

    # Shuffle if requested (use question text + seed for reproducible per-question shuffle)
    if shuffle:
        rng = random.Random()
        if seed is not None:
            rng.seed(hash(question["text"]) + seed)
        else:
            rng.seed(hash(question["text"]) + random.randint(0, 2**32))
        rng.shuffle(pairs)

    # Build formatted text and mapping
    label_to_model = {}
    lines = []

    for i, (model_name, answer) in enumerate(pairs):
        if blind:
            label = f"Response {BLIND_LABELS[i]}"
        else:
            label = model_name

        label_to_model[label] = model_name
        lines.append(f"--- {label} ---\n{answer}\n")

    return "\n".join(lines), label_to_model


def remap_scores_to_models(scores: dict, label_to_model: dict) -> dict:
    """Convert scores keyed by label back to scores keyed by actual model name."""
    remapped = {}
    for label, score_data in scores.items():
        model_name = label_to_model.get(label)
        if model_name:
            remapped[model_name] = score_data
        else:
            # Try fuzzy match for labels like "A" instead of "Response A"
            for full_label, model in label_to_model.items():
                if label in full_label or full_label in label:
                    remapped[model] = score_data
                    break
    return remapped


# BIAS_MODES imported from config.py


async def _run_evaluation_pass(questions: list, shuffle: bool, blind: bool, seed: int | None, mode_name: str) -> tuple[dict, dict]:
    """Run a single evaluation pass with specified bias settings."""
    evaluations = {n: {} for _, _, n in MODELS}
    timing = {n: [] for _, _, n in MODELS}

    async def evaluate(provider, model_id, name, q):
        responses_text, label_to_model = format_responses_for_eval(q, shuffle, blind, seed)
        label_example = "Response A" if blind else list(label_to_model.keys())[0]
        prompt = EVAL_PROMPT.format(
            question=q["text"],
            responses=responses_text,
            label_example=label_example
        )

        try:
            response, duration, _, _ = await call_llm(provider, model_id, prompt, max_tokens=MAX_TOKENS_EVAL, use_web_search=False, temperature=TEMPERATURE_EVAL)
            scores = extract_json(response)
            if scores and isinstance(scores, dict):
                remapped_scores = remap_scores_to_models(scores, label_to_model)
                return (name, q["text"], remapped_scores, duration)
            return (name, q["text"], {}, duration)
        except Exception as e:
            q_idx = questions.index(q) + 1 if q in questions else "?"
            print(f"      [ERROR] {name} Q#{q_idx}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return (name, q["text"], {}, 0)

    async def process_evaluator(provider, model_id, name, semaphore):
        """Process all questions for a single evaluator model."""
        print(f"      {name}: starting", flush=True)
        model_start = time.time()
        model_evaluations = {}
        model_timing = []
        total_batches = (len(questions) + 4) // 5

        for i in range(0, len(questions), 5):
            batch_num = i // 5 + 1
            batch = questions[i:i + 5]
            batch_start = time.time()

            async with semaphore:
                results = await asyncio.gather(*[
                    evaluate(provider, model_id, name, q) for q in batch
                ])

            batch_duration = time.time() - batch_start
            completed = min(i + 5, len(questions))

            for _, question, scores, duration in results:
                model_evaluations[question] = scores
                model_timing.append(duration)

            # Progress logging every 5 batches (25 questions) or on slow batches
            if batch_num % 5 == 0 or batch_duration > 60:
                avg_so_far = mean(model_timing) if model_timing else 0
                print(f"      {name}: {completed}/{len(questions)} ({batch_num}/{total_batches} batches, avg {avg_so_far:.1f}s/q)", flush=True)

        avg_time = mean(model_timing) if model_timing else 0
        print(f"    {name}: {len(questions)}/{len(questions)} (avg {avg_time:.2f}s/q) | {format_duration(time.time() - model_start)}", flush=True)

        return name, model_evaluations, model_timing

    # Create semaphores per provider for rate limiting
    providers = set(p for p, _, _ in MODELS)
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p in providers}

    print(f"    Processing all evaluators in parallel...", flush=True)

    # Run all evaluators in parallel
    results = await asyncio.gather(*[
        process_evaluator(p, m, n, semaphores[p]) for p, m, n in MODELS
    ])

    # Merge results
    for name, model_evaluations, model_timing in results:
        evaluations[name] = model_evaluations
        timing[name] = model_timing

    return evaluations, timing


async def phase3_evaluate_answers() -> dict:
    """Phase 3: Run 3 bias test modes and collect comparative data."""
    phase_start = time.time()
    bias_config = get_bias_test_config()
    seed = bias_config["seed"]

    questions = load_json("phase2_answers.json")["questions"]

    print(f"\n{'=' * 60}")
    print("  PHASE 3: Cross-Evaluation (Bias Analysis)")
    print(f"{'-' * 60}")
    print(f"  Revision:    {get_revision()}")
    print(f"  Evaluators:  {len(MODELS)}")
    print(f"  Questions:   {len(questions)}")
    print("  Passes:      3 (shuffle_only, blind_only, shuffle_blind)")
    print("  Web search:  OFF")
    print(f"{'=' * 60}")

    all_evaluations = {}
    all_timing = {}
    mode_durations = {}

    for mode_name, shuffle, blind in BIAS_MODES:
        mode_start = time.time()
        mode_desc = []
        if shuffle:
            mode_desc.append("shuffle")
        if blind:
            mode_desc.append("blind")

        print(f"\n  [{mode_name}] {' + '.join(mode_desc)}")
        print(f"  {'-' * 40}")

        evaluations, timing = await _run_evaluation_pass(questions, shuffle, blind, seed, mode_name)
        all_evaluations[mode_name] = evaluations
        all_timing[mode_name] = calculate_timing_stats(timing)
        mode_durations[mode_name] = round(time.time() - mode_start, 2)

        print(f"  Pass complete: {format_duration(mode_durations[mode_name])}")

        # Incremental save after each mode for crash recovery
        revision = get_revision()
        partial_output = {
            "revision": revision,
            "timestamp": datetime.now().isoformat(),
            "phase": 3,
            "duration_seconds": round(time.time() - phase_start, 2),
            "mode_durations": mode_durations,
            "bias_test_config": {"seed": seed, "modes": [m[0] for m in BIAS_MODES]},
            "timing_stats_by_mode": all_timing,
            "evaluations_by_mode": all_evaluations,
            "complete": mode_name == "shuffle_blind",
        }
        # Add backward-compat fields only when shuffle_blind is done
        if "shuffle_blind" in all_evaluations:
            partial_output["evaluations"] = all_evaluations["shuffle_blind"]
            partial_output["timing_stats"] = all_timing["shuffle_blind"]
        save_json("phase3_rankings.json", partial_output)
        print(f"  Saved checkpoint ({len(all_evaluations)}/3 modes)", flush=True)

    output = partial_output  # Final output is the last checkpoint

    print(f"\n{'=' * 60}")
    print(f"  Phase 3 complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")
    return output
