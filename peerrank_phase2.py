"""
phase2.py - Answer All Questions (with web search)
"""

import asyncio
import time
from datetime import datetime
from statistics import mean

from peerrank.config import MODELS, MAX_TOKENS_ANSWER, MAX_ANSWER_WORDS, PROVIDER_CONCURRENCY, save_json, load_json, format_duration, get_revision, calculate_timing_stats, get_phase2_web_search, calculate_cost
from peerrank.providers import call_llm

ANSWER_PROMPT = f"""Answer this question directly and concisely in {MAX_ANSWER_WORDS} words or less. Do not start with "Based on..." or similar preambles.

{{question}}"""


async def phase2_answer_questions() -> dict:
    """Phase 2: Each LLM answers ALL questions with web search enabled."""
    web_search_enabled = get_phase2_web_search()

    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Answer Questions")
    print(f"{'-' * 60}")
    print(f"  Revision:    {get_revision()}")

    phase_start = time.time()
    phase1_data = load_json("phase1_questions.json")

    # Flatten questions
    questions = []
    for model_name, model_qs in phase1_data["questions_by_model"].items():
        for q in model_qs:
            if isinstance(q, dict) and "question" in q:
                questions.append({"source": model_name, "category": q.get("category", "general"),
                                  "text": q["question"], "answers": {}})
            elif isinstance(q, str) and not q.startswith("Error"):
                questions.append({"source": model_name, "category": "general", "text": q, "answers": {}})

    print(f"  Models:      {len(MODELS)}")
    print(f"  Questions:   {len(questions)}")
    print(f"  API calls:   {len(questions) * len(MODELS)}")
    print(f"  Web search:  {'ON' if web_search_enabled else 'OFF'}")
    print(f"{'=' * 60}")

    timing = {n: [] for _, _, n in MODELS}
    costs = {n: {"total": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0} for _, _, n in MODELS}
    errors = []
    all_answers = {n: {} for _, _, n in MODELS}  # model -> {idx: answer_data}

    async def answer(provider, model_id, name, idx, text):
        try:
            prompt = ANSWER_PROMPT.format(question=text)
            ans, duration, input_tokens, output_tokens = await call_llm(provider, model_id, prompt, max_tokens=MAX_TOKENS_ANSWER, use_web_search=web_search_enabled)
            cost = calculate_cost(model_id, input_tokens, output_tokens)
            return (name, idx, ans, duration, input_tokens, output_tokens, cost, None)
        except Exception as e:
            print(f"    [ERROR] {name} Q#{idx}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return (name, idx, f"Error: {e}", 0, 0, 0, 0.0, {"model": name, "q": idx, "error": str(e)[:200]})

    async def process_model(provider, model_id, name, semaphore):
        """Process all questions for a single model."""
        model_start = time.time()
        model_timing = []
        model_costs = {"total": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
        model_answers = {}
        model_errors = []

        for i in range(0, len(questions), 5):
            batch = range(i, min(i + 5, len(questions)))
            async with semaphore:
                results = await asyncio.gather(*[answer(provider, model_id, name, j, questions[j]["text"]) for j in batch])

            for _, idx, ans, duration, input_tokens, output_tokens, cost, err in results:
                model_answers[idx] = {
                    "text": ans,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": round(cost, 6)
                }
                model_timing.append(duration)
                model_costs["total"] += cost
                model_costs["calls"] += 1
                model_costs["input_tokens"] += input_tokens
                model_costs["output_tokens"] += output_tokens
                if err:
                    model_errors.append(err)

        avg_time = mean(model_timing) if model_timing else 0
        avg_cost = model_costs["total"] / model_costs["calls"] if model_costs["calls"] > 0 else 0
        print(f"  {name}: {len(questions)}/{len(questions)} (avg {avg_time:.2f}s/q, ${avg_cost:.4f}/q) | {format_duration(time.time() - model_start)} | ${model_costs['total']:.4f}", flush=True)

        return name, model_answers, model_timing, model_costs, model_errors

    # Create semaphores per provider for rate limiting
    providers = set(p for p, _, _ in MODELS)
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p in providers}

    print(f"\n  Processing all models in parallel...", flush=True)

    # Run all models in parallel
    results = await asyncio.gather(*[
        process_model(p, m, n, semaphores[p]) for p, m, n in MODELS
    ])

    # Merge results
    for name, model_answers, model_timing, model_costs, model_errors in results:
        all_answers[name] = model_answers
        timing[name] = model_timing
        costs[name] = model_costs
        errors.extend(model_errors)

    # Update questions with answers
    for name, answers_dict in all_answers.items():
        for idx, answer_data in answers_dict.items():
            questions[idx]["answers"][name] = answer_data

    # Calculate total cost
    total_cost = sum(c["total"] for c in costs.values())

    revision = get_revision()
    output = {
        "revision": revision,
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "web_search": web_search_enabled,
        "duration_seconds": round(time.time() - phase_start, 2),
        "timing_stats": calculate_timing_stats(timing),
        "cost_stats": costs,
        "total_cost": round(total_cost, 4),
        "questions": questions,
        "errors": errors,
        "error_count": len(errors),
    }
    save_json("phase2_answers.json", output)

    print(f"\n{'=' * 60}\nPhase 2 complete: {len(questions) * len(MODELS)} answers ({len(errors)} errors) in {format_duration(time.time() - phase_start)}\nTotal Cost: ${total_cost:.4f}\n{'=' * 60}")
    return output
