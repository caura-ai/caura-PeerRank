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


def _empty_costs():
    """Create empty cost tracking dict."""
    return {"total": 0.0, "llm": 0.0, "tavily": 0.0, "calls": 0, "in_tok": 0, "out_tok": 0}


async def phase2_answer_questions() -> dict:
    """Phase 2: Each LLM answers ALL questions with web search enabled."""
    web_search = get_phase2_web_search()

    print(f"\n{'=' * 60}\n  PHASE 2: Answer Questions\n{'-' * 60}")
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
    print(f"  Web search:  {'ON' if web_search else 'OFF'}")
    print(f"{'=' * 60}\n  Processing all models in parallel...", flush=True)

    model_names = [n for _, _, n in MODELS]
    timing = {n: [] for n in model_names}
    costs = {n: _empty_costs() for n in model_names}
    all_answers = {n: {} for n in model_names}
    errors = []

    async def answer(provider, model_id, name, idx, text):
        try:
            prompt = ANSWER_PROMPT.format(question=text)
            ans, dur, in_tok, out_tok, tav_cost = await call_llm(
                provider, model_id, prompt, max_tokens=MAX_TOKENS_ANSWER, use_web_search=web_search)
            llm_cost = calculate_cost(model_id, in_tok, out_tok)
            return (idx, ans, dur, in_tok, out_tok, llm_cost, tav_cost, None)
        except Exception as e:
            print(f"    [ERROR] {name} Q#{idx}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return (idx, f"Error: {e}", 0, 0, 0, 0.0, 0.0, {"model": name, "q": idx, "error": str(e)[:200]})

    async def process_model(provider, model_id, name, semaphore):
        model_start = time.time()
        answers, times, errs = {}, [], []
        c = _empty_costs()

        for i in range(0, len(questions), 5):
            batch = range(i, min(i + 5, len(questions)))
            async with semaphore:
                results = await asyncio.gather(*[answer(provider, model_id, name, j, questions[j]["text"]) for j in batch])

            for idx, ans, dur, in_tok, out_tok, llm_cost, tav_cost, err in results:
                total = llm_cost + tav_cost
                answers[idx] = {"text": ans, "in_tok": in_tok, "out_tok": out_tok,
                                "llm_cost": round(llm_cost, 6), "tavily_cost": round(tav_cost, 6), "cost": round(total, 6)}
                times.append(dur)
                c["total"] += total
                c["llm"] += llm_cost
                c["tavily"] += tav_cost
                c["calls"] += 1
                c["in_tok"] += in_tok
                c["out_tok"] += out_tok
                if err:
                    errs.append(err)

        avg_t = mean(times) if times else 0
        avg_c = c["total"] / c["calls"] if c["calls"] else 0
        tav_str = f" +${c['tavily']:.4f} Tavily" if c['tavily'] > 0 else ""
        print(f"  {name}: {len(questions)}/{len(questions)} (avg {avg_t:.2f}s/q, ${avg_c:.4f}/q) | {format_duration(time.time() - model_start)} | ${c['llm']:.4f}{tav_str}", flush=True)
        return name, answers, times, c, errs

    # Create semaphores per provider for rate limiting
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p in set(p for p, _, _ in MODELS)}

    results = await asyncio.gather(*[process_model(p, m, n, semaphores[p]) for p, m, n in MODELS])

    # Merge results
    for name, answers, times, c, errs in results:
        all_answers[name] = answers
        timing[name] = times
        costs[name] = c
        errors.extend(errs)

    # Update questions with answers
    for name, ans_dict in all_answers.items():
        for idx, data in ans_dict.items():
            questions[idx]["answers"][name] = data

    # Calculate totals
    total = sum(c["total"] for c in costs.values())
    total_llm = sum(c["llm"] for c in costs.values())
    total_tav = sum(c["tavily"] for c in costs.values())

    output = {
        "revision": get_revision(),
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "web_search": web_search,
        "duration_seconds": round(time.time() - phase_start, 2),
        "timing_stats": calculate_timing_stats(timing),
        "cost_stats": costs,
        "total_cost": round(total, 4),
        "total_llm_cost": round(total_llm, 4),
        "total_tavily_cost": round(total_tav, 4),
        "questions": questions,
        "errors": errors,
        "error_count": len(errors),
    }
    save_json("phase2_answers.json", output)

    tav_str = f" (LLM: ${total_llm:.4f} + Tavily: ${total_tav:.4f})" if total_tav > 0 else ""
    print(f"\n{'=' * 60}\nPhase 2 complete: {len(questions) * len(MODELS)} answers ({len(errors)} errors) in {format_duration(time.time() - phase_start)}\nTotal Cost: ${total:.4f}{tav_str}\n{'=' * 60}")
    return output
