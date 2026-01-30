"""
phase2.py - Answer All Questions (with web search)
"""

import asyncio
import time
from datetime import datetime
from statistics import mean

from peerrank.config import MODELS, MAX_TOKENS_ANSWER, MAX_ANSWER_WORDS, PROVIDER_CONCURRENCY, save_json, load_json, format_duration, get_revision, calculate_timing_stats, get_phase2_web_search, calculate_cost, get_grounding_cost, get_web_grounding_provider
from peerrank.providers import call_llm, web_grounding_search

ANSWER_PROMPT = f"""Answer this question directly and concisely in {MAX_ANSWER_WORDS} words or less. Do not start with "Based on..." or similar preambles.

{{question}}"""

def _empty_costs():
    """Create empty cost tracking dict."""
    return {"total": 0.0, "llm": 0.0, "calls": 0, "in_tok": 0, "out_tok": 0}


def _needs_web_grounding(category: str) -> bool:
    """Check if a category needs web grounding (only current events)."""
    return "current" in category.lower()


async def _fetch_web_grounding(questions: list, revision: str) -> tuple[dict, float]:
    """Fetch web grounding for questions that need it (current events only).

    Uses configured provider (Tavily or SerpAPI).

    Returns:
        tuple: (grounding_dict: {question_idx: grounding_text}, total_cost)
    """
    grounding = {}
    grounding_log = []
    total_cost = 0.0

    provider = get_web_grounding_provider()
    cost_per_search = get_grounding_cost()

    # Count questions needing grounding
    needs_grounding = [q for q in questions if _needs_web_grounding(q["category"])]
    print(f"\n  Fetching web grounding via {provider.upper()} for {len(needs_grounding)}/{len(questions)} questions (current events only)...", flush=True)

    fetched = 0
    for idx, q in enumerate(questions):
        if not _needs_web_grounding(q["category"]):
            # Skip non-current categories
            entry = {
                "index": idx,
                "question": q["text"],
                "category": q["category"],
                "source": q["source"],
                "duration": 0,
                "success": None,  # Not attempted
                "grounding": None,
                "skipped": True
            }
            grounding_log.append(entry)
            grounding[idx] = ""
            continue

        query = q["text"][:400]  # Limit query length
        search_results, duration, success = await web_grounding_search(query)

        entry = {
            "index": idx,
            "question": q["text"],
            "category": q["category"],
            "source": q["source"],
            "duration": round(duration, 2),
            "success": success,
            "grounding": search_results if success else None,
            "skipped": False
        }
        grounding_log.append(entry)

        if success:
            grounding[idx] = search_results
            total_cost += cost_per_search
        else:
            grounding[idx] = ""

        fetched += 1
        if fetched % 10 == 0:
            print(f"    Grounding progress: {fetched}/{len(needs_grounding)}", flush=True)

    # Save all grounding data to single JSON file
    attempted = sum(1 for e in grounding_log if not e.get("skipped"))
    successful = sum(1 for e in grounding_log if e.get("success") is True)
    skipped = sum(1 for e in grounding_log if e.get("skipped"))
    grounding_data = {
        "revision": revision,
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "cost_per_search": cost_per_search,
        "total_questions": len(questions),
        "attempted": attempted,
        "successful": successful,
        "failed": attempted - successful,
        "skipped": skipped,
        "total_cost": round(total_cost, 4),
        "grounding": grounding_log
    }
    save_json("phase2_web_grounding.json", grounding_data)

    print(f"  Web grounding complete ({provider}): {successful}/{attempted} successful, {skipped} skipped, ${total_cost:.4f}", flush=True)
    return grounding, total_cost


async def phase2_answer_questions() -> dict:
    """Phase 2: Each LLM answers ALL questions with standardized web grounding."""
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

    provider = get_web_grounding_provider()
    print(f"  Models:      {len(MODELS)}")
    print(f"  Questions:   {len(questions)}")
    print(f"  API calls:   {len(questions) * len(MODELS)}")
    print(f"  Web grounding: {'ON (' + provider.upper() + ')' if web_search else 'OFF'}")
    print(f"{'=' * 60}")

    # Fetch standardized web grounding for all questions (if enabled)
    grounding = {}
    grounding_cost = 0.0
    if web_search:
        grounding, grounding_cost = await _fetch_web_grounding(questions, get_revision())

    print(f"\n  Processing all models in parallel...", flush=True)

    model_names = [n for _, _, n in MODELS]
    timing = {n: [] for n in model_names}
    costs = {n: _empty_costs() for n in model_names}
    all_answers = {n: {} for n in model_names}
    errors = []

    async def answer(provider, model_id, name, idx, text, grounding_text):
        try:
            prompt = ANSWER_PROMPT.format(question=text)
            # Pass grounding_text for standardized web context (no native web search)
            ans, dur, in_tok, out_tok, _ = await call_llm(
                provider, model_id, prompt, max_tokens=MAX_TOKENS_ANSWER,
                use_web_search=False, grounding_text=grounding_text)
            llm_cost = calculate_cost(model_id, in_tok, out_tok)
            return (idx, ans, dur, in_tok, out_tok, llm_cost, None)
        except Exception as e:
            print(f"    [ERROR] {name} Q#{idx}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return (idx, f"Error: {e}", 0, 0, 0, 0.0, {"model": name, "q": idx, "error": str(e)[:200]})

    async def process_model(provider, model_id, name, semaphore):
        model_start = time.time()
        answers, times, errs = {}, [], []
        c = _empty_costs()

        for i in range(0, len(questions), 5):
            batch = range(i, min(i + 5, len(questions)))
            async with semaphore:
                # Pass grounding text for each question (empty string if no grounding)
                results = await asyncio.gather(*[
                    answer(provider, model_id, name, j, questions[j]["text"], grounding.get(j, ""))
                    for j in batch
                ])

            for idx, ans, dur, in_tok, out_tok, llm_cost, err in results:
                answers[idx] = {"text": ans, "in_tok": in_tok, "out_tok": out_tok,
                                "llm_cost": round(llm_cost, 6), "cost": round(llm_cost, 6)}
                times.append(dur)
                c["total"] += llm_cost
                c["llm"] += llm_cost
                c["calls"] += 1
                c["in_tok"] += in_tok
                c["out_tok"] += out_tok
                if err:
                    errs.append(err)

        avg_t = mean(times) if times else 0
        avg_c = c["total"] / c["calls"] if c["calls"] else 0
        print(f"  {name}: {len(questions)}/{len(questions)} (avg {avg_t:.2f}s/q, ${avg_c:.4f}/q) | {format_duration(time.time() - model_start)} | ${c['llm']:.4f}", flush=True)
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

    # Calculate totals (grounding cost is shared across all models)
    total_llm = sum(c["llm"] for c in costs.values())
    total = total_llm + grounding_cost

    output = {
        "revision": get_revision(),
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "web_search": web_search,
        "web_grounding_provider": provider if web_search else None,
        "duration_seconds": round(time.time() - phase_start, 2),
        "timing_stats": calculate_timing_stats(timing),
        "cost_stats": costs,
        "total_cost": round(total, 4),
        "total_llm_cost": round(total_llm, 4),
        "total_grounding_cost": round(grounding_cost, 4),
        "grounding_file": f"phase2_web_grounding_{get_revision()}.json" if web_search else None,
        "questions": questions,
        "errors": errors,
        "error_count": len(errors),
    }
    save_json("phase2_answers.json", output)

    grounding_str = f" (LLM: ${total_llm:.4f} + Grounding: ${grounding_cost:.4f})" if grounding_cost > 0 else ""
    print(f"\n{'=' * 60}\nPhase 2 complete: {len(questions) * len(MODELS)} answers ({len(errors)} errors) in {format_duration(time.time() - phase_start)}\nTotal Cost: ${total:.4f}{grounding_str}\n{'=' * 60}")
    return output
