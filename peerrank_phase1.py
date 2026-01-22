"""
phase1.py - Question Generation
"""

import asyncio
import time
from datetime import datetime

import config
from config import MODELS, CATEGORIES, MAX_TOKENS_SHORT, MAX_RETRIES, extract_json, save_json, format_duration, get_revision
from providers import call_llm


def get_question_prompt() -> str:
    cat_list = ", ".join([f'"{cat}"' for cat in CATEGORIES])
    return f"""Generate exactly {config.NUM_QUESTIONS} diverse questions for testing AI capabilities.

Use ONLY these exact category values: {cat_list}

Return as JSON object: {{"questions": [{{"category": "factual knowledge", "question": "Your question"}}]}}"""


async def phase1_generate_questions() -> dict:
    """Phase 1: Each LLM generates questions."""
    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Question Generation")
    print(f"{'-' * 60}")
    print(f"  Revision:    {get_revision()}")
    print(f"  Models:      {len(MODELS)}")
    print(f"  Categories:  {len(CATEGORIES)} - {', '.join(c.split()[0] for c in CATEGORIES)}")
    print(f"  Questions:   {config.NUM_QUESTIONS} per model")
    print(f"{'=' * 60}\n")

    phase_start = time.time()
    results, timing = {}, {}
    prompt = get_question_prompt()

    async def generate(provider: str, model_id: str, name: str):
        for attempt in range(MAX_RETRIES + 1):
            try:
                use_json = attempt == 0 and provider in ("openai", "deepseek")
                response, duration, _, _ = await call_llm(provider, model_id, prompt, max_tokens=MAX_TOKENS_SHORT,
                    use_web_search=False, response_format={"type": "json_object"} if use_json else None)
                data = extract_json(response)

                questions = data.get("questions", data) if isinstance(data, dict) else data
                if isinstance(questions, list):
                    normalized = [{"category": q.get("category", "general"), "question": q["question"]}
                                  for q in questions if isinstance(q, dict) and "question" in q]
                    normalized = normalized[:config.NUM_QUESTIONS]
                    if normalized:
                        return (name, normalized, duration, f"{len(normalized)} questions in {duration:.1f}s")
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    return (name, [], 0, f"FAILED ({str(e)[:50]})")
        return (name, [], 0, "FAILED")

    responses = await asyncio.gather(*[generate(p, m, n) for p, m, n in MODELS])

    max_len = max(len(n) for _, _, n in MODELS)
    for name, questions, duration, status in responses:
        print(f"  {name:<{max_len}}  {status}")
        results[name] = questions
        timing[name] = round(duration, 2)

    revision = get_revision()
    output = {
        "revision": revision,
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "duration_seconds": round(time.time() - phase_start, 2),
        "timing_by_model": timing,
        "questions_by_model": results,
    }
    save_json("phase1_questions.json", output)

    total = sum(len(q) for q in results.values() if isinstance(q, list))
    print(f"\nPhase 1 complete: {total} questions in {format_duration(time.time() - phase_start)}")
    return output
