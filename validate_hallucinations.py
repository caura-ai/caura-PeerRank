"""
Hallucination Validation - Rank models by factual accuracy

Tests models on questions with known facts, measures hallucination rate.
Uses NLI-based verification against evidence documents.

Usage:
    python validate_hallucinations.py              # Run all
    python validate_hallucinations.py --phase 1    # Generate questions
    python validate_hallucinations.py --phase 2    # Get model answers
    python validate_hallucinations.py --phase 3    # Score hallucinations
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from config import MODELS, DATA_DIR, format_duration, PROVIDER_CONCURRENCY
from providers import call_llm, clear_clients

# =============================================================================
# CONFIGURATION
# =============================================================================

HALLUC_DIR = DATA_DIR / "HALLUC"
VALIDATION_REVISION = "HALLUC"

# Questions with verifiable facts - each has topic, question, and evidence passages
FACT_QUESTIONS = [
    {
        "topic": "Eiffel Tower",
        "question": "Tell me about the Eiffel Tower - when was it built, how tall is it, and who designed it?",
        "evidence": [
            "The Eiffel Tower is located in Paris, France.",
            "The Eiffel Tower was completed in 1889 for the World's Fair.",
            "The Eiffel Tower is 330 meters (1,083 ft) tall including antennas.",
            "The tower's base height without antennas is 300 meters (984 ft).",
            "Gustave Eiffel's engineering company designed and built the tower.",
            "Construction took 2 years, 2 months and 5 days.",
            "The tower was the world's tallest man-made structure until 1930.",
        ],
    },
    {
        "topic": "Moon Landing",
        "question": "Describe the first Moon landing - when did it happen, who was involved, and what spacecraft was used?",
        "evidence": [
            "Apollo 11 was the first crewed mission to land on the Moon.",
            "The Moon landing occurred on July 20, 1969.",
            "Neil Armstrong was the first person to walk on the Moon.",
            "Buzz Aldrin was the second person to walk on the Moon.",
            "Michael Collins remained in lunar orbit aboard the Command Module Columbia.",
            "The Lunar Module was named Eagle.",
            "Neil Armstrong's first words on the Moon were 'That's one small step for man, one giant leap for mankind.'",
            "The mission launched from Kennedy Space Center in Florida.",
        ],
    },
    {
        "topic": "Python Programming",
        "question": "What is Python? When was it created and by whom? What are its main features?",
        "evidence": [
            "Python is a high-level, general-purpose programming language.",
            "Python was created by Guido van Rossum.",
            "Python was first released in 1991.",
            "Python emphasizes code readability with significant indentation.",
            "Python is dynamically typed and garbage-collected.",
            "Python supports multiple programming paradigms including procedural, object-oriented, and functional.",
            "The name Python comes from Monty Python's Flying Circus.",
        ],
    },
    {
        "topic": "Albert Einstein",
        "question": "Tell me about Albert Einstein - when and where was he born, what were his major contributions?",
        "evidence": [
            "Albert Einstein was born on March 14, 1879 in Ulm, Germany.",
            "Einstein developed the theory of relativity.",
            "Einstein's mass-energy equivalence formula is E=mc².",
            "Einstein won the Nobel Prize in Physics in 1921.",
            "Einstein won the Nobel Prize for his explanation of the photoelectric effect.",
            "Einstein became a US citizen in 1940.",
            "Einstein died on April 18, 1955 in Princeton, New Jersey.",
            "Einstein worked at the Institute for Advanced Study in Princeton.",
        ],
    },
    {
        "topic": "Amazon River",
        "question": "Describe the Amazon River - how long is it, where is it located, and what makes it significant?",
        "evidence": [
            "The Amazon River is located in South America.",
            "The Amazon is approximately 6,400 kilometers (4,000 miles) long.",
            "The Amazon carries more water than any other river in the world.",
            "The Amazon flows through Brazil, Peru, and Colombia.",
            "The Amazon rainforest surrounds much of the river.",
            "The Amazon basin covers about 7 million square kilometers.",
            "The Amazon has over 1,100 tributaries.",
        ],
    },
    {
        "topic": "Shakespeare",
        "question": "Who was William Shakespeare? When did he live and what were his famous works?",
        "evidence": [
            "William Shakespeare was born in Stratford-upon-Avon, England in 1564.",
            "Shakespeare died on April 23, 1616.",
            "Shakespeare wrote approximately 39 plays.",
            "Shakespeare's famous plays include Hamlet, Macbeth, Romeo and Juliet, and Othello.",
            "Shakespeare wrote 154 sonnets.",
            "Shakespeare's works were performed at the Globe Theatre in London.",
            "Shakespeare is often called England's national poet.",
        ],
    },
    {
        "topic": "Great Wall of China",
        "question": "Tell me about the Great Wall of China - how long is it and when was it built?",
        "evidence": [
            "The Great Wall of China is over 21,000 kilometers (13,000 miles) long including all branches.",
            "The main wall built during the Ming Dynasty is about 8,850 kilometers.",
            "Construction of the wall began in the 7th century BC.",
            "The most well-known sections were built during the Ming Dynasty (1368-1644).",
            "The wall was built to protect against invasions from the north.",
            "The Great Wall is a UNESCO World Heritage Site.",
            "Contrary to popular myth, the wall is not visible from space with the naked eye.",
        ],
    },
    {
        "topic": "Human Heart",
        "question": "How does the human heart work? How many chambers does it have and how often does it beat?",
        "evidence": [
            "The human heart has four chambers: two atria and two ventricles.",
            "The average adult heart beats about 60-100 times per minute at rest.",
            "The heart pumps about 5 liters of blood per minute.",
            "The right side of the heart pumps blood to the lungs.",
            "The left side of the heart pumps blood to the rest of the body.",
            "The heart is about the size of a fist.",
            "The average heart beats about 100,000 times per day.",
        ],
    },
    {
        "topic": "World War II",
        "question": "When did World War II occur and which countries were the main participants?",
        "evidence": [
            "World War II lasted from 1939 to 1945.",
            "The war began when Germany invaded Poland on September 1, 1939.",
            "The Allied Powers included the United States, United Kingdom, Soviet Union, and France.",
            "The Axis Powers included Germany, Italy, and Japan.",
            "The war ended in Europe on May 8, 1945 (V-E Day).",
            "The war ended in the Pacific on September 2, 1945 (V-J Day).",
            "An estimated 70-85 million people died during World War II.",
        ],
    },
    {
        "topic": "Solar System",
        "question": "How many planets are in our solar system? Name them in order from the Sun.",
        "evidence": [
            "There are 8 planets in our solar system.",
            "The planets in order from the Sun are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
            "Pluto was reclassified as a dwarf planet in 2006.",
            "Jupiter is the largest planet in our solar system.",
            "Mercury is the smallest planet in our solar system.",
            "Earth is the third planet from the Sun.",
            "The four inner planets are rocky (terrestrial), the four outer are gas giants.",
        ],
    },
]


def save_halluc_json(filename: str, data: dict):
    HALLUC_DIR.mkdir(parents=True, exist_ok=True)
    base = filename.rsplit(".", 1)[0]
    filepath = HALLUC_DIR / f"{base}_{VALIDATION_REVISION}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath.name}")


def load_halluc_json(filename: str) -> dict:
    base = filename.rsplit(".", 1)[0]
    filepath = HALLUC_DIR / f"{base}_{VALIDATION_REVISION}.json"
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# PHASE 1: Prepare Questions and Evidence
# =============================================================================

def phase1_prepare():
    """Prepare questions and evidence for hallucination testing."""
    print(f"\n{'=' * 60}")
    print("  PHASE 1: Prepare Questions & Evidence")
    print(f"{'=' * 60}")

    save_halluc_json("phase1_questions.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "questions": FACT_QUESTIONS,
    })

    print(f"  Prepared {len(FACT_QUESTIONS)} questions with evidence")
    for q in FACT_QUESTIONS:
        print(f"    - {q['topic']}: {len(q['evidence'])} evidence passages")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 2: Get Model Answers
# =============================================================================

async def phase2_answer():
    """Have all models answer the questions."""
    print(f"\n{'=' * 60}")
    print("  PHASE 2: Get Model Answers")
    print(f"{'=' * 60}")

    phase_start = time.time()
    questions = load_halluc_json("phase1_questions.json")["questions"]
    model_names = [n for _, _, n in MODELS]
    total = len(questions) * len(model_names)
    completed = 0
    lock = asyncio.Lock()

    print(f"  Models: {len(model_names)} | Questions: {len(questions)} | Total: {total}")

    async def answer_one(provider, model_id, model_name, question, semaphore):
        nonlocal completed

        prompt = f"""Answer this question with specific facts. Be precise with dates, numbers, and names.
Keep your answer concise (2-4 sentences).

Question: {question["question"]}"""

        try:
            async with semaphore:
                response, duration, in_tok, out_tok = await call_llm(
                    provider, model_id, prompt,
                    max_tokens=512, timeout=60, temperature=0,
                    use_web_search=False  # Test model knowledge, not search
                )
            result = {
                "text": response.strip(),
                "duration": duration,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        except Exception as e:
            result = {
                "text": f"Error: {e}",
                "duration": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        async with lock:
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"\r  Progress: {completed}/{total} ({100*completed//total}%)", end="", flush=True)

        return model_name, result

    # Process all questions
    output_questions = []
    semaphores = {p: asyncio.Semaphore(PROVIDER_CONCURRENCY.get(p, 5)) for p, _, _ in MODELS}

    for question in questions:
        tasks = [answer_one(p, m, n, question, semaphores[p]) for p, m, n in MODELS]
        results = await asyncio.gather(*tasks)
        output_questions.append({
            "topic": question["topic"],
            "question": question["question"],
            "evidence": question["evidence"],
            "answers": {name: result for name, result in results},
        })

    print()
    save_halluc_json("phase2_answers.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "duration_seconds": round(time.time() - phase_start, 2),
        "questions": output_questions,
    })
    print(f"  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# PHASE 3: Score Hallucinations
# =============================================================================

def phase3_score():
    """Score each model's answers for hallucinations using NLI."""
    print(f"\n{'=' * 60}")
    print("  PHASE 3: Score Hallucinations (NLI)")
    print(f"{'=' * 60}")

    # Import hallucination scoring tools
    try:
        from halucinations import EvidenceIndex, ClaimVerifier, hallucination_score
    except ImportError as e:
        print(f"  Error importing hallucination tools: {e}")
        print("  Make sure sentence-transformers, faiss-cpu, and transformers are installed")
        return

    phase_start = time.time()
    phase2_data = load_halluc_json("phase2_answers.json")
    questions = phase2_data["questions"]
    model_names = [n for _, _, n in MODELS]

    print("  Loading NLI model (this may take a moment)...")
    verifier = ClaimVerifier()

    # Score each model's answers
    model_scores = {n: [] for n in model_names}
    model_details = {n: [] for n in model_names}

    for i, q in enumerate(questions):
        print(f"\r  Processing question {i+1}/{len(questions)}: {q['topic']}", end="", flush=True)

        # Build evidence index for this question
        evidence_index = EvidenceIndex(q["evidence"])

        for model in model_names:
            answer = q["answers"].get(model, {}).get("text", "")
            if answer.startswith("Error:"):
                model_scores[model].append(1.0)  # Treat errors as 100% hallucination
                continue

            result = hallucination_score(answer, evidence_index, verifier)
            model_scores[model].append(result["hallucination_rate"])
            model_details[model].append({
                "topic": q["topic"],
                "answer": answer[:200],
                "hallucination_rate": result["hallucination_rate"],
                "supported_rate": result["supported_rate"],
                "contradicted_rate": result["contradicted_rate"],
                "num_claims": len(result["claims"]),
            })

    print()

    # Calculate summary statistics
    summary = {}
    for model in model_names:
        scores = model_scores[model]
        if scores:
            avg_halluc = sum(scores) / len(scores)
            summary[model] = {
                "avg_hallucination_rate": round(avg_halluc * 100, 1),
                "avg_supported_rate": round((1 - avg_halluc) * 100, 1),
                "num_questions": len(scores),
                "details": model_details[model],
            }

    # Print rankings
    ranked = sorted(summary.items(), key=lambda x: x[1]["avg_hallucination_rate"])

    print(f"\n  {'Rank':<6} {'Model':<25} {'Halluc%':>10} {'Supported%':>12}")
    print(f"  {'-' * 55}")
    for rank, (model, stats) in enumerate(ranked, 1):
        print(f"  {rank:<6} {model:<25} {stats['avg_hallucination_rate']:>9.1f}% {stats['avg_supported_rate']:>11.1f}%")

    # Save results
    save_halluc_json("phase3_scores.json", {
        "revision": VALIDATION_REVISION,
        "timestamp": datetime.now().isoformat(),
        "phase": 3,
        "duration_seconds": round(time.time() - phase_start, 2),
        "summary": summary,
        "rankings": [{"rank": i+1, "model": m, **s} for i, (m, s) in enumerate(ranked)],
    })

    # Generate markdown report
    report = f"""# Hallucination Validation Report

Revision: {VALIDATION_REVISION}
Models: {len(model_names)}
Questions: {len(questions)}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Rankings (Lower Hallucination = Better)

| Rank | Model | Halluc% | Supported% |
|------|-------|---------|------------|
"""
    for rank, (model, stats) in enumerate(ranked, 1):
        report += f"| {rank} | {model} | {stats['avg_hallucination_rate']:.1f}% | {stats['avg_supported_rate']:.1f}% |\n"

    report += f"""
## Methodology

- Models answered {len(questions)} factual questions without web search
- Each answer was split into claims
- Claims were verified against evidence using NLI (Natural Language Inference)
- Hallucination rate = claims not supported by evidence / total claims

## Topics Tested

"""
    for q in questions:
        report += f"- {q['topic']}\n"

    report_file = HALLUC_DIR / f"hallucination_report_{VALIDATION_REVISION}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {report_file.name}")

    print(f"\n  Complete in {format_duration(time.time() - phase_start)}")
    print(f"{'=' * 60}")


# =============================================================================
# MAIN
# =============================================================================

async def run_all():
    """Run all phases."""
    print(f"\n{'#' * 60}")
    print(f"  HALLUCINATION VALIDATION")
    print(f"{'#' * 60}")
    print(f"  Models: {len(MODELS)} | Questions: {len(FACT_QUESTIONS)}")
    print(f"{'#' * 60}\n")

    start = time.time()
    phase1_prepare()
    await phase2_answer()
    phase3_score()

    print(f"\n{'#' * 60}")
    print(f"  COMPLETE in {format_duration(time.time() - start)}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination Validation")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.phase == 1:
        phase1_prepare()
    elif args.phase == 2:
        clear_clients()
        asyncio.run(phase2_answer())
    elif args.phase == 3:
        phase3_score()
    elif args.all:
        clear_clients()
        asyncio.run(run_all())
    else:
        clear_clients()
        asyncio.run(run_all())
