"""
Minimal hallucination test for an LLM (groundedness / faithfulness).

What it measures:
- Splits the model answer into "claims" (sentences).
- Flags a claim as "supported" if it has strong lexical overlap with the context.
- Reports unsupported claim rate.

Limitations (by design): lexical overlap is a heuristic, not true entailment.
Upgrade later by replacing `is_supported_lexical` with an entailment judge model.
"""

from dataclasses import dataclass
import re
from typing import Callable, List, Tuple, Dict

# -----------------------------
# 1) Tiny test set (edit this)
# -----------------------------
@dataclass
class Example:
    name: str
    context: str
    question: str

TESTS: List[Example] = [
    Example(
        name="apollo_11",
        context=(
            "Apollo 11 was the first crewed mission to land on the Moon. "
            "It landed on July 20, 1969. Neil Armstrong and Buzz Aldrin walked on the lunar surface, "
            "while Michael Collins remained in lunar orbit."
        ),
        question="When did Apollo 11 land on the Moon, and who walked on the lunar surface?"
    ),
    Example(
        name="python_release",
        context=(
            "Python is a programming language created by Guido van Rossum. "
            "Python 3.0 was released on December 3, 2008."
        ),
        question="Who created Python, and when was Python 3.0 released?"
    ),
]

# -----------------------------
# 2) Minimal claim extraction
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def extract_claims(answer: str) -> List[str]:
    # Split into sentences and keep non-trivial ones
    parts = [p.strip() for p in _SENT_SPLIT.split(answer.strip()) if p.strip()]
    claims = []
    for p in parts:
        # Skip very short fragments
        if len(p) >= 20:
            claims.append(p)
    return claims if claims else [answer.strip()]  # fallback: treat whole answer as one claim

# -----------------------------
# 3) Minimal support check (lexical overlap)
# -----------------------------
_WORD = re.compile(r"[A-Za-z0-9]+")

def tokenize(s: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(s)]

def is_supported_lexical(claim: str, context: str, *, min_overlap: int = 4) -> Tuple[bool, int]:
    """
    A claim is "supported" if at least `min_overlap` distinct tokens from the claim appear in the context.
    This is a crude but useful baseline for regression testing.

    Tune `min_overlap` based on how verbose your model is.
    """
    c_tokens = set(tokenize(claim))
    ctx_tokens = set(tokenize(context))

    # Remove very common stop-ish tokens (tiny list to stay minimal)
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "was", "is", "are", "were", "be", "by", "it"}
    c_tokens = {t for t in c_tokens if t not in stop and len(t) > 2}

    overlap = len(c_tokens.intersection(ctx_tokens))
    return overlap >= min_overlap, overlap

# -----------------------------
# 4) Runner + report
# -----------------------------
@dataclass
class ClaimResult:
    claim: str
    supported: bool
    overlap: int

@dataclass
class ExampleResult:
    name: str
    answer: str
    claim_results: List[ClaimResult]

def evaluate_one(llm: Callable[[str], str], ex: Example) -> ExampleResult:
    prompt = (
        "Answer the question using ONLY the provided context. "
        "If the context doesn't contain the answer, say \"I don't know.\".\n\n"
        f"Context:\n{ex.context}\n\n"
        f"Question:\n{ex.question}\n"
    )
    answer = llm(prompt)
    claims = extract_claims(answer)

    claim_results: List[ClaimResult] = []
    for claim in claims:
        ok, overlap = is_supported_lexical(claim, ex.context)
        claim_results.append(ClaimResult(claim=claim, supported=ok, overlap=overlap))

    return ExampleResult(name=ex.name, answer=answer, claim_results=claim_results)

def summarize(results: List[ExampleResult]) -> Dict[str, float]:
    total_claims = sum(len(r.claim_results) for r in results)
    unsupported = sum(1 for r in results for c in r.claim_results if not c.supported)

    return {
        "examples": float(len(results)),
        "claims_total": float(total_claims),
        "claims_unsupported": float(unsupported),
        "unsupported_claim_rate": (unsupported / total_claims) if total_claims else 0.0,
    }

def print_report(results: List[ExampleResult]) -> None:
    summary = summarize(results)
    print("\n=== HALLUCINATION (GROUNDEDNESS) REPORT ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    for r in results:
        print(f"\n--- {r.name} ---")
        print("Answer:")
        print(r.answer.strip())
        print("\nClaims:")
        for c in r.claim_results:
            status = "SUPPORTED" if c.supported else "UNSUPPORTED"
            print(f"- [{status}] (overlap={c.overlap}) {c.claim}")

# -----------------------------
# 5) Plug in your model here
# -----------------------------
def dummy_llm(prompt: str) -> str:
    """
    Replace this with your real LLM call.
    This dummy intentionally hallucinates to show what gets caught.
    """
    if "Apollo 11" in prompt:
        return "Apollo 11 landed on July 20, 1969. Neil Armstrong and Buzz Aldrin walked on the Moon. The mission launched from Texas."
    if "Python 3.0" in prompt:
        return "Python was created by Guido van Rossum. Python 3.0 was released on December 3, 2008."
    return "I don't know."

if __name__ == "__main__":
    llm = dummy_llm  # swap in your LLM
    results = [evaluate_one(llm, ex) for ex in TESTS]
    print_report(results)
