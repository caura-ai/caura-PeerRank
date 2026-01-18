"""
config.py - Configuration constants and utilities for PeerRank.ai
"""

import json
import os
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

load_dotenv()

# Paths and constants
DATA_DIR = Path(__file__).parent / "data"


REVISION = "v1"


def get_revision() -> str:
    """Get current revision string."""
    return REVISION


def set_revision(rev: str):
    """Set revision string."""
    global REVISION
    REVISION = rev.strip() or "v1"


def set_bias_test_config(seed: int | None = None):
    """Configure seed for Phase 3 (runs all 3 bias modes automatically)."""
    global BIAS_TEST_SEED
    BIAS_TEST_SEED = seed


def get_bias_test_config() -> dict:
    """Get current bias testing configuration."""
    return {"seed": BIAS_TEST_SEED}

# Token limits
MAX_TOKENS_SHORT = 2048
MAX_TOKENS_ANSWER = 8192
MAX_TOKENS_EVAL = 12000
MAX_TOKENS_DEEPSEEK = 8192
MAX_TOKENS_GOOGLE = 16000
MAX_ANSWER_WORDS = 200
DEFAULT_TIMEOUT = 180
MAX_RETRIES = 4
RETRY_DELAY = 3

# Temperature settings
TEMPERATURE_DEFAULT = 0.7
TEMPERATURE_EVAL = 0

# Model-specific temperature overrides (for models that don't support certain values)
MODEL_TEMPERATURE_OVERRIDES = {
    "gpt-5-mini": 1.0,  # GPT-5-mini doesn't support 0.7
}

# Efficiency calculation exponent - rewards higher peer scores
# 1.0 = linear, 1.5 = moderate score bonus, 2.0 = strong score bonus
EFFICIENCY_QUALITY_EXPONENT = 2

# Token costs per million tokens (input, output) - Updated Jan 2026
# Format: model_id -> (input_cost_per_1M, output_cost_per_1M)
TOKEN_COSTS = {
    # OpenAI
    "gpt-5.2": (1.75, 14.00),
    "gpt-5-mini": (0.25, 2.00),

    # Anthropic (with prompt caching, cache reads are 90% off: $0.50/M)
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-5": (3.00, 15.00),

    # Google Gemini
    "gemini-3-pro-preview": (2.00, 12.00),  # Base price, up to $4/$18 for long context
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3-flash-thinking": (0.50, 3.00),  # Flash with thinking=high (same pricing, more tokens)
    "gemini-2.5-pro": (1.25, 10.00),   # Smart "Thinking" model (Input $1.25 / Output $10.00)
    "gemini-2.5-flash": (0.15, 0.60),  # Fast "Workhorse" model (Input $0.15 / Output $0.60)

    # xAI
    "grok-4-1-fast": (0.60, 3.00),

    # DeepSeek (with auto caching: cache hit $0.028/M, cache miss $0.28/M)
    "deepseek-chat": (0.28, 0.42),

    # Together AI
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.27),

    # Perplexity (includes native web search)
    "sonar-pro": (3.00, 15.00),

    # Moonshot AI (Kimi) - with auto caching: cache hit $0.15/M
    "kimi-k2-0905-preview": (0.60, 2.50),  # K2 model (256K context)
    "kimi-k2-0711-preview": (0.60, 2.50),  # K2 model (128K context)
    "kimi-k2-turbo-preview": (1.15, 8.00), # Turbo variant
    "kimi-k2-thinking": (0.60, 2.50),      # Thinking model

    # Mistral AI
    "mistral-large-latest": (2.00, 6.00),  # Mistral Large (128K context)
}

# Evaluation settings
NUM_QUESTIONS = 2

# Phase 2 web search toggle
PHASE2_WEB_SEARCH = True  # Default: enabled (current behavior)


def get_phase2_web_search() -> bool:
    """Get Phase 2 web search setting."""
    return PHASE2_WEB_SEARCH


def set_phase2_web_search(enabled: bool):
    """Set Phase 2 web search setting."""
    global PHASE2_WEB_SEARCH
    PHASE2_WEB_SEARCH = enabled


# Phase 3 runs 3 bias modes automatically (shuffle_only, blind_only, shuffle_blind)
BIAS_TEST_SEED = None  # Random seed for reproducible shuffling (None = random)

# Phase 5 judge model (provider, model_id, display_name)
PHASE5_JUDGE = ("openai", "gpt-5.2", "gpt-5.2")


def get_phase5_judge() -> tuple:
    """Get Phase 5 judge model tuple."""
    return PHASE5_JUDGE


def set_phase5_judge(provider: str, model_id: str, display_name: str):
    """Set Phase 5 judge model."""
    global PHASE5_JUDGE
    PHASE5_JUDGE = (provider, model_id, display_name)


ALL_CATEGORIES = [
    "current events (needs recent info)",
    "factual knowledge",
    "reasoning/logic",
    "creative/open-ended",
    "practical how-to",
]
CATEGORIES = ALL_CATEGORIES.copy()

# Models: (provider, model_id, display_name)
ALL_MODELS = [
    ("openai", "gpt-5.2", "gpt-5.2"),
    ("openai", "gpt-5-mini", "gpt-5-mini"),
    ("anthropic", "claude-opus-4-5", "claude-opus-4-5"),
    ("anthropic", "claude-sonnet-4-5", "claude-sonnet-4-5"),
    ("google", "gemini-3-pro-preview", "gemini-3-pro-preview"),
    #("google", "gemini-3-flash-preview", "gemini-3-flash-preview"),
    ("google", "gemini-3-flash-thinking", "gemini-3-flash-thinking"),  # Flash with thinking=high
    #("google", "gemini-2.5-pro", "gemini-2.5-pro"),
    #("google", "gemini-2.5-flash", "gemini-2.5-flash"),
    ("grok", "grok-4-1-fast", "grok-4-1-fast"),
    ("deepseek", "deepseek-chat", "deepseek-chat"),
    ("together", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "llama-4-maverick"),
    ("perplexity", "sonar-pro", "sonar-pro"),
    ("kimi", "kimi-k2-0905-preview", "kimi-k2-0905"),
    ("mistral", "mistral-large-latest", "mistral-large"),
]
MODELS = ALL_MODELS.copy()

# Google service account config
GOOGLE_SERVICE_ACCOUNT_FILE = Path(__file__).parent / "alpine-theory-469016-c8-2a7f2f635a03.json"
GOOGLE_PROJECT_ID = "alpine-theory-469016-c8"
GOOGLE_LOCATION = "global"


# Provider concurrency limits (max concurrent requests per provider)
# Used for parallel model processing in Phase 2 and Phase 3
PROVIDER_CONCURRENCY = {
    "openai": 8,
    "anthropic": 8,
    "google": 8,
    "grok": 8,
    "deepseek": 8,
    "together": 8,
    "perplexity": 8,
    "kimi": 8,
    "mistral": 8,
}


def set_active_models(include: list[str] | None = None, exclude: list[str] | None = None):
    """Filter which models participate in the run."""
    filtered = ALL_MODELS.copy()
    if include:
        include_lower = [m.lower() for m in include]
        filtered = [m for m in filtered if m[2].lower() in include_lower]
    if exclude:
        exclude_lower = [m.lower() for m in exclude]
        filtered = [m for m in filtered if m[2].lower() not in exclude_lower]
    MODELS.clear()
    MODELS.extend(filtered)
    return MODELS


def list_available_models() -> list[str]:
    return [m[2] for m in ALL_MODELS]


def set_active_categories(include: list[str] | None = None, exclude: list[str] | None = None):
    """Filter which question categories are used."""
    filtered = ALL_CATEGORIES.copy()
    if include:
        include_lower = [c.lower() for c in include]
        filtered = [c for c in filtered if any(i in c.lower() for i in include_lower)]
    if exclude:
        exclude_lower = [c.lower() for c in exclude]
        filtered = [c for c in filtered if not any(e in c.lower() for e in exclude_lower)]
    CATEGORIES.clear()
    CATEGORIES.extend(filtered)
    return CATEGORIES


def list_available_categories() -> list[str]:
    return ALL_CATEGORIES.copy()


def get_api_key(provider: str) -> str:
    key_map = {
        "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY", "grok": "XAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY", "together": "TOGETHER_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY", "kimi": "KIMI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }
    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.getenv(env_var)
    # Fallback for grok: try GROK_API_KEY if XAI_API_KEY not set
    if not key and provider == "grok":
        key = os.getenv("GROK_API_KEY")
    if not key and provider == "google" and GOOGLE_SERVICE_ACCOUNT_FILE.exists():
        return ""
    if not key:
        raise ValueError(f"{env_var} not set")
    return key


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.1f}s"


def extract_json(text: str) -> dict | list | None:
    """Extract JSON from text that may contain markdown or extra content."""
    if not text:
        return None
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # From markdown code blocks
    if "```" in text:
        for part in text.split("```"):
            clean = part.strip().removeprefix("json").removeprefix("JSON").strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                continue

    # Find JSON by brackets
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start, end = text.find(start_char), text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def save_json(filename: str, data: dict):
    """Save JSON to data directory with revision tag in filename."""
    DATA_DIR.mkdir(exist_ok=True)
    base, ext = filename.rsplit(".", 1)
    filepath = DATA_DIR / f"{base}_{REVISION}.{ext}"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {filepath}")


def load_json(filename: str) -> dict:
    """Load JSON from data directory with current revision tag."""
    base, ext = filename.rsplit(".", 1)
    filepath = DATA_DIR / f"{base}_{REVISION}.{ext}"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath.name} not found")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_last_completed_phase() -> int:
    for phase, base in [(3, "phase3_rankings"), (2, "phase2_answers"), (1, "phase1_questions")]:
        if (DATA_DIR / f"{base}_{REVISION}.json").exists():
            return phase
    return 0


def calculate_timing_stats(timing: dict) -> dict:
    result = {}
    for name, t in timing.items():
        if t:
            successes = [x for x in t if x > 0]
            result[name] = {
                "total": round(sum(t), 2),
                "avg": round(mean(successes), 2) if successes else 0,
                "count": len(t),  # Total attempts
                "successes": len(successes),  # Successful calls only
            }
    return result


def format_table(headers: list[str], rows: list[list[str]], alignments: list[str] | None = None) -> str:
    """Format a markdown table with proper column alignment."""
    alignments = alignments or ['l'] * len(headers)
    widths = [max(len(h), max((len(str(row[i])) for row in rows), default=0)) for i, h in enumerate(headers)]

    def align(text, width, a):
        return text.ljust(width) if a == 'l' else text.rjust(width) if a == 'r' else text.center(width)

    sep = '|'.join(':' + '-' * w + ':' if a == 'c' else '-' * (w + 1) + ':' if a == 'r' else '-' * (w + 2)
                   for w, a in zip(widths, alignments))

    lines = ['| ' + ' | '.join(align(h, widths[i], alignments[i]) for i, h in enumerate(headers)) + ' |',
             '|' + sep + '|']
    lines.extend('| ' + ' | '.join(align(str(c), widths[i], alignments[i]) for i, c in enumerate(row)) + ' |' for row in rows)
    return '\n'.join(lines)


def match_model_name(name: str) -> str | None:
    """Match a possibly shortened model name to the display name in MODELS."""
    display_names = [n for _, _, n in MODELS]
    name = name.strip().strip('[]')

    if name in display_names:
        return name
    name_lower = name.lower()
    for full_name in display_names:
        if name_lower in full_name.lower() or full_name.lower().startswith(name_lower):
            return full_name
    return None


# Bias test modes for Phase 3 evaluation (tuples for backend)
BIAS_MODES = [
    ("shuffle_only", True, False),   # Randomize order, show real names
    ("blind_only", False, True),     # Fixed order, hide names
    ("shuffle_blind", True, True),   # Both protections (baseline)
]

# Bias test configs with UI metadata (dicts for frontend)
BIAS_CONFIGS = [
    {"name": "shuffle_only", "shuffle": True, "blind": False, "icon": "🔀", "desc": "Order random, names visible"},
    {"name": "blind_only", "shuffle": False, "blind": True, "icon": "🙈", "desc": "Order fixed, names hidden"},
    {"name": "shuffle_blind", "shuffle": True, "blind": True, "icon": "🎭", "desc": "Both protections (baseline)"},
]

# Display configs for UI (subset of modes, reordered for presentation)
UI_DISPLAY_MODES = [
    {"name": "shuffle_blind", "display_name": "Peer Score", "icon": "🏆", "desc": "Final ranking (shuffle + blind)"},
    {"name": "shuffle_only", "display_name": "Shuffle (names visible)", "icon": "🔀", "desc": "Order random, model names shown"},
]


def calculate_scores_from_evaluations(evaluations: dict, model_names: list[str] = None) -> dict:
    """
    Calculate peer scores, self scores, and raw scores from evaluation data.

    Args:
        evaluations: Dict of {evaluator: {question: {model: {score, reason}}}} (phase3 format)
                    or {evaluator: {evaluator, scores: {model: {score, reason}}}} (UI format)
        model_names: List of model display names (defaults to MODELS)

    Returns:
        Dict with keys: peer_scores, self_scores, raw_scores, judge_scores
        Each is a dict of {model_name: list of scores or single score}
    """
    if model_names is None:
        model_names = [n for _, _, n in MODELS]

    peer_scores = {n: [] for n in model_names}
    self_scores = {n: [] for n in model_names}
    raw_scores = {n: [] for n in model_names}
    judge_given = {n: [] for n in model_names}

    for evaluator_name, eval_data in evaluations.items():
        matched_evaluator = match_model_name(evaluator_name)

        # Handle both formats: phase3 nested dict or UI flat dict
        if isinstance(eval_data, dict) and "scores" in eval_data:
            # UI format: {evaluator, scores: {...}, ...}
            scores_dict = eval_data.get("scores", {})
            for model_name, score_data in scores_dict.items():
                if isinstance(score_data, dict) and "score" in score_data:
                    _record_score(score_data["score"], model_name, matched_evaluator,
                                 peer_scores, self_scores, raw_scores, judge_given)
        else:
            # Phase3 format: {question: {model: {score, reason}}}
            for question_scores in eval_data.values():
                if isinstance(question_scores, dict):
                    for model_name, score_data in question_scores.items():
                        if isinstance(score_data, dict) and "score" in score_data:
                            _record_score(score_data["score"], model_name, matched_evaluator,
                                         peer_scores, self_scores, raw_scores, judge_given)

    return {
        "peer_scores": peer_scores,
        "self_scores": self_scores,
        "raw_scores": raw_scores,
        "judge_given": judge_given,
    }


def _record_score(score: float, model_name: str, evaluator: str,
                  peer_scores: dict, self_scores: dict, raw_scores: dict, judge_given: dict):
    """Helper to record a score in the appropriate buckets."""
    # Convert score to float in case it's a string or int
    try:
        score = float(score)
    except (ValueError, TypeError):
        return  # Skip invalid scores

    matched = match_model_name(model_name)
    if matched:
        raw_scores[matched].append(score)
        if matched == evaluator:
            self_scores[matched].append(score)
        else:
            peer_scores[matched].append(score)
    # Only count scores given to OTHER models for judge generosity (exclude self-ratings)
    if evaluator and evaluator in judge_given and matched != evaluator:
        judge_given[evaluator].append(score)


def _pearson_correlation(x: list, y: list) -> float:
    """Calculate Pearson correlation coefficient between two lists."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    from math import sqrt
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def calculate_judge_agreement(evaluations: dict) -> dict:
    """
    Calculate pairwise correlation between judges based on scores given.

    Args:
        evaluations: Dict of {evaluator: {question: {model: {score, reason}}}}

    Returns:
        Dict with 'matrix' (judge -> judge -> correlation), 'pairs' list, and 'judges' list.
    """
    judges = list(evaluations.keys())

    # Build score vectors for each judge: {judge: {(question, model): score}}
    judge_scores = {j: {} for j in judges}
    for judge, questions in evaluations.items():
        for question, model_scores in questions.items():
            for model, score_data in model_scores.items():
                score = score_data.get("score") if isinstance(score_data, dict) else score_data
                if isinstance(score, (int, float)):
                    judge_scores[judge][(question, model)] = score

    # Calculate pairwise correlations
    matrix = {j1: {j2: 0.0 for j2 in judges} for j1 in judges}
    pairs = []

    for i, j1 in enumerate(judges):
        for j2 in judges[i:]:
            # Find common (question, model) pairs
            common_keys = set(judge_scores[j1].keys()) & set(judge_scores[j2].keys())
            if len(common_keys) < 3:
                corr = 0.0
            else:
                scores1 = [judge_scores[j1][k] for k in common_keys]
                scores2 = [judge_scores[j2][k] for k in common_keys]
                corr = _pearson_correlation(scores1, scores2)

            matrix[j1][j2] = corr
            matrix[j2][j1] = corr
            if j1 != j2:
                pairs.append((j1, j2, corr, len(common_keys)))

    # Sort pairs by correlation (highest agreement first)
    pairs.sort(key=lambda x: -x[2])

    return {"matrix": matrix, "pairs": pairs, "judges": judges}


def calculate_question_stats(evaluations: dict, questions: list = None) -> dict:
    """
    Calculate statistics for each question based on evaluation scores.

    Args:
        evaluations: Dict of {evaluator: {question: {model: {score, reason}}}}
        questions: Optional list of question dicts from phase1 (for metadata)

    Returns:
        Dict with:
        - 'questions': {q_id: {avg, std, min, max, count, scores, best_model, worst_model}}
        - 'hardest': Top 5 lowest avg score (hard questions)
        - 'easiest': Top 5 highest avg score (easy questions)
        - 'controversial': Top 5 highest std (most disagreement)
        - 'consensus': Top 5 lowest std (most agreement)
    """
    from statistics import stdev

    # Build question metadata lookup (use full question text as key to match evaluations)
    q_meta = {}
    if questions:
        for q in questions:
            q_text = q.get("question", "")
            q_meta[q_text] = {
                "text": q_text,
                "category": q.get("category", ""),
                "source": q.get("source_model", ""),
            }

    # Collect all scores per question
    question_scores = {}  # {question_id: [(score, model, evaluator), ...]}

    for evaluator, q_evals in evaluations.items():
        for question_id, model_scores in q_evals.items():
            if question_id not in question_scores:
                question_scores[question_id] = []

            for model, score_data in model_scores.items():
                score = score_data.get("score") if isinstance(score_data, dict) else score_data
                if isinstance(score, (int, float)):
                    question_scores[question_id].append((score, model, evaluator))

    # Calculate stats per question
    question_stats = {}
    for q_id, scores_list in question_scores.items():
        if not scores_list:
            continue

        scores = [s[0] for s in scores_list]
        avg = sum(scores) / len(scores)
        std = stdev(scores) if len(scores) > 1 else 0.0
        min_score = min(scores)
        max_score = max(scores)

        # Find best/worst performing models on this question
        model_avgs = {}
        for score, model, _ in scores_list:
            if model not in model_avgs:
                model_avgs[model] = []
            model_avgs[model].append(score)
        model_avgs = {m: sum(s)/len(s) for m, s in model_avgs.items()}

        best_model = max(model_avgs, key=model_avgs.get) if model_avgs else None
        worst_model = min(model_avgs, key=model_avgs.get) if model_avgs else None

        # Get metadata if available
        meta = q_meta.get(q_id, {})

        question_stats[q_id] = {
            "avg": avg,
            "std": std,
            "min": min_score,
            "max": max_score,
            "count": len(scores),
            "best_model": best_model,
            "best_score": model_avgs.get(best_model, 0),
            "worst_model": worst_model,
            "worst_score": model_avgs.get(worst_model, 0),
            "text": meta.get("text", q_id),
            "category": meta.get("category", ""),
            "source": meta.get("source", ""),
        }

    # Sort for leaderboards
    sorted_by_avg = sorted(question_stats.items(), key=lambda x: x[1]["avg"])
    sorted_by_std = sorted(question_stats.items(), key=lambda x: x[1]["std"], reverse=True)

    return {
        "questions": question_stats,
        "hardest": sorted_by_avg[:5],  # Lowest avg = hardest
        "easiest": sorted_by_avg[-5:][::-1],  # Highest avg = easiest
        "controversial": sorted_by_std[:5],  # Highest std = most disagreement
        "consensus": sorted_by_std[-5:][::-1],  # Lowest std = most agreement
    }


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate API cost for a given model and token usage.

    Args:
        model_id: Model identifier (e.g., "gpt-5.2", "claude-opus-4-5")
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated

    Returns:
        Total cost in USD, or 0.0 if model pricing not found
    """
    if model_id not in TOKEN_COSTS:
        return 0.0

    input_cost_per_m, output_cost_per_m = TOKEN_COSTS[model_id]

    # Convert per-million pricing to actual cost
    input_cost = (input_tokens / 1_000_000) * input_cost_per_m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_m

    return input_cost + output_cost
