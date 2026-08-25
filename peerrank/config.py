"""
config.py - Configuration constants and utilities for PeerRank.ai
"""

import json
import os
import re
from pathlib import Path
from statistics import mean, stdev

from dotenv import load_dotenv

from .models import ALL_MODELS

# Derived from ALL_MODELS
TOKEN_COSTS = {m["model_id"]: m["cost"] for m in ALL_MODELS}

load_dotenv(override=True)

# Paths and constants
DATA_DIR = Path(__file__).parent.parent / "data"


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
MAX_TOKENS_SHORT = 4096  # Phase 1 question generation (increased for verbose models)
MAX_TOKENS_ANSWER = 16384
MAX_TOKENS_EVAL = 32000
MAX_TOKENS_DEEPSEEK = 32000     # deepseek-v4-flash is a CoT model: a 12-response eval spends ~14.7k tokens on reasoning ALONE, so the old 8192 cap hit finish_reason=length with empty content. Verified 2026-08-25: at 32000 it finishes (stop) and emits full JSON. (Was 8192, suited to non-reasoning deepseek-chat.)
MAX_ANSWER_WORDS = 200
MAX_RETRIES = 5
RETRY_DELAY =  4
DEFAULT_TIMEOUT = 180
# Model-specific timeout overrides (seconds)
MODEL_TIMEOUT_OVERRIDES = {
}

# Temperature settings
TEMPERATURE_DEFAULT = 0.5
TEMPERATURE_EVAL = 0

# Model-specific temperature overrides (for models that don't support certain values)
MODEL_TEMPERATURE_OVERRIDES = {
    "gpt-5-mini": 1.0,  # GPT-5-mini doesn't support 0.7
    "kimi-k3": 1.0,  # Kimi only allows temperature=1
    "gpt-5.6": 1.0,  # GPT-5.6 only allows temperature=1
    "gpt-5.6-terra": 1.0,  # GPT-5.6 family only allows temperature=1
    "gpt-5.6-luna": 1.0,  # GPT-5.6 family only allows temperature=1
    "claude-opus-5": 1.0,  # Opus 5 requires temperature=1 (thinking on by default)
    "claude-fable-5": 1.0,  # Fable 5 requires temperature=1 (extended thinking)
    "claude-sonnet-5": 1.0,  # Sonnet 5 requires temperature=1 (adaptive thinking)
}

# Efficiency calculation exponent - rewards higher peer scores
# 1.0 = linear, 1.5 = moderate score bonus, 2.0 = strong score bonus
EFFICIENCY_QUALITY_EXPONENT = 2

# Web grounding provider selection
WEB_GROUNDING_PROVIDER = "tavily"  # "tavily" or "serpapi"

# Web grounding costs per search
TAVILY_COST_PER_SEARCH = 0.008    # Tavily basic search = 1 credit = $0.008
SERPAPI_COST_PER_SEARCH = 0.01   # SerpAPI ~$0.01 per search (varies by plan)

def get_grounding_cost() -> float:
    """Get cost per search for current grounding provider."""
    if WEB_GROUNDING_PROVIDER == "serpapi":
        return SERPAPI_COST_PER_SEARCH
    return TAVILY_COST_PER_SEARCH

def get_web_grounding_provider() -> str:
    """Get current web grounding provider."""
    return WEB_GROUNDING_PROVIDER

def set_web_grounding_provider(provider: str):
    """Set web grounding provider ('tavily' or 'serpapi')."""
    global WEB_GROUNDING_PROVIDER
    if provider.lower() not in ("tavily", "serpapi"):
        raise ValueError(f"Invalid provider: {provider}. Must be 'tavily' or 'serpapi'")
    WEB_GROUNDING_PROVIDER = provider.lower()

# Google thinking budget
GOOGLE_THINKING_BUDGET = 8192  # -1=dynamic, N=fixed budget (0 invalid for thinking models)

# Evaluation settings
NUM_QUESTIONS = 2
ELO_INITIAL_RATING = 1500
ELO_K_FACTOR = 32

# Phase toggles
PHASE2_WEB_SEARCH = True   # Web grounding for answering (current events only)
PHASE3_WEB_SEARCH = False  # Web grounding for evaluation (reuses Phase 2 data)


def get_phase2_web_search() -> bool:
    """Get Phase 2 web search setting."""
    return PHASE2_WEB_SEARCH


def set_phase2_web_search(enabled: bool):
    """Set Phase 2 web search setting."""
    global PHASE2_WEB_SEARCH
    PHASE2_WEB_SEARCH = enabled


def get_phase3_web_search() -> bool:
    """Get Phase 3 web search setting."""
    return PHASE3_WEB_SEARCH


def set_phase3_web_search(enabled: bool):
    """Set Phase 3 web search setting."""
    global PHASE3_WEB_SEARCH
    PHASE3_WEB_SEARCH = enabled


# Phase 3 runs 3 bias modes automatically (shuffle_only, blind_only, shuffle_blind)
BIAS_TEST_SEED = None  # Random seed for reproducible shuffling (None = random)

# How many models a judge may omit from one evaluation before that evaluation is
# discarded. 0 = strict (every response must be scored).
#
# Set to 1 on evidence from the z1 run: of 20 genuine single-omission evaluations,
# HALF came from one judge (gemini-3.7-flash) while the omitted models were spread
# evenly across 8 targets, and none coincided with a blank/refused answer. So the
# omission is a property of the JUDGE, not of the model being skipped — discarding
# the whole evaluation threw away 11 valid scores AND removed half of that judge's
# ballots, quietly reweighting the judge panel (judges differ in generosity).
# Keeping the 11 is the smaller distortion. Raise the bar back to 0 to reproduce
# strict runs; anything above 1 is not recommended (a response missing several
# entries usually signals a malformed reply, not a slip).
PHASE3_MAX_MISSING_SCORES = 1

# Phase 5 judge model (provider, model_id, display_name)
PHASE5_JUDGE = ("openai", "gpt-5.6", "gpt-5.6")


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

# Active models for PeerRank (tuple format for backward compat)
PEERRANK_MODELS = [(m["provider"], m["model_id"], m["name"]) for m in ALL_MODELS if m["peerrank"]]
MODELS = PEERRANK_MODELS.copy()

# Provider key -> display label (for figures and clustering analysis).
# 'together' maps to Meta because the label names the model family, not the host.
PROVIDER_LABELS = {
    'openai': 'OpenAI', 'anthropic': 'Anthropic', 'google': 'Google',
    'grok': 'xAI', 'deepseek': 'DeepSeek', 'together': 'Meta',
    'kimi': 'Moonshot', 'mistral': 'Mistral', 'perplexity': 'Perplexity',
    'minimax': 'MiniMax',
}

# Model display name -> provider label, derived from ALL_MODELS so that renaming
# a model cannot silently drop it out of provider clustering (it did: gpt-5.6-sol,
# claude-fable-5, gemini-3.7-flash and grok-4.6 were all missing before 2026-08-24).
PROVIDER_MAP = {m['name']: PROVIDER_LABELS.get(m['provider'], m['provider'].title())
                for m in ALL_MODELS}

# Short names for compact display
MODEL_SHORTCUTS = {
    'gpt-5.6-sol': 'gpt-sol', 'gpt-5.6-terra': 'gpt-terra', 'gpt-5.6-luna': 'gpt-luna',
    'claude-fable-5': 'fable-5', 'claude-opus-5': 'opus-5', 'claude-sonnet-5': 'sonnet-5',
    'claude-haiku-4-5': 'haiku-4.5',
    'gemini-3.1-pro-preview': 'gem-3.1-pro', 'gemini-3.7-flash': 'gem-3.7-fl',
    'grok-4.6': 'grok-4.6', 'grok-code-fast-1': 'grok-code',
    'deepseek-v4-flash': 'deepseek', 'deepseek-v4-pro': 'deepseek-pro',
    'llama-3.3-70b': 'llama-3.3', 'kimi-k3': 'kimi', 'mistral-large': 'mistral',
    'pplx-agent-medium': 'pplx-med',
}


def get_short_name(model: str, max_len: int = 12) -> str:
    """Get short display name for a model."""
    return MODEL_SHORTCUTS.get(model, model)[:max_len]


# Google service account config (set via environment variables)
_google_sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
GOOGLE_SERVICE_ACCOUNT_FILE = Path(_google_sa_path) if _google_sa_path else None
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "global")


# Provider concurrency limits (max concurrent requests per provider)
# Used for parallel model processing in Phase 2 and Phase 3
PROVIDER_CONCURRENCY = {
    "openai": 8,
    "anthropic": 8,
    "google": 3,  # Reduced to avoid MAX_TOKENS errors with thinking models
    "grok": 8,
    "deepseek": 8,
    "together": 8,
    "perplexity": 1,  # Agent API allows only 1 in-flight request: measured 2026-08-24, 2 concurrent -> 1x429, 8 concurrent -> 7x429 (request_rate_limit_exceeded, no Retry-After header). Sequential back-to-back is fine.
    "kimi": 10,
    "mistral": 2,  # Mistral tier has strict per-minute rate limits
    "minimax": 8,
}


def set_active_models(include: list[str] | None = None, exclude: list[str] | None = None):
    """Filter which models participate in the run."""
    filtered = PEERRANK_MODELS.copy()
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
    return [m[2] for m in PEERRANK_MODELS]


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
        "mistral": "MISTRAL_API_KEY", "minimax": "MINIMAX_API_KEY",
    }
    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.getenv(env_var)
    # Fallback for grok: try GROK_API_KEY if XAI_API_KEY not set
    if not key and provider == "grok":
        key = os.getenv("GROK_API_KEY")
    if not key and provider == "google" and GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_SERVICE_ACCOUNT_FILE.exists():
        return ""
    if not key:
        raise ValueError(f"{env_var} not set")
    return key


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.1f}s"


def _balanced_json_span(s: str, start: int) -> str | None:
    """Substring from `start` to its matching bracket, tracking strings/escapes.

    Returns the remainder if the structure is unterminated (so the caller can still
    attempt a repair); returns None only if `start` is not a bracket.
    """
    open_ch = s[start]
    if open_ch not in "{[":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s[start:]  # unterminated (e.g. truncated output)


def _fix_stray_parens(s: str, repl: str) -> str:
    """Replace ')' outside string literals with `repl` ('}' or ']').

    A parenthesis is never valid JSON structure, so one appearing outside a string
    is a mistyped closer. Observed from a judge: '"flags": ["incorrect"),' where a
    '}' was meant. Content inside strings is left untouched.
    """
    out = []
    in_str = esc = False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch == ')':
            out.append(repl)
        else:
            out.append(ch)
    return "".join(out)


def _close_json(s: str) -> str | None:
    """Best-effort repair of JSON whose closing bracket(s) the model omitted.

    Strips a code-fence wrapper and any leading prose, drops a trailing comma, and
    appends closers for still-open brackets. Returns None if the text is truncated
    mid-string, or if nothing is open to close (the failure isn't a missing bracket
    and must not be silently "repaired" here).
    """
    s = s.strip()
    if s.startswith("```"):
        s = max(s.split("```"), key=len).strip()
        if s[:4].lower() == "json":
            s = s[4:].strip()
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        return None
    s = s[min(starts):]
    stack = []
    in_str = esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    if in_str or not stack:
        return None
    body = s.rstrip()
    if body.endswith(","):
        body = body[:-1].rstrip()
    return body + "".join("}" if c == "{" else "]" for c in reversed(stack))


def extract_json(text: str) -> dict | list | None:
    """Extract JSON from text that may contain markdown or extra prose.

    Tolerant of a leading/trailing prose wrapper and of trailing commas. When the
    text is not directly parseable, the '{...}' and '[...]' spans are both tried
    and the longest one that parses wins, so an incidental empty "[]" (e.g. a
    ``"flags": []`` value) can never shadow the real top-level object, while a
    genuine top-level array still beats any nested object.
    """
    if not text:
        return None
    text = text.strip()

    def _loads(s: str):
        """Strict parse, then light repairs for the ways judges mistype JSON:
        a trailing comma, and ')' used where '}' or ']' belongs."""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        no_comma = re.sub(r',(\s*[}\]])', r'\1', s)  # trailing commas before } or ]
        for cand in (no_comma,
                     _fix_stray_parens(s, '}'),
                     _fix_stray_parens(s, ']'),
                     _fix_stray_parens(no_comma, '}')):
            if cand == s:
                continue
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
        return None

    # Direct parse
    r = _loads(text)
    if r is not None:
        return r

    # From markdown code blocks
    if "```" in text:
        for part in text.split("```"):
            clean = part.strip().removeprefix("json").removeprefix("JSON").strip()
            r = _loads(clean)
            if r is not None:
                return r

    # Scan for one or more consecutive top-level JSON values, using string-aware
    # balanced matching. This captures a full object even when it nests arrays
    # (e.g. "flags": []) — a nested "[]" can never shadow the real object — and
    # critically handles a judge that emits ONE OBJECT PER LINE instead of a single
    # combined object (observed with thinking models): all such objects are merged.
    # Only CONSECUTIVE top-level values are collected: between values we skip
    # whitespace/commas only. Walking over arbitrary text would descend into the
    # inner '{"score": ...}' objects of a malformed response and mask the real shape.
    values = []
    n = len(text)
    # Start at the first bracket, so a leading prose preamble is skipped.
    starts = [p for p in (text.find('{'), text.find('[')) if p != -1]
    i = min(starts) if starts else n
    while i < n and text[i] in "{[":
        span = _balanced_json_span(text, i)
        if not span:
            break
        v = _loads(span)
        if v is None:
            break  # a broken value stops the scan
        values.append(v)
        i += len(span)
        while i < n and text[i] in " \t\r\n,":
            i += 1

    # Before accepting a parsed result, handle the "early outer close" shape: the
    # model emitted '}}' where only '}' belongs, closing the outer object early and
    # leaving the rest as bare '"key": {...}' pairs. Two variants are observed:
    # once after the first entry ('{"a": {...}},\n"b": {...}, ...') and after EVERY
    # entry ('{"a": {..}}, "b": {..}}, "c": {..}}, ...'), so every such boundary is
    # spliced, not just the first. Parsing "succeeds" on the truncated prefix, so the
    # repair is accepted only when it recovers strictly more keys — which also keeps
    # a legitimately nested object (whose '}},"' is real) from being mangled.
    if values and isinstance(values[0], dict):
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end > start:
            span = text[start:end + 1]
            spliced = re.sub(r'\}\}\s*,\s*(?=")', '},\n', span)
            if spliced != span:
                merged_len = sum(len(v) for v in values if isinstance(v, dict))
                for cand in (spliced, spliced + '}', spliced.rstrip().rstrip(',') + '}'):
                    alt = _loads(cand)
                    if isinstance(alt, dict) and len(alt) > merged_len:
                        return alt

    if values:
        if len(values) == 1:
            return values[0]
        # Multiple top-level values: merge if all objects (per-line output);
        # otherwise fall back to the first parsed value.
        if all(isinstance(v, dict) for v in values):
            merged = {}
            for v in values:
                merged.update(v)
            return merged
        return values[0]

    # Last resort A: the model omitted the closing bracket(s) — an unterminated
    # object/array, sometimes inside a code fence. Append the missing closers.
    repaired = _close_json(text)
    if repaired is not None:
        r = _loads(repaired)
        if r is not None:
            return r

    # Last resort B: same "early outer close" shape as above, for text where nothing
    # parsed at all. Splice every '}},"' boundary down to '},"'.
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        span = text[start:end + 1]
        spliced = re.sub(r'\}\}\s*,\s*(?=")', '},\n', span)
        if spliced != span:
            for cand in (spliced, spliced + '}', spliced.rstrip().rstrip(',') + '}'):
                r = _loads(cand)
                if isinstance(r, dict):
                    return r
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


def _record_score(score, model_name: str, evaluator: str,
                  peer_scores: dict, self_scores: dict, raw_scores: dict, judge_given: dict):
    """Helper to record a score in the appropriate buckets."""
    # Convert score to float in case it's a string or int
    try:
        if isinstance(score, (int, float)):
            score = float(score)
        elif isinstance(score, str) and score.strip().replace('.', '').isdigit():
            score = float(score.strip())
        else:
            print(f"  [WARN] Malformed score for {model_name} by {evaluator}: {str(score)[:50]}", flush=True)
            return
    except (ValueError, TypeError):
        print(f"  [WARN] Invalid score for {model_name} by {evaluator}: {str(score)[:50]}", flush=True)
        return

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


def _spearman_correlation(x: list, y: list) -> float:
    """Calculate Spearman rank correlation coefficient between two lists."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    def _rank(data: list) -> list:
        """Convert values to ranks (1-based, average for ties)."""
        sorted_indices = sorted(range(len(data)), key=lambda i: data[i], reverse=True)
        ranks = [0.0] * len(data)
        i = 0
        while i < len(sorted_indices):
            # Find all tied values
            j = i
            while j < len(sorted_indices) and data[sorted_indices[j]] == data[sorted_indices[i]]:
                j += 1
            # Assign average rank to all tied values
            avg_rank = (i + 1 + j) / 2  # 1-based ranks
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank
            i = j
        return ranks

    # Convert to ranks and compute Pearson on ranks
    return _pearson_correlation(_rank(x), _rank(y))


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
        model_id: Model identifier (e.g., "gpt-5.5", "claude-opus-4-7")
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


def _elo_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A given ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _convert_to_pairwise_matches(evaluations: dict, model_names: list[str], exclude_self: bool = True) -> list[tuple]:
    """
    Convert evaluation scores to pairwise match results.

    For each (evaluator, question), generate C(N,2) pairwise comparisons.
    score_a > score_b → A wins (1.0, 0.0)
    score_a < score_b → B wins (0.0, 1.0)
    score_a == score_b → tie (0.5, 0.5)

    Args:
        evaluations: Dict of {evaluator: {question: {model: {score, reason}}}}
        model_names: List of model display names
        exclude_self: If True, skip evaluator's own ratings

    Returns:
        List of (model_a, model_b, outcome_a, outcome_b) tuples
    """
    matches = []

    for evaluator_name, questions in evaluations.items():
        matched_evaluator = match_model_name(evaluator_name) if exclude_self else None

        for question_id, model_scores in questions.items():
            # Extract scores for this question
            scores = {}
            for model_name, score_data in model_scores.items():
                if isinstance(score_data, dict) and "score" in score_data:
                    matched = match_model_name(model_name)
                    if matched and matched in model_names:
                        # Skip self-evaluations if requested
                        if exclude_self and matched == matched_evaluator:
                            continue
                        # Safely convert score to float (handle malformed responses)
                        try:
                            score_val = score_data["score"]
                            if isinstance(score_val, (int, float)):
                                scores[matched] = float(score_val)
                            elif isinstance(score_val, str) and score_val.strip().isdigit():
                                scores[matched] = float(score_val.strip())
                            else:
                                # Non-numeric score value
                                print(f"  [WARN] Malformed score from {evaluator_name} for {model_name}: {str(score_val)[:50]}", flush=True)
                        except (ValueError, TypeError) as e:
                            print(f"  [WARN] Invalid score from {evaluator_name} for {model_name}: {e}", flush=True)

            # Generate pairwise comparisons
            scored_models = list(scores.keys())
            for i, model_a in enumerate(scored_models):
                for model_b in scored_models[i + 1:]:
                    score_a, score_b = scores[model_a], scores[model_b]

                    if score_a > score_b:
                        outcome_a, outcome_b = 1.0, 0.0
                    elif score_a < score_b:
                        outcome_a, outcome_b = 0.0, 1.0
                    else:
                        outcome_a, outcome_b = 0.5, 0.5

                    matches.append((model_a, model_b, outcome_a, outcome_b))

    return matches


def calculate_elo_ratings(
    evaluations: dict,
    model_names: list[str] = None,
    initial_rating: int = None,
    k_factor: int = None,
    exclude_self: bool = True,
    seed: int = None
) -> dict:
    """
    Calculate Elo ratings from pairwise comparisons in evaluation data.

    Args:
        evaluations: Dict of {evaluator: {question: {model: {score, reason}}}}
        model_names: List of model display names (defaults to MODELS)
        initial_rating: Starting Elo rating (defaults to ELO_INITIAL_RATING)
        k_factor: K-factor for rating updates (defaults to ELO_K_FACTOR)
        exclude_self: If True, exclude self-evaluations
        seed: Random seed for match processing order

    Returns:
        Dict with:
        - 'ratings': {model: final_elo_rating}
        - 'matches': {model: (wins, losses, ties)}
        - 'win_rates': {model: win_percentage}
        - 'total_matches': Total number of pairwise comparisons
    """
    import random

    if model_names is None:
        model_names = [n for _, _, n in MODELS]
    if initial_rating is None:
        initial_rating = ELO_INITIAL_RATING
    if k_factor is None:
        k_factor = ELO_K_FACTOR

    # Initialize ratings
    ratings = {name: initial_rating for name in model_names}
    matches = {name: [0, 0, 0] for name in model_names}  # [wins, losses, ties]

    # Convert evaluations to pairwise matches
    pairwise_matches = _convert_to_pairwise_matches(evaluations, model_names, exclude_self)

    # Always shuffle matches to avoid order-dependent bias
    # Use provided seed for reproducibility, or default seed for consistency
    shuffle_seed = seed if seed is not None else 42
    random.seed(shuffle_seed)
    random.shuffle(pairwise_matches)

    # Process each match
    for model_a, model_b, outcome_a, outcome_b in pairwise_matches:
        # Calculate expected scores
        expected_a = _elo_expected_score(ratings[model_a], ratings[model_b])
        expected_b = 1 - expected_a

        # Update ratings
        ratings[model_a] += k_factor * (outcome_a - expected_a)
        ratings[model_b] += k_factor * (outcome_b - expected_b)

        # Track match outcomes
        if outcome_a == 1.0:
            matches[model_a][0] += 1  # win
            matches[model_b][1] += 1  # loss
        elif outcome_a == 0.0:
            matches[model_a][1] += 1  # loss
            matches[model_b][0] += 1  # win
        else:
            matches[model_a][2] += 1  # tie
            matches[model_b][2] += 1  # tie

    # Round ratings to integers
    ratings = {name: int(round(rating)) for name, rating in ratings.items()}

    # Calculate win rates
    win_rates = {}
    for name, (wins, losses, ties) in matches.items():
        total = wins + losses + ties
        if total > 0:
            # Win rate: wins + 0.5*ties / total
            win_rates[name] = (wins + 0.5 * ties) / total * 100
        else:
            win_rates[name] = 0.0

    return {
        "ratings": ratings,
        "matches": {name: tuple(m) for name, m in matches.items()},
        "win_rates": win_rates,
        "total_matches": len(pairwise_matches),
    }
