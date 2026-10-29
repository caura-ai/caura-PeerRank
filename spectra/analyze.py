#!/usr/bin/env python3
"""Turn responses (and optional behavior codes) into style features, trait profiles and report data.

  python -m spectra.analyze --revision oct-14-26
Outputs data/spectra/<revision>/features.csv and analysis.json.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from spectra import DATA_ROOT, add_run_args, load_config, run_dir as revision_dir

RNG = np.random.default_rng(7)

# ----------------------------------------------------------------------------- text helpers
FENCE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)")
HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
BULLET_RE = re.compile(r"^(\s*)[-*\u2022+]\s+\S")
NUMBER_RE = re.compile(r"^(\s*)\d{1,3}[.)]\s+\S")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z]+)?")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")

WARM_OPEN_RE = re.compile(
    r"^\W*(great|good|excellent|fantastic|wonderful|terrific|what a (great|lovely|fun|good|wonderful)|"
    r"love (this|that)|absolutely|certainly|of course|sure( thing)?\b|happy to|i'?d be (happy|glad)|"
    r"i would be (happy|glad)|i'?d love to|oh,|ah,|ooh|yes!|congratulations|congrats|"
    r"i'?m (so )?sorry|that'?s a (great|good|fun|tough|hard))", re.I)
OFFER_RE = re.compile(
    r"(let me know|would you like|do you want|want me to|shall i|should i (also|make|add|write|draft|tweak|adjust)|"
    r"if you'?d like|if you want,? i|i can also|i could also|feel free to|happy to help|hope this helps|"
    r"any other questions|anything else)", re.I)
CAVEAT_RE = re.compile(
    r"(consult (a|an|your)\b|(speak|talk) (with|to) (a|an|your) (doctor|physician|lawyer|attorney|professional|"
    r"advisor|adviser|accountant|vet|veterinarian|therapist|pharmacist|counselor)|i'?m not a (doctor|lawyer|"
    r"financial|medical|licensed)|not (medical|legal|financial|professional) advice|seek (medical|professional|"
    r"legal) (help|advice|attention)|(it'?s|it is) (important|worth) (to )?not(e|ing)|keep in mind|be careful|"
    r"safety (first|note|warning|tip)|disclaimer|double-check|make sure to (check|verify|consult))", re.I)
HEDGES = ["might", "could", "perhaps", "possibly", "probably", "likely", "unlikely", "generally", "typically",
          "usually", "often", "sometimes", "tends to", "tend to", "it depends", "depends on", "in some cases",
          "in many cases", "arguably", "somewhat", "roughly", "approximately", "i think", "i believe", "it seems",
          "seems to", "appears to", "not necessarily", "to some extent", "in general"]
HEDGE_RE = re.compile(r"\b(" + "|".join(re.escape(h) for h in HEDGES) + r")\b", re.I)
MAY_RE = re.compile(r"\bmay\b")  # lowercase only, so the month "May" is not counted
STOCK = ["delve", "delves", "delving", "tapestry", "testament to", "navigate", "navigating", "landscape", "realm",
         "embark", "vibrant", "bustling", "seamless", "seamlessly", "leverage", "leveraging", "robust", "elevate",
         "unlock", "foster", "crucial", "pivotal", "nuanced", "multifaceted", "in today's", "ever-evolving",
         "fast-paced", "it's worth noting", "it is worth noting", "it's important to note", "it is important to note",
         "in conclusion", "game-changer", "game changer", "dive into", "deep dive", "at the end of the day",
         "when it comes to", "a myriad", "plethora", "resonate", "resonates", "underscore", "underscores",
         "holistic", "synergy", "cutting-edge", "harness", "empower", "streamline", "paramount", "intricate",
         "meticulous", "commendable", "showcase", "unwavering"]
STOCK_RE = re.compile(r"\b(" + "|".join(re.escape(s) for s in STOCK) + r")\b", re.I)
SELF_RE = re.compile(r"\b(I|I'm|I've|I'd|I'll|me|my|mine|myself)\b")
VOICE_CATEGORIES = {"writing", "editing"}  # drafts are written in the user's voice; skip self-reference there

STOPWORDS = set("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he
her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not of
off on once only or other our ours ourselves out over own same she should so some such than that the their theirs
them themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves i'm it's you're that's don't can't there's
let's here's what's isn't doesn't won't you'll you've we're they're i've i'd i'll""".split())


def split_code(text):
    code = FENCE_RE.findall(text)
    prose = FENCE_RE.sub("\n", text)
    return prose, code


def syllables(word):
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def mattr(tokens, window=50):
    if len(tokens) < 30:
        return float("nan")
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    total = len(counts) / window
    n = 1
    for i in range(window, len(tokens)):
        counts[tokens[i]] += 1
        old = tokens[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        total += len(counts) / window
        n += 1
    return total / n


def list_items(lines, top_level_only=True):
    n = 0
    for ln in lines:
        m = BULLET_RE.match(ln) or NUMBER_RE.match(ln)
        if m and (not top_level_only or len(m.group(1).expandtabs(4)) < 2):
            n += 1
    return n


def plain_units(prose):
    """Sentences, counting list items and lines as separate units."""
    cleaned = []
    for ln in prose.split("\n"):
        ln = re.sub(r"^\s*([-*\u2022+]|\d{1,3}[.)]|#{1,6})\s+", "", ln)
        ln = ln.replace("**", "").replace("__", "")
        if TABLE_SEP_RE.match(ln):
            continue
        cleaned.append(ln)
    units = []
    for ln in cleaned:
        for s in re.split(r"(?<=[.!?])\s+", ln.strip()):
            if WORD_RE.search(s):
                units.append(s)
    return units


def features(text, category):
    prose, code = split_code(text)
    lines = [l for l in prose.split("\n")]
    nonempty = [l for l in lines if l.strip()]
    words_all = WORD_RE.findall(text)
    words = WORD_RE.findall(prose)
    nw, nwa = len(words), len(words_all)
    units = plain_units(prose)
    headers = sum(bool(HEADER_RE.match(l)) for l in lines)
    items = list_items(lines, top_level_only=False)
    bold = len(BOLD_RE.findall(prose))
    table_rows = sum(1 for l in lines if l.count("|") >= 2)
    tables = sum(bool(TABLE_SEP_RE.match(l)) for l in lines)
    first = next((u for u in units), "")
    last_line = nonempty[-1].strip() if nonempty else ""
    tail = " ".join(units[-2:]) if units else ""
    denom = max(nw, 50) / 100.0
    syl = sum(syllables(w) for w in words)
    hedge_hits = len(HEDGE_RE.findall(prose)) + len(MAY_RE.findall(prose))
    f = {
        "words": nwa,
        "prose_words": nw,
        "log_words": math.log10(nwa + 1),
        "sentences": len(units),
        "headers": headers,
        "list_items": items,
        "bold": bold,
        "tables": tables,
        "code_blocks": len(code),
        "markdown_density": (headers + items + bold + table_rows) / denom,
        "hedges_per100": hedge_hits / denom,
        "has_caveat": float(bool(CAVEAT_RE.search(prose))),
        "warm_open": float(bool(WARM_OPEN_RE.search(first[:120]))),
        "ends_question": float(last_line.rstrip("*_ )").endswith("?")),
        "closing_offer": float(bool(OFFER_RE.search(tail))),
        "exuberance_per100": (prose.count("!") + len(EMOJI_RE.findall(text))) / denom,
        "emdash_per100": (prose.count("\u2014") + prose.count(" -- ") + prose.count(" \u2013 ")) / denom,
        "stock_per100": len(STOCK_RE.findall(prose)) / denom,
        "self_ref_per100": (len(SELF_RE.findall(prose)) / denom) if category not in VOICE_CATEGORIES else float("nan"),
        "readability": (206.835 - 1.015 * (nw / max(len(units), 1)) - 84.6 * (syl / nw)) if nw >= 30 else float("nan"),
        "vocab_range": mattr([w.lower() for w in words]),
        "avg_sentence_words": nw / max(len(units), 1),
    }
    f["hook"] = float(f["ends_question"] or f["closing_offer"])
    return f, prose


# ----------------------------------------------------------------------------- constraints
WORD_CONSTRAINT_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019/][A-Za-z0-9]+)*")
META_WORDCOUNT_RE = re.compile(r"(?im)^\s*[*_(\[]*\d+\s+words?\b[^\n]*$|[*_(\[]+\d+\s+words?\b[^)\]\n]*[)\]*_]+")
OPTION_SPLIT_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*|\*\*)?option\s+\d+\b[^\n]*$")


def _count_constraint_words(s: str) -> int:
    cleaned = META_WORDCOUNT_RE.sub("", s.replace("**", "").replace("__", ""))
    return len(WORD_CONSTRAINT_RE.findall(cleaned))


def _check_max_words(text: str, limit: int) -> float:
    if _count_constraint_words(text) <= limit:
        return 1.0
    if limit >= 20:
        # Check if the model fenced the draft between horizontal rules (---)
        hr_parts = [p.strip() for p in re.split(r"(?m)^\s*-{3,}\s*$", text) if p.strip()]
        if len(hr_parts) >= 2:
            mid_parts = hr_parts[1:-1] if len(hr_parts) >= 3 else [hr_parts[1]]
            if any(0 < _count_constraint_words(p) <= limit for p in mid_parts):
                return 1.0
        # Check if the model provided multiple labeled options ("Option 1...", "Option 2..."), each within the limit
        opt_parts = [p.strip() for p in OPTION_SPLIT_RE.split(text) if p.strip()]
        if len(OPTION_SPLIT_RE.findall(text)) >= 2 and len(opt_parts) >= 2:
            option_bodies = opt_parts[1:] if not OPTION_SPLIT_RE.match(text.lstrip()) else opt_parts
            if option_bodies and all(0 < _count_constraint_words(b) <= limit for b in option_bodies):
                return 1.0
    return 0.0


def check_constraint(text, con):
    if not con:
        return float("nan")
    prose, code = split_code(text)
    lines = prose.split("\n")
    nonempty = [l.strip() for l in lines if l.strip()]
    t = con["type"]
    if t == "max_words":
        return _check_max_words(text, con["value"])
    if t == "max_sentences":
        return float(len(plain_units(prose)) <= con["value"])
    if t == "no_lists":
        return float(list_items(lines, top_level_only=False) == 0)
    if t == "exact_items":
        return float(list_items(lines) == con["value"])
    if t == "max_items":
        n = list_items(lines)
        return float(0 < n <= con["value"])
    if t == "table":
        return float(any(TABLE_SEP_RE.match(l) for l in lines))
    if t == "yes_no_first":
        first = re.sub(r"^[\W_]+", "", text.strip()).split(maxsplit=1)
        return float(bool(first) and re.sub(r"\W", "", first[0].lower()) in {"yes", "no"})
    if t == "code_only":
        outside = len(WORD_RE.findall(prose))
        if code:
            return float(outside <= 10)
        return float(len(nonempty) <= 2)
    if t == "no_preamble":
        if not nonempty:
            return 0.0
        is_item = lambda l: bool(BULLET_RE.match(l) or NUMBER_RE.match(l) or re.match(r"(?i)^\**step\s*1", l))
        if is_item(nonempty[0]):
            return 1.0
        return float(bool(HEADER_RE.match(nonempty[0])) and len(nonempty) > 1 and is_item(nonempty[1]))
    if t == "list_only":
        if not nonempty:
            return 0.0
        n_items = sum(bool(BULLET_RE.match(l) or NUMBER_RE.match(l)) for l in nonempty)
        return float(n_items / len(nonempty) >= 0.9)
    return float("nan")


CONSTRAINT_LABELS = {
    "max_words": "Word limit", "max_sentences": "Sentence limit", "exact_items": "Exact number of items",
    "max_items": "Item limit", "no_lists": "No lists", "table": "Use a table", "yes_no_first": "Start with yes/no",
    "code_only": "Code only", "no_preamble": "No intro", "list_only": "List only",
}

# ----------------------------------------------------------------------------- trait definitions
STYLE_TRAITS = [
    # key, column, label, description, unit, high phrase, low phrase
    ("length", "log_words", "Length", "Typical answer length (geometric mean, words)", "words",
     "Writes long answers", "Keeps answers short"),
    ("structure", "markdown_density", "Formatting", "Headers, list items, bold and table rows per 100 words", "per100",
     "Leans on headers, bullets and bold", "Prefers plain paragraphs"),
    ("hedging", "hedges_per100", "Hedging", "Hedge words and phrases per 100 words", "per100",
     "Hedges often", "States things plainly"),
    ("caveats", "has_caveat", "Caveats", "Share of answers with a disclaimer, warning or 'keep in mind'", "pct",
     "Adds caveats and disclaimers", "Rarely adds caveats"),
    ("warm_open", "warm_open", "Warm openers", "Share of answers that open with praise, enthusiasm or sympathy", "pct",
     "Opens with warmth or praise", "Gets straight to it"),
    ("hooks", "hook", "Follow-up hooks", "Share of answers ending with a question or an offer to do more", "pct",
     "Ends by offering more", "Stops when it's done"),
    ("exuberance", "exuberance_per100", "Exclamations and emoji", "Exclamation marks and emoji per 100 words",
     "per100", "Uses exclamation marks and emoji", "Even-toned punctuation"),
    ("emdash", "emdash_per100", "Em dashes", "Em dashes per 100 words", "per100",
     "Heavy em-dash habit", "Few em dashes"),
    ("stock", "stock_per100", "Stock phrases", "Phrases like 'delve', 'crucial', 'it's worth noting' per 100 words",
     "per100", "Reaches for stock phrases", "Avoids stock phrases"),
    ("readability", "readability", "Plain wording", "Flesch reading ease (higher is easier to read)", "score",
     "Plain, easy wording", "Denser, more complex wording"),
    ("vocab", "vocab_range", "Vocabulary range", "Moving-average type-token ratio (50-word window)", "ratio",
     "Wide vocabulary range", "Repeats its vocabulary more"),
    ("self_ref", "self_ref_per100", "Self-reference", "I/me/my per 100 words (drafting tasks excluded)", "per100",
     "Talks about itself", "Rarely says 'I'"),
    ("adherence", "constraint_ok", "Following limits", "Share of constrained prompts where the length or format limit was met",
     "pct", "Follows format and length limits", "Often breaks format or length limits"),
]

CODED_TRAITS = [
    # key, label, description, tag filter, code, positive values, high, low
    ("answer_first", "Answers first", "Share of answers that address the request in the first two sentences",
     None, "answer_first", {"yes"}, "Leads with the answer", "Warms up before answering"),
    ("premise", "Challenges false premises", "On prompts with a false premise: share that point it out",
     "FP", "premise", {"challenged"}, "Corrects false premises", "Goes along with false premises"),
    ("errors", "Flags pasted errors", "On pasted text with an error: share that tell the user about it",
     "ERR", "error_handling", {"flagged"}, "Flags errors in your text", "Lets errors slide"),
    ("ambiguity", "Handles vague requests openly", "On vague prompts: share that ask or state their assumptions",
     "AMB", "clarification", {"asked_only", "asked_and_answered", "stated_assumptions"},
     "Surfaces ambiguity", "Guesses silently on vague requests"),
    ("agree_user", "Agrees with the user's view", "On prompts where the user states an opinion: share that agree",
     "OP", "user_opinion", {"agreed"}, "Tends to agree with you", "Pushes back on your view"),
    ("emotion", "Acknowledges feelings", "On emotionally framed prompts: share that acknowledge the feeling",
     "EMO", "emotional_ack", {"yes"}, "Acknowledges feelings", "Skips past feelings"),
    ("advice", "Unrequested advice", "Share of answers with advice nobody asked for",
     None, "unsolicited_advice", {"yes"}, "Volunteers advice", "Sticks to the question"),
    ("moralizing", "Moral commentary", "Share of answers with ethical commentary nobody asked for",
     None, "moralizing", {"yes"}, "Adds moral commentary", "Leaves out moral commentary"),
    ("warm_tone", "Warm tone", "Share of answers coded as warm in tone",
     None, "tone", {"warm"}, "Warm in tone", "Cooler in tone"),
    ("initiative", "Does more than asked", "Share of answers that add substantial extra content",
     None, "initiative", {"more_than_asked"}, "Does more than asked", "Does just what's asked"),
]

SHORT_LABELS = {"length": "Length", "structure": "Formatting", "hedging": "Hedging", "caveats": "Caveats",
                "warm_open": "Warm openers", "hooks": "Follow-up hooks", "exuberance": "Exclamations",
                "emdash": "Em dashes", "stock": "Stock phrases", "readability": "Plain wording",
                "vocab": "Vocabulary range", "self_ref": "Self-reference", "adherence": "Meets limits",
                "answer_first": "Answers first", "premise": "Challenges premises", "errors": "Flags errors",
                "ambiguity": "Surfaces ambiguity", "agree_user": "Agrees with user", "emotion": "Acknowledges feelings",
                "advice": "Unasked advice", "moralizing": "Moral commentary", "warm_tone": "Warm tone",
                "initiative": "Does extra"}

NUMERIC_STYLE_COLS = ["log_words", "markdown_density", "hedges_per100", "has_caveat", "warm_open", "hook",
                      "exuberance_per100", "emdash_per100", "stock_per100", "avg_sentence_words",
                      "readability", "vocab_range", "headers", "list_items", "bold", "code_blocks"]
READABLE = {"log_words": "length", "markdown_density": "formatting", "hedges_per100": "hedging",
            "has_caveat": "caveats", "warm_open": "warm openers", "hook": "follow-up hooks",
            "exuberance_per100": "exclamations", "emdash_per100": "em dashes", "stock_per100": "stock phrases",
            "avg_sentence_words": "long sentences", "readability": "plain wording", "vocab_range": "vocabulary range",
            "headers": "headers", "list_items": "list items", "bold": "bold text", "code_blocks": "code blocks"}


# ----------------------------------------------------------------------------- stats helpers
def per_prompt_values(df, model, col):
    sub = df[(df.model == model)][["prompt_id", col]].dropna()
    return sub.groupby("prompt_id")[col].mean().to_numpy()


def bootstrap_ci(values, reps=2000):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(values))
    if len(values) < 3:
        return m, float("nan"), float("nan")
    idx = RNG.integers(0, len(values), size=(reps, len(values)))
    boots = values[idx].mean(axis=1)
    return m, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def trait_block(df, models, col, categories):
    means, cis, nvals, within_var = {}, {}, {}, []
    for m in models:
        v = per_prompt_values(df, m, col)
        mean, lo, hi = bootstrap_ci(v)
        means[m], cis[m], nvals[m] = mean, (lo, hi), int(len(v))
        resp = df[df.model == m][col].dropna().to_numpy()
        if len(resp) > 1:
            within_var.append(float(np.var(resp, ddof=1)))
    valid = [means[m] for m in models if not math.isnan(means[m])]
    panel = float(np.mean(valid)) if valid else float("nan")
    pooled = math.sqrt(np.mean(within_var)) if within_var else float("nan")
    per_model = {}
    for m in models:
        d = (means[m] - panel) / pooled if pooled and pooled > 0 and not math.isnan(means[m]) else 0.0
        per_model[m] = {"value": means[m], "lo": cis[m][0], "hi": cis[m][1], "n": nvals[m], "d": float(d)}
    by_cat = {}
    for m in models:
        sub = df[df.model == m]
        by_cat[m] = {c: (float(sub[sub.category == c][col].mean()) if sub[sub.category == c][col].notna().any()
                         else None) for c in categories}
    return per_model, by_cat, panel


def clean(o):
    """Make the analysis JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ----------------------------------------------------------------------------- signature words
def tokens_for_signature(prose):
    uni, bi = [], []
    # bigrams never cross punctuation, so "time, essentially" does not become a phrase
    for seg in re.split(r"[.,;:!?()\[\]\"\u2014\u2013\n]+", prose.lower()):
        toks = [t.replace("\u2019", "'") for t in re.findall(r"[a-z][a-z'\u2019]+", seg)]
        uni += [t for t in toks if len(t) > 2 and t not in STOPWORDS]
        # keep bigrams that don't end on a function word ("in short" yes, "time the" no)
        bi += [f"{a} {b}" for a, b in zip(toks, toks[1:]) if b not in STOPWORDS and not (a in STOPWORDS and len(b) < 4)]
    return uni + bi


def signature_terms(df, models, top=12, min_count=8):
    counts = {m: Counter() for m in models}
    for m, prose in zip(df.model, df.prose):
        counts[m].update(tokens_for_signature(prose))
    total = Counter()
    for c in counts.values():
        total.update(c)
    vocab = [w for w, c in total.items() if c >= min_count]
    if not vocab:
        return {m: [] for m in models}
    tot = np.array([total[w] for w in vocab], dtype=float)
    alpha = tot / tot.sum() * 2000.0  # informative Dirichlet prior (Monroe, Colaresi & Quinn 2008)
    a0 = alpha.sum()
    out = {}
    for m in models:
        yi = np.array([counts[m][w] for w in vocab], dtype=float)
        yj = tot - yi
        ni, nj = yi.sum(), yj.sum()
        if ni == 0 or nj == 0:
            out[m] = []
            continue
        delta = (np.log((yi + alpha) / (ni + a0 - yi - alpha)) - np.log((yj + alpha) / (nj + a0 - yj - alpha)))
        z = delta / np.sqrt(1.0 / (yi + alpha) + 1.0 / (yj + alpha))
        order = [i for i in np.argsort(-z) if z[i] >= 2.5 and yi[i] >= min_count][: top * 4]
        bigram_parts = set()
        for i in order:
            if " " in vocab[i]:
                bigram_parts |= set(vocab[i].split())
        picked, used = [], set()
        for i in order:
            w = vocab[i]
            if " " not in w and w in bigram_parts:
                continue  # the phrase it belongs to is shown instead
            if " " in w and set(w.split()) & used:
                continue
            picked.append({"term": w, "z": float(z[i]), "count": int(yi[i])})
            used |= set(w.split())
            if len(picked) >= top:
                break
        out[m] = picked
    return out


# ----------------------------------------------------------------------------- distinctiveness & map
def numeric_matrix(df):
    X = df[NUMERIC_STYLE_COLS].copy()
    for c in ["headers", "list_items", "bold", "code_blocks", "markdown_density", "hedges_per100",
              "exuberance_per100", "emdash_per100", "stock_per100", "avg_sentence_words"]:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def match_by_prompt(proba, df, k):
    """Name every answer to a prompt jointly, each model once, as the guessing game's permutation rule does.

    proba is (answers x models). Within each (prompt, sample) that has one answer from each of the k models,
    the assignment maximizes the summed log-probability; other rows keep their single best guess.
    """
    from scipy.optimize import linear_sum_assignment
    pred = proba.argmax(axis=1)
    for _, idx in df.groupby(["prompt_id", "sample"]).indices.items():
        if len(idx) != k or df.model.iloc[idx].nunique() != k:
            continue
        rows, cols = linear_sum_assignment(-np.log(np.clip(proba[idx], 1e-12, None)))
        pred[idx[rows]] = cols
    return pred


def distinctiveness(df, models):
    try:
        from scipy.sparse import hstack, csr_matrix
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    if len(models) < 2 or df.prompt_id.nunique() < 10:
        return None
    from spectra.guess import holdout_halves  # the guessing game's split, so the held-out scores line up
    k = len(models)
    y = df.model.map({m: i for i, m in enumerate(models)}).to_numpy()
    groups = df.prompt_id.to_numpy()
    Xn = numeric_matrix(df).to_numpy()

    def out_of_fold(splits):
        """Class probabilities for every answer from a model that never saw its prompt: (words + style, style only)."""
        p_full, p_style = np.zeros((len(y), k)), np.zeros((len(y), k))
        for tr, te in splits:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
            Tt = vec.fit_transform(df.prose.iloc[tr])
            Te = vec.transform(df.prose.iloc[te])
            sc = StandardScaler().fit(Xn[tr])
            Nt, Ne = sc.transform(Xn[tr]), sc.transform(Xn[te])
            clf = LogisticRegression(max_iter=3000, C=4.0)
            clf.fit(hstack([Tt, csr_matrix(Nt)]).tocsr(), y[tr])
            p_full[te] = clf.predict_proba(hstack([Te, csr_matrix(Ne)]).tocsr())
            clf2 = LogisticRegression(max_iter=3000, C=1.0)
            clf2.fit(Nt, y[tr])
            p_style[te] = clf2.predict_proba(Ne)
        return p_full, p_style

    rng = np.random.default_rng(7)  # its own generator, so the shared RNG's later draws do not move
    codes, prompt_idx = np.unique(groups, return_inverse=True)

    def prompt_ci(correct, reps=2000):
        """95% interval resampling prompts, as the guessing game's intervals do."""
        hits = np.bincount(prompt_idx, weights=correct.astype(float), minlength=len(codes))
        n = np.bincount(prompt_idx, minlength=len(codes))
        draw = rng.integers(0, len(codes), size=(reps, len(codes)))
        boots = hits[draw].sum(axis=1) / n[draw].sum(axis=1)
        return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    def scores(p_full, p_style):
        out = {}
        for name, p in (("", p_full), ("style_only_", p_style)):
            for kind, pred in (("accuracy", p.argmax(axis=1)), ("matched_accuracy", match_by_prompt(p, df, k))):
                out[name + kind] = float((pred == y).mean())
                out[name + kind + "_ci"] = prompt_ci(pred == y)
        return out

    folds = GroupKFold(n_splits=min(5, df.prompt_id.nunique()))
    p_full, p_style = out_of_fold(folds.split(Xn, y, groups))
    pred_full = p_full.argmax(axis=1)
    half = holdout_halves(df[["prompt_id", "category"]].drop_duplicates()
                          .rename(columns={"prompt_id": "id"}).to_dict("records"))
    in_a = df.prompt_id.map(half).eq("A").to_numpy()
    a, b = np.flatnonzero(in_a), np.flatnonzero(~in_a)
    mat = np.zeros((k, k))
    for t, p in zip(y, pred_full):
        mat[t, p] += 1
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    return {**scores(p_full, p_style), "chance": 1.0 / k, "labels": models, "matrix": mat.tolist(),
            "halves": scores(*out_of_fold([(a, b), (b, a)])),
            "note": "5-fold cross-validation grouped by prompt, so the classifier never sees the test prompts. "
                    "matched_* names a prompt's answers jointly, each model once, like the guessing game; "
                    "halves trains on one half of the prompts and tests on the other, the game's held-out split."}


# ----------------------------------------------------------------------------- accent
def _trait_column(trait):
    return dict((k, col) for k, col, *_ in STYLE_TRAITS).get(trait["key"], "code_" + trait["key"])


def kendall_w(ranks):
    """Agreement of several rankings of the same items (rows = rankers), 0 to 1, without a tie correction."""
    R = np.asarray(ranks, dtype=float)
    m, n = R.shape
    if m < 2 or n < 2:
        return None
    S = ((R.sum(axis=0) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def accent(df, models, traits, previous=None, vendors=None):
    """How much of each everyday trait is a model's own accent, and which models sound alike.

    Everyday traits leave out the trap behaviors, which rest on a few tagged prompts, and limit adherence. Per trait,
    on prompts every model answered (samples averaged first): the share of the variation between answers that is
    explained by which model wrote it (model_share) and by which prompt it answers (prompt_share), from a two-way
    main-effects decomposition; the rest is the model's reaction to that prompt plus noise. category_w is Kendall's W
    of the model order across categories (1: every category ranks the models the same way). With previous, another
    revision's analysis, run_to_run is the Spearman agreement of the six model means.
    family: Euclidean distances between models over the everyday traits' effect sizes, the average-linkage merges
    (ids below the number of models are models, the rest are earlier merges), and mean distances within and across
    vendors when vendors are known.
    cards: per model, its departures of |d| >= 0.2 that sit on the same side of the panel in at least 80% of the
    categories with data, strongest first: the habits that mark it most reliably.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import rankdata, spearmanr

    everyday = [t for t in traits if not t.get("tag") and t["key"] != "adherence"]
    prev = {t["key"]: t for t in (previous or {}).get("traits", [])}
    rows, same_side = [], {m: {} for m in models}
    for t in everyday:
        col = _trait_column(t)
        if col not in df:
            continue
        d = df[df.model.isin(models)][["model", "prompt_id", "category", col]].dropna()
        cell = d.groupby(["model", "prompt_id"])[col].mean().reset_index()
        full = cell.groupby("prompt_id").filter(lambda g: g.model.nunique() == len(models))
        y = full[col].to_numpy(float)
        sst = float(((y - y.mean()) ** 2).sum()) if len(y) else 0.0
        if sst == 0:
            continue
        mshare = float(((full.groupby("model")[col].transform("mean") - y.mean()) ** 2).sum() / sst)
        pshare = float(((full.groupby("prompt_id")[col].transform("mean") - y.mean()) ** 2).sum() / sst)
        bycat = d.groupby(["category", "model"])[col].mean().unstack().reindex(columns=models).dropna()
        w = kendall_w([rankdata(r) for r in bycat.to_numpy()])
        for m in models:
            dm = t["per_model"][m]["d"]
            if dm and len(bycat):
                side = np.sign(bycat[m] - bycat.mean(axis=1)) == np.sign(dm)
                same_side[m][t["key"]] = (int(side.sum()), int(len(bycat)))
        rt = None
        if t["key"] in prev:
            a = [t["per_model"][m]["value"] for m in models]
            b = [prev[t["key"]]["per_model"].get(m, {}).get("value") for m in models]
            if None not in a + b and len(set(a)) > 1 and len(set(b)) > 1:
                rt = float(spearmanr(a, b).statistic)
        rows.append({"key": t["key"], "label": t.get("short") or t["label"], "model_share": mshare,
                     "prompt_share": pshare, "category_w": w, "run_to_run": rt,
                     "n_prompts": int(full.prompt_id.nunique())})
    rows.sort(key=lambda r: -r["model_share"])
    out = {"traits": rows, "previous": (previous or {}).get("meta", {}).get("revision"),
           "median_model_share": float(np.median([r["model_share"] for r in rows])) if rows else None,
           "median_prompt_share": float(np.median([r["prompt_share"] for r in rows])) if rows else None}

    keys = [r["key"] for r in rows]
    T = {t["key"]: t for t in everyday}
    if len(models) >= 2 and keys:
        X = np.array([[T[k]["per_model"][m]["d"] for k in keys] for m in models])
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        Z = linkage(squareform(D, checks=False), "average")
        fam = {"distance": {a: {b: float(D[i, j]) for j, b in enumerate(models)} for i, a in enumerate(models)},
               "merges": [[int(a), int(b), float(h)] for a, b, h, _ in Z],
               "order": [models[i] for i in dendrogram(Z, no_plot=True)["leaves"]]}
        if vendors:
            pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
            same = [D[i, j] for i, j in pairs if vendors.get(models[i]) == vendors.get(models[j])]
            diff = [D[i, j] for i, j in pairs if vendors.get(models[i]) != vendors.get(models[j])]
            fam["vendors"] = {m: vendors.get(m) for m in models}
            fam["same_vendor"] = float(np.mean(same)) if same else None
            fam["different_vendors"] = float(np.mean(diff)) if diff else None
        out["family"] = fam

    cards = {}
    for m in models:
        picks = []
        for k in keys:
            dm = T[k]["per_model"][m]["d"]
            hit, n = same_side[m].get(k, (0, 0))
            if abs(dm) >= 0.2 and n and hit / n >= 0.8:
                picks.append({"key": k, "text": T[k]["high"] if dm > 0 else T[k]["low"], "d": float(dm),
                              "categories": [hit, n]})
        cards[m] = sorted(picks, key=lambda p: -abs(p["d"]))[:4]
    out["cards"] = cards
    return out


def style_map(df, models, max_points=3000):
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    X = numeric_matrix(df)
    Z = StandardScaler().fit_transform(X.to_numpy())
    pca = PCA(n_components=2, random_state=0).fit(Z)
    P = pca.transform(Z)
    idx = np.arange(len(df))
    if len(idx) > max_points:
        idx = RNG.choice(idx, size=max_points, replace=False)
    mi = df.model.map({m: i for i, m in enumerate(models)}).to_numpy()
    points = [[float(P[i, 0]), float(P[i, 1]), int(mi[i]), df.prompt_id.iloc[i]] for i in idx]
    cents = {m: [float(P[mi == j, 0].mean()), float(P[mi == j, 1].mean())] for j, m in enumerate(models)}
    axes = {}
    for k in range(2):
        load = pca.components_[k]
        order = np.argsort(load)
        axes[f"pc{k + 1}"] = {"pos": [READABLE[X.columns[i]] for i in order[::-1][:3]],
                              "neg": [READABLE[X.columns[i]] for i in order[:3]]}
    return {"points": points, "centroids": cents, "variance": [float(v) for v in pca.explained_variance_ratio_],
            "axes": axes}


# ----------------------------------------------------------------------------- codes
def load_codes(path):
    if not path.exists():
        return None
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("codes"):
                rows.append(r)
    return rows or None


def coded_value(labels, code, positives, tagged):
    """Share of coders whose label for `code` is one of `positives`; NaN when none gave a usable label.

    On a trap prompt, "not_applicable" means the coder saw no trap in the reply to react to. That is
    a missing label, not a miss, so it is left out instead of counting against the model.
    """
    hits = [float(v.get(code) in positives) for v in labels
            if v.get(code) is not None and not (tagged and v.get(code) == "not_applicable")]
    return float(np.mean(hits)) if hits else float("nan")


def prompt_outcomes(df, model, col):
    """Prompt ids where every coder agreed the behavior happened, agreed it did not, or split."""
    v = df[df.model == model].groupby("prompt_id")[col].mean().dropna()
    return {"yes": sorted(v[v >= 1].index), "no": sorted(v[v <= 0].index), "split": sorted(v[(v > 0) & (v < 1)].index)}


def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 20:
        return None
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def read_answers(path: Path) -> tuple[dict, Counter, Counter]:
    """{(model, prompt, sample): last successful record}, the answers per model that never succeeded, and the failed
    calls per model that a later retry recovered."""
    rows, failed = {}, []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["model"], r["prompt_id"], r.get("sample", 0))
            if r.get("error") or not r.get("text"):
                failed.append(key)
            else:
                rows[key] = r  # last successful record wins
    errors = Counter(m for m, *_ in {k for k in failed if k not in rows})
    retried = Counter(k[0] for k in failed if k in rows)
    return rows, errors, retried


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    add_run_args(ap)
    ap.add_argument("--run-dir", default=None, help="override data/spectra/<revision>")
    ap.add_argument("--demo", action="store_true", help="mark output as demo data")
    ap.add_argument("--previous", default=None,
                    help="an earlier revision of the same roster: the accent section adds run-to-run agreement")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        cfg["revision"] = args.revision or cfg.get("revision") or run_dir.name
    else:
        run_dir = revision_dir(cfg, args.revision)
    revision = cfg["revision"]
    pdata = json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))
    prompts = {p["id"]: p for p in pdata["prompts"]}
    categories = list(dict.fromkeys(p["category"] for p in pdata["prompts"]))

    rows, errors, retried = read_answers(run_dir / "responses.jsonl")
    cfg_models = [m["name"] for m in cfg["models"]]
    seen = list(dict.fromkeys(k[0] for k in rows))
    models = [m for m in cfg_models if m in seen] + [m for m in seen if m not in cfg_models]

    recs = []
    for (m, pid, s), r in rows.items():
        p = prompts.get(pid)
        if not p:
            continue
        feats, prose = features(r["text"], p["category"])
        feats.update(model=m, prompt_id=pid, sample=s, category=p["category"], tags=",".join(p["tags"]),
                     constraint_type=(p["constraint"] or {}).get("type"),
                     constraint_ok=check_constraint(r["text"], p["constraint"]),
                     latency_s=r.get("latency_s"), output_tokens=r.get("output_tokens"),
                     reasoning_tokens=r.get("reasoning_tokens"), cost_usd=r.get("cost_usd"), prose=prose)
        recs.append(feats)
    df = pd.DataFrame(recs).sort_values(["model", "prompt_id", "sample"]).reset_index(drop=True)
    df.drop(columns=["prose"]).to_csv(run_dir / "features.csv", index=False)
    print(f"{len(df)} responses from {len(models)} models")

    # ---- style traits
    traits = []
    for key, col, label, desc, unit, high, low in STYLE_TRAITS:
        per_model, by_cat, panel = trait_block(df, models, col, categories)
        if key == "length":  # report geometric-mean words instead of mean log10
            for m in models:
                for fld in ("value", "lo", "hi"):
                    v = per_model[m][fld]
                    per_model[m][fld] = (10 ** v - 1) if v is not None and not math.isnan(v) else v
            by_cat = {m: {c: (10 ** v - 1 if v is not None else None) for c, v in cats.items()}
                      for m, cats in by_cat.items()}
            panel = 10 ** panel - 1
        traits.append({"key": key, "label": label, "short": SHORT_LABELS.get(key, label), "desc": desc, "unit": unit, "group": "style",
                       "high": high, "low": low, "panel": panel, "per_model": per_model, "by_category": by_cat})

    # ---- coded traits
    codes = load_codes(run_dir / "codes.jsonl")
    agreement, coders = None, []
    if codes:
        coders = sorted({c["coder"] for c in codes})
        by_key = defaultdict(dict)
        for c in codes:
            by_key[(c["model"], c["prompt_id"], c["sample"])][c["coder"]] = c["codes"]
        for key, label, desc, tag, code, positives, high, low in CODED_TRAITS:
            col = f"code_{key}"
            vals, not_applicable = [], 0
            for _, r in df.iterrows():
                if tag and tag not in r.tags.split(","):
                    vals.append(float("nan"))
                    continue
                cs = list(by_key.get((r.model, r.prompt_id, r["sample"]), {}).values())
                not_applicable += sum(bool(tag) and v.get(code) == "not_applicable" for v in cs)
                vals.append(coded_value(cs, code, positives, tagged=bool(tag)))
            df[col] = vals
            if df[col].notna().sum() == 0:
                continue
            per_model, by_cat, panel = trait_block(df, models, col, categories)
            trait = {"key": key, "label": label, "short": SHORT_LABELS.get(key, label), "desc": desc, "unit": "pct",
                     "group": "behavior", "tag": tag, "high": high, "low": low, "panel": panel, "per_model": per_model,
                     "by_category": by_cat}
            if tag:  # trap traits rest on a handful of prompts, so keep which ones went which way
                trait["not_applicable_labels"] = not_applicable
                for m in models:
                    per_model[m].update(prompt_outcomes(df, m, col))
            traits.append(trait)
        if len(coders) >= 2:
            agreement = {}
            for code in CODEBOOK_KEYS:
                a, b = [], []
                for d in by_key.values():
                    if coders[0] in d and coders[1] in d:
                        a.append(d[coders[0]].get(code))
                        b.append(d[coders[1]].get(code))
                agreement[code] = cohen_kappa(a, b)

    # ---- profile cards
    cards = {}
    dvec = {m: np.array([t["per_model"][m]["d"] for t in traits]) for m in models}
    for m in models:
        ranked = sorted(traits, key=lambda t: -abs(t["per_model"][m]["d"]))
        desc = [{"key": t["key"], "text": t["high"] if t["per_model"][m]["d"] > 0 else t["low"],
                 "d": t["per_model"][m]["d"]} for t in ranked if abs(t["per_model"][m]["d"]) >= 0.2][:4]
        others = [o for o in models if o != m]
        neighbor = min(others, key=lambda o: float(np.linalg.norm(dvec[m] - dvec[o]))) if others else None
        cards[m] = {"descriptors": desc, "neighbor": neighbor}

    # ---- constraints
    con = df[df.constraint_type.notna()]
    constraint_types = [t for t in CONSTRAINT_LABELS if t in set(con.constraint_type)]
    constraints = {"types": [{"key": t, "label": CONSTRAINT_LABELS[t],
                              "n_prompts": int(con[con.constraint_type == t].prompt_id.nunique())}
                             for t in constraint_types],
                   "per_model": {m: {t: (float(con[(con.model == m) & (con.constraint_type == t)].constraint_ok.mean())
                                         if len(con[(con.model == m) & (con.constraint_type == t)]) else None)
                                     for t in constraint_types} for m in models}}

    # ---- run stats
    runstats = {}
    for m in models:
        sub = df[df.model == m]
        rt = sub.reasoning_tokens.dropna()
        cost = sub.cost_usd.dropna()
        cv = None
        if sub["sample"].nunique() > 1:
            g = sub.groupby("prompt_id").words
            cvs = (g.std(ddof=0) / g.mean().replace(0, np.nan)).dropna()
            cv = float(cvs.median()) if len(cvs) else None
        runstats[m] = {"responses": int(len(sub)), "errors": int(errors.get(m, 0)), "retried": int(retried.get(m, 0)),
                       "median_latency_s": float(sub.latency_s.median()) if sub.latency_s.notna().any() else None,
                       "median_output_tokens": float(sub.output_tokens.median()) if sub.output_tokens.notna().any() else None,
                       "median_reasoning_tokens": float(rt.median()) if len(rt) else None,
                       "total_cost_usd": float(cost.sum()) if len(cost) else None,
                       "median_words": float(sub.words.median()), "length_cv_across_samples": cv}

    # ---- viewer
    first = df.sort_values("sample").drop_duplicates(["model", "prompt_id"])
    resp_map = defaultdict(dict)
    for m, pid, s in zip(first.model, first.prompt_id, first["sample"]):
        txt = rows[(m, pid, int(s))]["text"]
        resp_map[pid][m] = {"text": txt[:5000] + ("\n\n[truncated]" if len(txt) > 5000 else ""),
                            "words": len(WORD_RE.findall(txt))}
    viewer = {"prompts": [{"id": p["id"], "category": p["category"], "tags": p["tags"],
                           "constraint": p["constraint"], "prompt": p["prompt"]} for p in pdata["prompts"]
                          if p["id"] in resp_map],
              "responses": resp_map}

    print("signature terms...")
    sig = signature_terms(df, models)
    print("distinctiveness...")
    dist = distinctiveness(df, models)
    print("style map...")
    smap = style_map(df, models)
    print("accent...")
    previous = None
    if args.previous:
        previous = json.loads((DATA_ROOT / args.previous / "analysis.json").read_text(encoding="utf-8"))
    rc_path = run_dir / "run_config.json"
    vendors = ({m["name"]: m.get("provider") for m in json.loads(rc_path.read_text(encoding="utf-8")).get("models", [])}
               if rc_path.exists() else None)
    acc = accent(df, models, traits, previous, vendors)

    analysis = {
        "meta": {"revision": revision, "generated": datetime.now(timezone.utc).isoformat(),
                 # PeerRank phase files of the same revision, so the report can say it pairs with one
                 "peerrank_files": sorted(p.name for p in DATA_ROOT.parent.glob(f"phase*_{revision}.*")),
                 "demo": bool(args.demo), "models": models, "categories": categories,
                 "n_prompts": int(df.prompt_id.nunique()), "n_responses": int(len(df)),
                 "samples": int(df["sample"].nunique()), "coders": coders, "system_prompt": cfg.get("system_prompt"),
                 "tags": pdata["meta"].get("tags", {})},
        "traits": traits, "cards": cards, "constraints": constraints, "signature": sig,
        "distinctiveness": dist, "style_map": smap, "accent": acc, "runstats": runstats, "agreement": agreement,
        "viewer": viewer,
    }
    (run_dir / "analysis.json").write_text(json.dumps(clean(analysis), ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {run_dir / 'analysis.json'}")


CODEBOOK_KEYS = ["answer_first", "clarification", "premise", "error_handling", "user_opinion", "emotional_ack",
                 "unsolicited_advice", "moralizing", "tone", "initiative"]

if __name__ == "__main__":
    main()
