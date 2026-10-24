"""Spectra profiles for the guessing game: the research brief plus optional parts, all built from one source.

The source is a run directory holding analysis.json and responses.jsonl: a held-out half of this run's prompts,
or another revision. Nothing in a profile comes from the answers being guessed or from any guessing result.

Parts, added to the plain brief in this order:
  portrait  one paragraph per model, placed before the trait table
  pairs     how to tell apart the pairs that an author classifier trained on the source confuses most
  examples  typical opening and closing lines per model, quoted from source answers in different categories
  category  each model's habits in the category of the prompt being guessed

"base" is the plain brief, byte for byte the one the game has always used, so earlier games stay comparable.
"""
import json
import re
from itertools import combinations
from pathlib import Path

PARTS = ("portrait", "pairs", "examples", "category")
CATEGORY_TRAITS = ("length", "structure", "emdash", "hedging", "answer_first", "warm_open", "hooks")
QUOTE_WORDS = 30


def parse_profile(spec: str | None) -> str:
    """'base', 'all', or a comma list of parts -> canonical name such as 'pairs+examples'."""
    if not spec or spec == "base":
        return "base"
    parts = set(PARTS) if spec == "all" else {s.strip() for s in re.split(r"[,+]", spec) if s.strip()}
    unknown = sorted(parts - set(PARTS))
    if unknown:
        raise ValueError(f"unknown profile part(s) {', '.join(unknown)}; choose from {', '.join(PARTS)}, all or base")
    return "+".join(p for p in PARTS if p in parts)


def parts_of(profile: str) -> list[str]:
    return [] if profile == "base" else profile.split("+")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


class Profile:
    """The brief for one source. text(category) is what a guesser sees for a prompt in that category."""

    def __init__(self, source: Path, models: list[str], profile: str, base_brief: str):
        self.source, self.models, self.profile = Path(source), models, profile
        self.parts = parts_of(profile)
        self.base = base_brief
        if not self.parts:
            return
        analysis = json.loads((self.source / "analysis.json").read_text(encoding="utf-8"))
        answers = {}
        for r in _load_jsonl(self.source / "responses.jsonl"):
            if r.get("text") and r.get("model") in models:
                answers[(r["model"], r["prompt_id"], r.get("sample", 0))] = r  # last successful record wins
        answers = list(answers.values())
        self.portrait = portraits(analysis, models) if "portrait" in self.parts else ""
        self.pairs = pair_lines(analysis, answers, models) if "pairs" in self.parts else ""
        self.examples = example_lines(analysis, answers, models) if "examples" in self.parts else ""
        self.categories = {}
        if "category" in self.parts:
            cats = sorted({r.get("category") for r in answers if r.get("category")})
            self.categories = {c: category_section(analysis, models, c) for c in cats}

    def text(self, category: str | None = None) -> str:
        if not self.parts:
            return self.base
        blocks = [self.portrait, self.base, self.pairs, self.examples, self.categories.get(category, "")]
        return "\n\n".join(b for b in blocks if b)


# ----------------------------------------------------------------------------- parts
def _traits(analysis: dict) -> dict:
    return {t["key"]: t for t in analysis.get("traits") or []}


def _value(trait: dict, model: str):
    v = ((trait.get("per_model") or {}).get(model) or {}).get("value")
    return None if v is None or v != v else v


def _everyday(trait: dict) -> bool:
    """A trait measured on every answer. Trap traits and limit adherence rest on a few prompts each and say
    nothing about an ordinary answer, so they stay out of portraits and pair lines."""
    return not trait.get("tag") and trait.get("key") != "adherence"


def portraits(analysis: dict, models: list[str]) -> str:
    """One short paragraph per model: its strongest everyday departures, length, wording and nearest neighbor."""
    T, cards, signature = _traits(analysis), analysis.get("cards") or {}, analysis.get("signature") or {}
    lines = ["Portraits, one per model, from the same measurements as the notes below:"]
    for m in models:
        card = cards.get(m) or {}
        departures = sorted(((abs(d), t["high"] if d > 0 else t["low"]) for t in T.values()
                             if _everyday(t) and t.get("high") and t.get("low")
                             and isinstance(d := ((t.get("per_model") or {}).get(m) or {}).get("d"), (int, float))),
                            key=lambda x: -x[0])
        bits = [p if i == 0 else p[0].lower() + p[1:] for i, (_, p) in enumerate(departures[:3])]
        if "length" in T and _value(T["length"], m) is not None:
            bits.append(f"typically about {_value(T['length'], m):.0f} words")
        if "answer_first" in T and _value(T["answer_first"], m) is not None:
            bits.append(f"answers within its first two sentences {100 * _value(T['answer_first'], m):.0f}% of the time")
        words = [s["term"] for s in (signature.get(m) or [])[:3]]
        if words:
            bits.append("often says " + ", ".join(f"'{w}'" for w in words))
        if card.get("neighbor"):
            bits.append(f"writes most like {card['neighbor']}")
        lines.append(f"- {m}: " + "; ".join(bits) + ".")
    return "\n".join(lines)


def pair_lines(analysis: dict, answers: list[dict], models: list[str], top: int = 3) -> str:
    """For the pairs the source's author classifier confused most: the traits and words that separate them."""
    from spectra.guess import _fmt_value  # the brief's own number format

    dist = analysis.get("distinctiveness") or {}
    labels, matrix = dist.get("labels") or [], dist.get("matrix") or []
    if not matrix:
        return ""
    idx = {m: i for i, m in enumerate(labels)}
    pairs = sorted(((a, b, matrix[idx[a]][idx[b]] + matrix[idx[b]][idx[a]])
                    for a, b in combinations([m for m in models if m in idx], 2)), key=lambda p: -p[2])[:top]
    T = _traits(analysis)
    words = _pair_words(answers, pairs)
    lines = ["How to tell apart the models most often mixed up with each other (by a classifier trained on the same "
             "source answers). The traits that differ most between each pair, then words each one uses more than the other:"]
    for a, b, _ in pairs:
        gaps = []
        for key, t in T.items():
            if not _everyday(t):
                continue
            da = ((t.get("per_model") or {}).get(a) or {}).get("d")
            db = ((t.get("per_model") or {}).get(b) or {}).get("d")
            va, vb = _value(t, a), _value(t, b)
            if None not in (da, db, va, vb):
                gaps.append((abs(da - db), t, va, vb))
        gaps.sort(key=lambda g: -g[0])
        traits = "; ".join(f"{t.get('label') or t['key']}: {a} {_fmt_value(t.get('unit'), va)}, "
                           f"{b} {_fmt_value(t.get('unit'), vb)}" for _, t, va, vb in gaps[:3])
        wa, wb = words.get((a, b), ([], []))
        said = []
        if wa:
            said.append(f"{a} says " + ", ".join(f"'{w}'" for w in wa) + " more")
        if wb:
            said.append(f"{b} says " + ", ".join(f"'{w}'" for w in wb) + " more")
        lines.append(f"- {a} vs {b}: {traits}." + (" " + "; ".join(said) + "." if said else ""))
    return "\n".join(lines)


def _pair_words(answers: list[dict], pairs: list[tuple], top: int = 4) -> dict:
    """{(a, b): (words a uses more than b, words b uses more than a)}, log-odds between the two models only."""
    import pandas as pd

    from spectra.analyze import signature_terms, split_code

    out = {}
    for a, b, _ in pairs:
        rows = [{"model": r["model"], "prose": split_code(r["text"])[0]} for r in answers if r["model"] in (a, b)]
        if not rows:
            continue
        terms = signature_terms(pd.DataFrame(rows), [a, b], top=top, min_count=5)
        out[(a, b)] = ([t["term"] for t in terms.get(a, [])], [t["term"] for t in terms.get(b, [])])
    return out


def _clean(line: str) -> str:
    line = re.sub(r"^\s*(?:#+|>+|[-*•]+|\d+[.)])\s*", "", line)
    return re.sub(r"[*_`]+", "", line).strip()


def _quote(text: str, last: bool) -> str:
    lines = [_clean(x) for x in text.splitlines() if not x.lstrip().startswith("|")]  # table rows are not a voice
    lines = [x for x in lines if re.search(r"[A-Za-z]", x)]
    if not lines:
        return ""
    line = lines[-1] if last else lines[0]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
    sentence = (sentences[-1] if last else sentences[0]) if sentences else line
    words = sentence.split()
    return " ".join(words[:QUOTE_WORDS]) + (" …" if len(words) > QUOTE_WORDS else "")


def example_lines(analysis: dict, answers: list[dict], models: list[str], k: int = 3) -> str:
    """Per model, the opening and closing lines of k source answers from different categories.

    Answers are ranked by how many of the model's signature words (from the same source) their opening and
    closing contain, so the quotes show its usual voice rather than a random one; ties go to the lower prompt id.
    """
    from spectra.analyze import split_code, tokens_for_signature

    signature = analysis.get("signature") or {}
    lines = [f"Typical openings and closings, quoted from source answers ({k} per model, each from a different "
             "category of prompt). Topics differ from the prompt below; look at the voice, not the subject:"]
    for m in models:
        z = {s["term"]: s["z"] for s in signature.get(m) or []}
        ranked = []
        for r in answers:
            if r["model"] != m:
                continue
            prose = split_code(r["text"])[0]
            opening, closing = _quote(prose, last=False), _quote(prose, last=True)
            if not opening:
                continue
            found = set(tokens_for_signature(opening + "\n" + closing))
            ranked.append((-sum(v for t, v in z.items() if t in found), r["prompt_id"], r.get("category"), opening, closing))
        ranked.sort(key=lambda x: (x[0], x[1]))
        picked, seen = [], set()
        for _, _, cat, opening, closing in ranked:
            if cat in seen:
                continue
            seen.add(cat)
            picked.append((opening, closing))
            if len(picked) == k:
                break
        lines.append(f"- {m}:")
        for opening, closing in picked:
            tail = f' … ends: "{closing}"' if closing and closing != opening else ""
            lines.append(f'    starts: "{opening}"{tail}')
    return "\n".join(lines)


def category_section(analysis: dict, models: list[str], category: str) -> str:
    """Each model's habits in one category of prompt, from the same source."""
    from spectra.guess import _fmt_value

    T = _traits(analysis)
    lines = [f"In {category} prompts specifically, the kind of prompt below, the same measurements give:"]
    for key in CATEGORY_TRAITS:
        t = T.get(key)
        if not t:
            continue
        cells = {}
        for m in models:
            v = ((t.get("by_category") or {}).get(m) or {}).get(category)
            if v is not None and v == v:
                cells[m] = _fmt_value(t.get("unit"), v)
        if len(set(cells.values())) > 1:  # a row where every model reads the same tells the guesser nothing
            lines.append(f"- {t.get('label') or key}: " + "; ".join(f"{m} {c}" for m, c in cells.items()) + ".")
    return "\n".join(lines) if len(lines) > 1 else ""
