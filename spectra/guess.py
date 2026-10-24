#!/usr/bin/env python3
"""The guessing game: each model sees a prompt's answers with the authors stripped off,
and matches every answer to a model on the roster.

Three modes. ``blind`` (the default) gives the prompt, the answers, and the roster names.
``briefed`` adds a research brief: the measured habits from a Spectra analysis.json
(length, formatting, hedging, signature words, and the rest). ``heldout`` is briefed with
notes measured on the other half of this revision's prompts: the prompts split in two,
stratified by category, and each half is guessed with the other half's notes, so no prompt
or answer in front of the guesser went into its notes. The brief never labels the answers,
and the guesser is never told which model it is.

By default every model must be used once (there is one answer per model), so placing one
answer narrows the rest. ``--independent`` asks for a name per answer, judged on its own,
with repeats allowed. Guessers may still choose to use each model once, so the summary reports
how often a call actually repeated a name.

One call per guesser per prompt. Letters are shuffled per guesser, so position is not
a shared cue.

  python -m spectra.guess --revision oct-14-26 --limit 5              # blind smoke test
  python -m spectra.guess --revision oct-14-26                        # blind game
  python -m spectra.guess --revision oct-23-26 --mode briefed --research oct-14-26
  python -m spectra.guess --revision oct-23-26 --mode heldout --independent --dry-run

--dry-run builds every message (and, for heldout, the two half analyses) and estimates the
cost, without calling a model. Writes data/spectra/<revision>/guesses.jsonl. Records carry
mode, research and assignment style, so every game can share a file; the report shows each
one. Headline scores exclude a model's own answer. Chance is 1/k.
"""
import argparse
import asyncio
import hashlib
import json
import math
import random
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from spectra import DATA_ROOT, add_run_args, load_config, providers, run_dir as revision_dir, writer_lock
from spectra.run import select_prompts

ANSWER_CHAR_CAP = 20000
MODES = ("blind", "briefed", "heldout")
ASSIGN = ("permutation", "independent")


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def resolve_name(raw, models: list[str]) -> str | None:
    """Map a guess onto the roster. Ambiguous short names ('claude', 'gemini') do not match."""
    if raw is None:
        return None
    key = _norm(str(raw).strip().strip("\"'`[]"))
    if len(key) < 3:
        return None
    exact = [m for m in models if _norm(m) == key]
    if len(exact) == 1:
        return exact[0]
    hits = [m for m in models if key in _norm(m) or _norm(m) in key]
    return hits[0] if len(hits) == 1 else None


def parse_assignment(raw: str | None, letters: list[str], models: list[str]) -> dict | None:
    """Return {letter: roster name or None}. None if the reply has no JSON object."""
    from peerrank.config import extract_json

    obj = extract_json(raw or "")
    if isinstance(obj, str):  # the object came back as a JSON string: "{\"A\": \"...\"}"
        obj = extract_json(obj)
    if obj is None and raw and '\\"' in raw:  # or escaped inside prose
        obj = extract_json(raw.replace('\\"', '"'))
    if isinstance(obj, list):
        merged = {}
        for item in obj:
            if isinstance(item, dict):
                merged.update(item)
        obj = merged or None
    if not isinstance(obj, dict):
        return None

    by_letter = {}
    for key, value in obj.items():
        token = str(key).strip().strip("\"'").upper()
        for prefix in ("RESPONSE ", "ANSWER ", "LABEL "):
            if token.startswith(prefix):
                token = token[len(prefix):]
        token = token.strip(" .:)")
        if token in letters:
            by_letter[token] = value

    # Some models swap the mapping and put the model name in the key.
    if len(by_letter) < len(letters) // 2:
        inverted = {}
        for key, value in obj.items():
            token = str(value).strip().upper()
            for prefix in ("RESPONSE ", "ANSWER ", "LABEL "):
                if token.startswith(prefix):
                    token = token[len(prefix):]
            token = token.strip(" .:)")
            if token in letters and token not in inverted:
                inverted[token] = key
        if len(inverted) > len(by_letter):
            by_letter = inverted

    return {letter: resolve_name(by_letter.get(letter), models) for letter in letters}


def shuffle_models(prompt_id: str, sample: int, guesser: str, models: list[str]) -> list[str]:
    seed = int(hashlib.sha256(f"{prompt_id}:{sample}:{guesser}".encode()).hexdigest()[:8], 16)
    order = list(models)
    random.Random(seed).shuffle(order)
    return order


def holdout_halves(prompts: list[dict]) -> dict[str, str]:
    """prompt id -> "A" or "B". Each category splits in half; the split is fixed by a hash of the id."""
    by_cat: dict[str, list[str]] = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p["id"])
    half = {}
    for ids in by_cat.values():
        for i, pid in enumerate(sorted(ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())):
            half[pid] = "AB"[i % 2]
    return half


def _fmt_value(unit: str | None, value) -> str:
    if value is None:
        return "n/a"
    if unit == "pct":
        return f"{100 * value:.0f}%"
    if unit == "words":
        return f"{value:.0f} words"
    if unit == "per100":
        return f"{value:.2f}/100 words"
    if unit == "score":
        return f"{value:.0f}"
    if unit == "ratio":
        return f"{value:.3f}"
    return f"{value:.2f}"


def _trait_cell(trait: dict, model: str) -> str:
    pm = (trait.get("per_model") or {}).get(model) or {}
    value, effect = pm.get("value"), pm.get("d")
    if value is None:
        return "not measured"
    text = _fmt_value(trait.get("unit"), value)
    if isinstance(effect, (int, float)):
        text += f" (d={effect:+.2f})"
    return text


def research_brief(analysis: dict, models: list[str]) -> str:
    """Aggregate habits from a Spectra analysis. No per-answer labels, cost, or latency."""
    meta = analysis.get("meta") or {}
    rev = meta.get("revision") or "unknown"
    head = f"Source: Spectra revision {rev}"
    if meta.get("n_prompts") and meta.get("n_responses"):
        head += f", {meta['n_prompts']} prompts and {meta['n_responses']} answers"
    lines = [
        head + ", no system prompt.",
        "These are averages for that run, not labels for the answers you are about to see. "
        "d is standard deviations from the panel average. Positive d means more of that trait.",
        "",
        "Measured habits, one trait per line.",
    ]
    for trait in analysis.get("traits") or []:
        cells = [_trait_cell(trait, m) for m in models]
        if all(c == "not measured" for c in cells):
            continue
        label = trait.get("label") or trait.get("key")
        sample_n = next((int(pm["n"]) for m in models
                         if (pm := (trait.get("per_model") or {}).get(m) or {}).get("n")), None)
        extra = ""
        if trait.get("tag"):
            extra = f" Only prompts tagged {trait['tag']}" + (f" (n={sample_n})." if sample_n else ".")
        elif trait.get("key") == "adherence":
            extra = " Only prompts that stated a length or format limit" + (f" (n={sample_n})." if sample_n else ".")
        lines.append(f"- {label}: " + "; ".join(f"{m} {cell}" for m, cell in zip(models, cells)) + "." + extra)

    constraints = analysis.get("constraints") or {}
    types = constraints.get("types") or []
    per_model = constraints.get("per_model") or {}
    if types:
        lines.append("")
        lines.append("Explicit limits met, by kind of limit:")
        for kind in types:
            cells = []
            for m in models:
                rate = (per_model.get(m) or {}).get(kind["key"])
                cells.append("n/a" if rate is None else f"{100 * rate:.0f}%")
            n = kind.get("n_prompts")
            suffix = f" (n={n})" if n else ""
            lines.append(f"- {kind.get('label') or kind['key']}{suffix}: " + "; ".join(
                f"{m} {cell}" for m, cell in zip(models, cells)) + ".")

    lines.append("")
    lines.append("Words each model uses more than the others:")
    signature = analysis.get("signature") or {}
    for m in models:
        terms = signature.get(m) or []
        shown = ", ".join(f"{item['term']} (z={item['z']:+.1f})" for item in terms[:6])
        lines.append(f"- {m}: {shown or 'none stood out'}")

    lines.append("")
    lines.append("Strongest departures from the panel:")
    cards = analysis.get("cards") or {}
    for m in models:
        card = cards.get(m) or {}
        desc = [f"{item['text']} (d={item['d']:+.2f})" for item in (card.get("descriptors") or [])[:4]]
        sentence = "; ".join(desc) if desc else "no large departure"
        neighbor = card.get("neighbor")
        if neighbor:
            sentence += f". Closest overall habits: {neighbor}"
        lines.append(f"- {m}: {sentence}.")
    return "\n".join(lines)


def load_research(revision: str, models: list[str]) -> tuple[str, str]:
    path = DATA_ROOT / revision / "analysis.json"
    if not path.exists():
        raise SystemExit(
            f"No research profile at {path}. Run python -m spectra.analyze --revision {revision} "
            "first, or pass --research <revision> that already has analysis.json."
        )
    analysis = json.loads(path.read_text(encoding="utf-8"))
    return revision, research_brief(analysis, models)


def heldout_briefs(config_path: str, run_dir: Path, revision: str, models: list[str],
                   halves: dict[str, str]) -> dict[str, str]:
    """{half: brief} where each half's brief is measured on the other half's prompts only.

    Writes the filtered answers and codes to <run_dir>/holdout/half-<X>/ and runs spectra.analyze there.
    """
    from spectra import analyze  # numpy/pandas only when this mode runs

    briefs = {}
    for half, other in (("A", "B"), ("B", "A")):
        keep = {pid for pid, h in halves.items() if h == other}
        out = run_dir / "holdout" / f"half-{other}"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("responses.jsonl", "codes.jsonl"):
            rows = [r for r in load_jsonl(run_dir / name) if r.get("prompt_id") in keep]
            (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"Measuring half {other} ({len(keep)} prompts) for the guesses on half {half}")
        analyze.main(["--config", config_path, "--run-dir", str(out), "--revision", f"{revision}-half-{other}"])
        briefs[half] = research_brief(json.loads((out / "analysis.json").read_text(encoding="utf-8")), models)
    return briefs


def build_message(user_prompt: str, order: list[str], texts: dict[str, str], roster: list[str],
                  brief: str | None = None, independent: bool = False) -> str:
    letters = string.ascii_uppercase[: len(order)]
    blocks = []
    for letter, model in zip(letters, order):
        body = (texts[model] or "").strip()[:ANSWER_CHAR_CAP]
        blocks.append(f"===== RESPONSE {letter} =====\n{body}\n===== END RESPONSE {letter} =====")
    names = "\n".join(f"- {m}" for m in roster)
    letter_list = ", ".join(letters)
    example = ", ".join(f'"{L}": "<model>"' for L in letters)
    research = ""
    if brief:
        research = (
            "Research notes from a measurement of these same models are included below. "
            "The notes describe how each model usually writes across many prompts. "
            "They are not labels for these answers, and they do not say which letter is which. "
            "Use the notes together with the writing in front of you.\n\n"
            f"<research>\n{brief.strip()}\n</research>\n\n"
        )
        how = (
            "The answers are labeled with letters. Those letters were shuffled and carry no information "
            "about the author. The answers are not signed. One of them may be your own, but you are not "
            "told which, and you are not told which model you are.\n\n"
        )
    else:
        how = (
            "The answers are labeled with letters. Those letters were shuffled and carry no information "
            "about the author. The answers are not signed. One of them may be your own, but you are not "
            "told which, and you are not told which model you are. Judge only by how the answer is written.\n\n"
        )
    if independent:
        roster_line = "Each answer below was written by one of the models on this roster:\n"
        rule = ("Name the most likely model for each letter. Judge each answer on its own: "
                "a model may be named more than once, and you do not have to use every model.\n\n")
    else:
        roster_line = "Each answer below was written by a different model from this roster:\n"
        rule = "Assign every letter to exactly one model. Use each model once.\n\n"
    return (
        "You are matching anonymous answers to the models that wrote them. "
        "You are not grading quality.\n\n"
        + roster_line
        + f"{names}\n\n"
        + research
        + how
        + rule
        + f"<user_prompt>\n{user_prompt}\n</user_prompt>\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn only this JSON object, using the roster names exactly, "
        f"with one entry for each of {letter_list}:\n"
        "{" + example + "}"
    )


def panel_answers(responses: list[dict], models: list[str]) -> dict[tuple, dict[str, str]]:
    """(prompt_id, sample) -> {model: text} for rounds where every roster model answered."""
    grouped: dict[tuple, dict[str, str]] = {}
    meta: dict[tuple, str] = {}
    for r in responses:
        if not r.get("text") or r.get("error") or r["model"] not in models:
            continue
        key = (r["prompt_id"], int(r.get("sample") or 0))
        grouped.setdefault(key, {})[r["model"]] = r["text"]
        if r.get("category"):
            meta[key] = r["category"]
    full = {}
    for key, texts in grouped.items():
        if all(m in texts for m in models):
            full[key] = {"texts": texts, "category": meta.get(key)}
    return full


def game_key(rec: dict) -> tuple[str, str | None, str]:
    """(mode, research, assignment). Records written before a field existed count as blind or permutation."""
    mode = rec.get("mode") if rec.get("mode") in MODES else "blind"
    research = rec.get("research") if mode != "blind" else None
    assign = rec.get("assign") if rec.get("assign") in ASSIGN else "permutation"
    return mode, research, assign


def game_title(key: tuple) -> str:
    mode, research, assign = key
    title = {"blind": "Just guess",
             "briefed": f"Guess from the research ({research} notes)",
             "heldout": "Guess from held-out research (notes from the other half of the prompts)"}[mode]
    return title + (", each answer judged on its own" if assign == "independent" else "")


def _bootstrap(rounds: list[dict], reps: int = 1000, seed: int = 7) -> dict:
    """95% intervals of pooled rates, resampling whole rounds (one prompt's answers are judged together).

    rounds: [{name: [correct, n]}]. Returns {name: [lo, hi]}.
    """
    if not rounds:
        return {}
    rng = random.Random(seed)
    names = sorted({k for r in rounds for k in r})
    draws = {k: [] for k in names}
    for _ in range(reps):
        tot = {k: [0, 0] for k in names}
        for _ in range(len(rounds)):
            for k, (c, n) in rounds[rng.randrange(len(rounds))].items():
                tot[k][0] += c
                tot[k][1] += n
        for k, (c, n) in tot.items():
            if n:
                draws[k].append(c / n)
    out = {}
    for k, d in draws.items():
        d.sort()
        out[k] = [d[int(0.025 * (len(d) - 1))], d[int(0.975 * (len(d) - 1))]] if d else None
    return out


def _skill_weighted_vote(ballots: list[tuple], models: list[str], halves: dict[str, str]) -> dict | None:
    """{(prompt, sample): [answers placed correctly, answers]} for votes weighted by skill, each model named once per prompt.

    A guesser's vote on one half of the prompts weighs log((k-1)p/(1-p)), where p is its accuracy on the other half,
    so no answer helps set the weight of the vote cast on it and a guesser at chance weighs nothing (below chance
    counts as nothing too). Each prompt's answers then go to models one-to-one, maximizing the summed weight.
    None when the prompts do not split in two or scipy is missing.
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None
    if len({halves.get(r[0][0]) for r in ballots} - {None}) < 2:
        return None
    k, idx = len(models), {m: i for i, m in enumerate(models)}
    hits: dict[tuple, list] = {}
    for rnd, author, guesser, named in ballots:
        cell = hits.setdefault((guesser, halves[rnd[0]]), [0, 0])
        cell[0] += named == author
        cell[1] += 1

    def weight(guesser, half):
        c, n = hits.get((guesser, "B" if half == "A" else "A"), [0, 0])
        p = (c + 0.5) / (n + 1.0)
        return max(0.0, math.log((k - 1) * p / (1 - p)))

    scores, authors = {}, {}
    for rnd, author, guesser, named in ballots:
        authors.setdefault(rnd, set()).add(idx[author])
        if named in idx:
            scores.setdefault(rnd, np.zeros((k, k)))[idx[author], idx[named]] += weight(guesser, halves[rnd[0]])
    rng = np.random.default_rng(7)  # breaks exact ties at random, reproducibly
    out = {}
    for rnd, present in authors.items():
        board = scores.get(rnd, np.zeros((k, k)))
        rows, cols = linear_sum_assignment(-(board + 1e-9 * rng.random(board.shape)))
        placed = dict(zip(rows.tolist(), cols.tolist()))
        out[rnd] = [sum(placed.get(a) == a for a in present), len(present)]
    return out


def summarize_guesses(records: list[dict], models: list[str], key: tuple = ("blind", None, "permutation"),
                      answer_revision: str | None = None) -> dict | None:
    """Recognizability and guessing skill for one game. Own answers are excluded from both headlines.

    Intervals resample prompts. Under a permutation, placing one answer narrows the rest, so the
    summary also reports how guessers did on the others when they did and did not find their own answer.
    """
    latest = {}
    for r in records:
        if isinstance(r.get("guesses"), dict) and isinstance(r.get("labels"), dict) and game_key(r) == key:
            latest[(r["guesser"], r["prompt_id"], r.get("sample", 0))] = r  # a retried call replaces the earlier one
    usable = list(latest.values())
    if not usable or len(models) < 2:
        return None
    mode, research, assign = key
    k = len(models)
    idx = {m: i for i, m in enumerate(models)}
    other_correct = {m: 0 for m in models}
    other_n = {m: 0 for m in models}
    self_correct = {m: 0 for m in models}
    self_n = {m: 0 for m in models}
    recognized = {m: 0 for m in models}
    recognized_n = {m: 0 for m in models}
    matrix = [[0 for _ in models] for _ in models]
    unresolved = {m: 0 for m in models}
    rounds: dict[tuple, dict] = {}
    own = {True: [0, 0], False: [0, 0]}  # found own answer -> [others correct, others judged]
    repeats = 0
    votes: dict[tuple, list] = {}  # (prompt, sample, author) -> what each other guesser named
    ballots = []  # ((prompt, sample), author, guesser, named) for the skill-weighted vote
    by_cat: dict[str, list] = {}
    prompt_cat, prompt_half = {}, {}

    for rec in usable:
        guesser = rec["guesser"]
        if guesser not in idx:
            continue
        named = [g for g in rec["guesses"].values() if g]
        repeats += len(named) != len(set(named))
        own_letter = next((L for L, a in rec["labels"].items() if a == guesser), None)
        found = own_letter is not None and rec["guesses"].get(own_letter) == guesser
        rnd = (rec["prompt_id"], rec.get("sample", 0))
        prompt_cat[rec["prompt_id"]] = rec.get("category") or "all"
        if rec.get("half") in ("A", "B"):
            prompt_half[rec["prompt_id"]] = rec["half"]
        tally = rounds.setdefault(rnd, {})
        for letter, author in rec["labels"].items():
            if author not in idx:
                continue
            guess = rec["guesses"].get(letter)
            correct = guess == author
            if guesser == author:
                self_n[guesser] += 1
                self_correct[guesser] += int(correct)
                continue
            other_n[guesser] += 1
            other_correct[guesser] += int(correct)
            votes.setdefault((rec["prompt_id"], rec.get("sample", 0), author), []).append(guess)
            ballots.append((rnd, author, guesser, guess))
            cat_cell = by_cat.setdefault(prompt_cat[rec["prompt_id"]], [0, 0])
            cat_cell[0] += int(correct)
            cat_cell[1] += 1
            recognized_n[author] += 1
            recognized[author] += int(correct)
            own[found][0] += int(correct)
            own[found][1] += 1
            for name in ("all", "g:" + guesser, "a:" + author):
                cell = tally.setdefault(name, [0, 0])
                cell[0] += int(correct)
                cell[1] += 1
            if guess in idx:
                matrix[idx[author]][idx[guess]] += 1
            else:
                unresolved[author] += 1

    # Majority of the other guessers: each answer goes to the model most of them named. A tie splits
    # the credit evenly (the expected score of a random tie-break); an answer nobody resolved scores 0.
    # The author never votes on its own answer. This measures how identifiable a model is, not what one guesser knows.
    vote_credit = vote_ties = 0
    for (pid, sample, author), named in votes.items():
        counts = Counter(g for g in named if g)
        best = [m for m, n in counts.items() if n == max(counts.values())] if counts else []
        credit = (author in best) / len(best) if best else 0.0
        vote_credit += credit
        vote_ties += len(best) > 1
        cell = rounds.setdefault((pid, sample), {}).setdefault("vote", [0, 0])
        cell[0] += credit
        cell[1] += 1
    # The same votes weighted by each guesser's skill on the other half of the prompts, each model named once per prompt.
    halves = prompt_half if len(prompt_half) == len(prompt_cat) else holdout_halves(
        [{"id": pid, "category": c} for pid, c in prompt_cat.items()])
    weighted = _skill_weighted_vote(ballots, models, halves)
    for rnd, cell in (weighted or {}).items():
        rounds.setdefault(rnd, {})["wvote"] = cell

    def rate(n_ok, n):
        return (n_ok / n) if n else None

    ci = _bootstrap(list(rounds.values()))
    guessers = []
    recognizable = []
    for m in models:
        guessers.append({
            "model": m,
            "accuracy": rate(other_correct[m], other_n[m]),
            "ci": ci.get("g:" + m),
            "correct": other_correct[m],
            "n": other_n[m],
            "self_rate": rate(self_correct[m], self_n[m]),
            "self_n": self_n[m],
        })
        recognizable.append({
            "model": m,
            "recall": rate(recognized[m], recognized_n[m]),
            "ci": ci.get("a:" + m),
            "correct": recognized[m],
            "n": recognized_n[m],
            "unresolved": unresolved[m],
        })
    guessers.sort(key=lambda r: (-(r["accuracy"] or -1), r["model"]))
    recognizable.sort(key=lambda r: (-(r["recall"] or -1), r["model"]))

    row_rates = []
    for i, m in enumerate(models):
        denom = recognized_n[m] or 1
        row_rates.append([matrix[i][j] / denom for j in range(k)])

    n_other = sum(other_n.values())
    n_hit = sum(other_correct.values())
    misses = n_other - n_hit
    pairs = sorted(((models[i], models[j], matrix[i][j] + matrix[j][i]) for i in range(k) for j in range(i + 1, k)),
                   key=lambda p: -p[2])
    confused = [{"pair": [a, b], "misses": n, "share": n / misses} for a, b, n in pairs[:3] if n and misses]
    own_effect = {"found": rate(*own[True]), "n_found": own[True][1],
                  "not_found": rate(*own[False]), "n_not_found": own[False][1]}
    scoring = (
        "Accuracy is the share of other models' answers named correctly. "
        "Recall is the share of other guessers who named this author. A model's own answer is left out of both, "
        f"and is reported separately as self-recognition. A random guess is right {1.0 / k:.0%} of the time. "
        "Intervals are 95%, resampling prompts. The last row gives each answer to the model most of the other "
        "guessers named, ties split evenly: how identifiable the models are, not what one guesser knows."
    )
    if mode == "briefed":
        lead = (
            "Each guesser saw every answer labeled with shuffled letters and no author names, "
            f"plus research notes from {research or 'a Spectra profile'}. The notes do not label these answers, "
            "and the guesser is not told which model it is. "
        )
        if answer_revision and research == answer_revision:
            lead += "The notes were measured on this same set of answers, so each answer is one part of its model's average. "
        elif answer_revision and research:
            lead += "The notes come from a different run than these answers. "
    elif mode == "heldout":
        lead = (
            "Each guesser saw every answer labeled with shuffled letters and no author names, plus research notes "
            "measured on the other half of this run's prompts (split by category), so no prompt or answer in front "
            "of the guesser went into its notes. The guesser is not told which model it is. "
        )
    else:
        lead = (
            "Each guesser saw every answer to a prompt, labeled with shuffled letters and no author names, "
            "and was not told which model it is. "
        )
    repeat_rate = repeats / len(usable)
    if assign == "independent":
        # Allowing repeats does not make guessers use them: a call that repeats no name is still a permutation.
        lead += (f"Guessers could name a model more than once, and {repeat_rate:.0%} of calls did. The other calls "
                 "used each model at most once, so placing one answer could still narrow the rest. ")
    else:
        lead += "Every model had to be used once, so placing one answer narrows the rest. "
    if own_effect["found"] is not None and own_effect["not_found"] is not None:
        lead += (f"Guessers named {own_effect['found']:.0%} of the others correctly when they found their own answer "
                 f"and {own_effect['not_found']:.0%} when they did not. ")
    return {
        "mode": mode,
        "research": [research] if research else [],
        "assign": assign,
        "title": game_title(key),
        "k": k,
        "chance": 1.0 / k,
        "n_calls": len(usable),
        "n_judgments": n_other,
        "accuracy": rate(n_hit, n_other),
        "ci": ci.get("all"),
        "own_answer_effect": own_effect,
        "repeat_rate": repeat_rate,
        "vote": {"accuracy": rate(vote_credit, len(votes)), "ci": ci.get("vote"), "n": len(votes), "ties": vote_ties,
                 "voters": sorted(len(v) for v in votes.values())[len(votes) // 2] if votes else 0},
        "vote_weighted": None if not weighted else {
            "accuracy": rate(sum(c for c, _ in weighted.values()), sum(n for _, n in weighted.values())),
            "ci": ci.get("wvote"), "n": sum(n for _, n in weighted.values())},
        "confused_pairs": confused,
        "by_category": {c: rate(*v) for c, v in sorted(by_cat.items())},
        "labels": models,
        "guessers": guessers,
        "recognizable": recognizable,
        "matrix": row_rates,
        "note": lead + scoring,
    }


def summarize_all(records: list[dict], models: list[str], answer_revision: str | None = None) -> list[dict]:
    """One summary per game in the file: blind, briefed per research source, held-out, each by assignment style."""
    keys = {game_key(r) for r in records if isinstance(r.get("guesses"), dict)}
    order = sorted(keys, key=lambda k: (MODES.index(k[0]), k[1] or "", ASSIGN.index(k[2])))
    return [s for s in (summarize_guesses(records, models, key, answer_revision) for key in order) if s]


def prompt_effects(records: list[dict], models: list[str], key: tuple = ("blind", None, "permutation"),
                   reps: int = 500, seed: int = 7) -> dict | None:
    """How much of guessing success belongs to the prompt, and does finding its own answer help a guesser
    with the others once the prompt is held fixed?

    One row per judgment of another model's answer. Returns:
      icc: share of the variance in correctness that lies between prompts, between guessers and between
        authors (one-way intraclass correlation for each, so a factor with many levels gets no free credit);
      own_raw: correct rate on others when the guesser found its own answer minus when it did not;
      own_within: the same difference from a linear probability model with prompt, guesser and author fixed
        effects, so easy prompts and strong guessers no longer count for it, with a 95% interval that
        resamples prompts;
      by_category: accuracy per prompt category.
    """
    import numpy as np  # only for this analysis; the game itself needs no numpy

    latest = {}
    for r in records:
        if isinstance(r.get("guesses"), dict) and isinstance(r.get("labels"), dict) and game_key(r) == key:
            latest[(r["guesser"], r["prompt_id"], r.get("sample", 0))] = r
    rows = []
    for rec in latest.values():
        guesser = rec["guesser"]
        if guesser not in models:
            continue
        own_letter = next((L for L, a in rec["labels"].items() if a == guesser), None)
        found = own_letter is not None and rec["guesses"].get(own_letter) == guesser
        for letter, author in rec["labels"].items():
            if author in models and author != guesser:
                rows.append(((rec["prompt_id"], rec.get("sample", 0)), guesser, author, found,
                             rec["guesses"].get(letter) == author, rec.get("category")))
    if not rows:
        return None
    prompt_ids = sorted({r[0] for r in rows})
    position = {p: i for i, p in enumerate(prompt_ids)}
    p_idx = np.array([position[r[0]] for r in rows])
    g_idx = np.array([models.index(r[1]) for r in rows])
    a_idx = np.array([models.index(r[2]) for r in rows])
    found = np.array([r[3] for r in rows], dtype=float)
    y = np.array([r[4] for r in rows], dtype=float)

    def icc(groups):
        levels, inv = np.unique(groups, return_inverse=True)
        n_g = np.bincount(inv).astype(float)
        if len(levels) < 2 or len(y) <= len(levels):
            return None
        means = np.bincount(inv, weights=y) / n_g
        msb = float((n_g * (means - y.mean()) ** 2).sum() / (len(levels) - 1))
        msw = float(((y - means[inv]) ** 2).sum() / (len(y) - len(levels)))
        n0 = (len(y) - (n_g ** 2).sum() / len(y)) / (len(levels) - 1)
        denom = msb + (n0 - 1) * msw
        return max(0.0, float((msb - msw) / denom)) if denom > 0 else None

    def within(take, cluster):
        """Coefficient on `found` with prompt, guesser and author fixed effects.

        The prompt dummies are absorbed by subtracting each cluster's means (Frisch-Waugh-Lovell), which gives
        the same coefficient as the full dummy regression at a fraction of the cost.
        """
        X = np.hstack([found[take][:, None],
                       (g_idx[take][:, None] == np.arange(1, len(models))[None, :]).astype(float),
                       (a_idx[take][:, None] == np.arange(1, len(models))[None, :]).astype(float),
                       y[take][:, None]])
        _, inv = np.unique(cluster, return_inverse=True)
        n = np.bincount(inv).astype(float)
        sums = np.stack([np.bincount(inv, weights=X[:, j]) for j in range(X.shape[1])], axis=1)
        X = X - (sums / n[:, None])[inv]
        if not X[:, 0].any():
            return None
        coef, *_ = np.linalg.lstsq(X[:, :-1], X[:, -1], rcond=None)
        return float(coef[0])

    effect = within(np.arange(len(y)), p_idx)
    by_prompt = [np.flatnonzero(p_idx == i) for i in range(len(prompt_ids))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps if effect is not None else 0):
        picks = rng.integers(0, len(prompt_ids), size=len(prompt_ids))
        take = np.concatenate([by_prompt[i] for i in picks])
        cluster = np.concatenate([np.full(len(by_prompt[i]), j) for j, i in enumerate(picks)])
        b = within(take, cluster)
        if b is not None:
            draws.append(b)
    found_rate = lambda f: float(y[found == f].mean()) if (found == f).any() else None  # noqa: E731
    raw = (found_rate(1.0) - found_rate(0.0)) if None not in (found_rate(1.0), found_rate(0.0)) else None
    cats = sorted({r[5] for r in rows if r[5]})
    return {
        "key": list(key),
        "n_judgments": len(rows),
        "n_prompts": len(prompt_ids),
        "icc": {"prompt": icc(p_idx), "guesser": icc(g_idx), "author": icc(a_idx)},
        "own_raw": raw,
        "own_within": {"effect": effect,
                       "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws else None},
        "by_category": {c: float(np.mean([r[4] for r in rows if r[5] == c])) for c in cats},
    }


def print_summary(summary: dict):
    if not summary:
        print("No parsed guesses to score.")
        return
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"

    def span(ci):
        return f" [{pct(ci[0])}, {pct(ci[1])}]" if ci else ""

    print(f"\n{summary['title']}: {summary['n_calls']} calls, {summary['n_judgments']} judgments of other models' answers")
    print(f"Overall accuracy {pct(summary['accuracy'])}{span(summary['ci'])}  (chance {pct(summary['chance'])})")
    own = summary["own_answer_effect"]
    if own["found"] is not None:
        print(f"Others named correctly when a guesser found its own answer {pct(own['found'])}, "
              f"when it did not {pct(own['not_found'])}")
    if summary["assign"] == "independent":
        print(f"Calls that named some model twice: {pct(summary['repeat_rate'])} (repeats were allowed)")
    vote = summary["vote"]
    print(f"Majority of the other guessers: {pct(vote['accuracy'])}{span(vote['ci'])} over {vote['n']} answers "
          f"({vote['ties']} ties split evenly, typically {vote['voters']} voters)")
    wv = summary.get("vote_weighted")
    if wv:
        print(f"Votes weighted by skill on the other half, each model once per prompt: {pct(wv['accuracy'])}{span(wv['ci'])}")
    if summary.get("confused_pairs"):
        print("Most confused pairs: " + "; ".join(f"{p['pair'][0]} / {p['pair'][1]} {pct(p['share'])} of misses"
                                                  for p in summary["confused_pairs"]))
    print("\nBest guessers (other authors named correctly)")
    for i, row in enumerate(summary["guessers"], 1):
        print(f"  {i}. {row['model']:22} {pct(row['accuracy']):7}{span(row['ci']):20}  ({row['correct']}/{row['n']})"
              f"   self-recognition {pct(row['self_rate'])}")
    print("\nMost recognizable (named correctly by the others)")
    for i, row in enumerate(summary["recognizable"], 1):
        print(f"  {i}. {row['model']:22} {pct(row['recall']):7}{span(row['ci']):20}  ({row['correct']}/{row['n']})")


async def main():
    ap = argparse.ArgumentParser(description="Author-matching game across the Spectra roster")
    add_run_args(ap)
    ap.add_argument("--limit", type=int, default=None, help="only N prompts, stratified across categories")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="in-flight calls per provider (default: config), capped by PeerRank's PROVIDER_CONCURRENCY")
    ap.add_argument("--mode", choices=MODES, default="blind",
                    help="blind: prompt and answers only. briefed: also a Spectra research profile. "
                         "heldout: briefed with notes from the other half of this run's prompts")
    ap.add_argument("--research", default=None,
                    help="revision whose analysis.json briefs the guessers (briefed mode; default: this revision)")
    ap.add_argument("--independent", action="store_true",
                    help="name each answer on its own, repeats allowed, instead of using each model once")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the messages and estimate the cost without calling any model")
    args = ap.parse_args()
    if args.mode != "briefed" and args.research:
        raise SystemExit("--research is only used with --mode briefed")

    cfg = load_config(args.config)
    run_dir = revision_dir(cfg, args.revision)
    models = [m["name"] for m in cfg["models"]]
    if len(models) < 2:
        raise SystemExit("The guessing game needs at least two models in config.yaml")
    for name in models:
        providers.resolve({"name": name, "temperature": 0})

    all_prompts = json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))["prompts"]
    prompts = {p["id"]: p for p in all_prompts}
    chosen = {p["id"] for p in select_prompts(all_prompts, args.limit)}
    responses = load_jsonl(run_dir / "responses.jsonl")
    rounds = {k: v for k, v in panel_answers(responses, models).items() if k[0] in chosen}
    assign = "independent" if args.independent else "permutation"
    research_rev = {"blind": None, "briefed": args.research or cfg["revision"], "heldout": cfg["revision"]}[args.mode]
    key = (args.mode, research_rev, assign)
    out_path = run_dir / "guesses.jsonl"
    halves = holdout_halves(all_prompts)  # over the full set, so a --limit run keeps the same halves

    def prepare():
        """(briefs by half or None, jobs still to do, records so far)."""
        if args.mode == "briefed":
            _, brief = load_research(research_rev, models)
            print(f"Research brief from {research_rev} ({len(brief)} characters)")
            briefs = {"A": brief, "B": brief}
        elif args.mode == "heldout":
            briefs = heldout_briefs(args.config, run_dir, cfg["revision"], models, halves)
            print("Held-out briefs: " + ", ".join(f"half {h} {len(b)} characters" for h, b in briefs.items()))
        else:
            briefs = None
        records = load_jsonl(out_path)
        done = {(r["guesser"], r["prompt_id"], int(r.get("sample", 0)))
                for r in records if isinstance(r.get("guesses"), dict) and game_key(r) == key}
        jobs = [(guesser, pid, sample, round_)
                for (pid, sample), round_ in rounds.items()
                for guesser in models
                if (guesser, pid, sample) not in done]
        random.shuffle(jobs)
        print(f"{game_title(key)}: {len(jobs)} guesses to collect ({len(done)} already done) "
              f"on {len(rounds)} complete prompts -> {out_path}")
        return briefs, jobs, records

    def message_for(guesser, pid, sample, round_, briefs):
        order = shuffle_models(pid, sample, guesser, models)
        brief = briefs[halves[pid]] if briefs else None
        return order, build_message(prompts[pid]["prompt"], order, round_["texts"], models,
                                    brief=brief, independent=args.independent)

    if args.dry_run:
        briefs, jobs, records = prepare()
        if jobs:
            _, text = message_for(*jobs[0], briefs)
            preview = run_dir / "guess_preview.txt"
            preview.write_text(text, encoding="utf-8")
            sizes = [len(message_for(*job, briefs)[1]) for job in jobs]
            print(f"Messages: {len(jobs)}, median {sorted(sizes)[len(sizes) // 2]:,} characters. First one: {preview}")
            basis = "briefed" if args.mode == "heldout" else args.mode  # held-out messages are briefed-sized
            similar = [r["cost_usd"] for r in records if r.get("cost_usd") is not None and game_key(r)[0] == basis]
            if similar:
                print(f"Estimated cost ${len(jobs) * sum(similar) / len(similar):.2f} "
                      f"(mean of {len(similar)} earlier {basis} calls)")
        print("Dry run: no model was called.")
        return

    with writer_lock(out_path):
        briefs, jobs, records = prepare()
        if not jobs:
            print_summary(summarize_guesses(records, models, key, cfg["revision"]))
            return

        sems = providers.semaphores([{"name": m} for m in models], args.concurrency or cfg["concurrency_per_model"])
        # Guessers reason at length over six answers and a profile; at the answering budget some Gemini calls
        # ran out of tokens before writing the JSON.
        guess_tokens = int(cfg.get("guess_max_tokens", 32768))
        lock = asyncio.Lock()
        counts = {"ok": 0, "err": 0}

        async def work(guesser, pid, sample, round_):
            order, message = message_for(guesser, pid, sample, round_, briefs)
            letters = string.ascii_uppercase[: len(order)]
            rec = {"guesser": guesser, "prompt_id": pid, "sample": sample, "mode": args.mode, "assign": assign,
                   "category": round_.get("category"), "labels": dict(zip(letters, order)),
                   "timestamp": datetime.now(timezone.utc).isoformat()}
            if research_rev:
                rec["research"] = research_rev
            if args.mode == "heldout":
                rec["half"] = halves[pid]
            async with sems[guesser]:
                try:
                    reply = await providers.complete({"name": guesser, "temperature": 0, "max_tokens": guess_tokens},
                                                     message, timeout=cfg["timeout_seconds"], retries=cfg["retries"])
                    parsed = parse_assignment(reply.text, list(letters), models)
                    if parsed is None:
                        rec["error"] = "unparseable"
                        rec["raw"] = (reply.text or "")[:500]
                        if len(reply.text or "") > 500:
                            rec["raw_tail"] = reply.text[-1500:]  # where a truncated or rambling reply gave up
                        counts["err"] += 1
                    else:
                        rec["guesses"] = parsed
                        rec["cost_usd"] = reply.cost_usd
                        counts["ok"] += 1
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    counts["err"] += 1
            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n = counts["ok"] + counts["err"]
                if n % 25 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)} done, {counts['err']} errors")

        try:
            await asyncio.gather(*(work(*job) for job in jobs))
        finally:
            await providers.close()
        print(f"Finished: {counts['ok']} ok, {counts['err']} errors -> {out_path}")
        if counts["err"]:
            print("Re-run the same command to retry failed or unparseable calls.")
        print_summary(summarize_guesses(load_jsonl(out_path), models, key, cfg["revision"]))


if __name__ == "__main__":
    asyncio.run(main())
