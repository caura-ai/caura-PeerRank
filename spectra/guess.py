#!/usr/bin/env python3
"""The guessing game: each model sees a prompt's answers with the authors stripped off,
and matches every answer to a model on the roster.

Two modes. ``blind`` (the default) gives the prompt, the answers, and the roster names.
``briefed`` adds a research brief: the measured habits from a Spectra analysis.json
(length, formatting, hedging, signature words, and the rest). The brief does not label
the answers in front of the guesser, and the guesser is still not told which model it is.

One call per guesser per prompt. Letters are shuffled per guesser, so position is not
a shared cue. Each author is used once (there is one answer per model).

  python -m spectra.guess --revision oct-14-26 --limit 5              # blind smoke test
  python -m spectra.guess --revision oct-14-26                        # blind game
  python -m spectra.guess --revision oct-23-26 --mode briefed --research oct-14-26

Writes data/spectra/<revision>/guesses.jsonl. Records are keyed by mode, so the two
games can share a file. The report reads both on the next `python -m spectra.report`.
Headline scores exclude a model's own answer. Chance is 1/k.
"""
import argparse
import asyncio
import hashlib
import json
import random
import string
from datetime import datetime, timezone
from pathlib import Path

from spectra import DATA_ROOT, add_run_args, load_config, providers, run_dir as revision_dir
from spectra.run import select_prompts

ANSWER_CHAR_CAP = 20000
MODES = ("blind", "briefed")


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


def build_message(user_prompt: str, order: list[str], texts: dict[str, str], roster: list[str],
                  brief: str | None = None) -> str:
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
    return (
        "You are matching anonymous answers to the models that wrote them. "
        "You are not grading quality.\n\n"
        "Each answer below was written by a different model from this roster:\n"
        f"{names}\n\n"
        + research
        + how
        + "Assign every letter to exactly one model. Use each model once.\n\n"
        f"<user_prompt>\n{user_prompt}\n</user_prompt>\n\n"
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


def _record_mode(rec: dict) -> str:
    mode = rec.get("mode") or "blind"
    return mode if mode in MODES else "blind"


def summarize_guesses(records: list[dict], models: list[str], mode: str = "blind",
                      answer_revision: str | None = None) -> dict | None:
    """Recognizability and guessing skill for one mode. Own answers are excluded from both headlines.

    Records written before modes existed count as blind.
    """
    usable = [r for r in records
              if isinstance(r.get("guesses"), dict) and isinstance(r.get("labels"), dict) and _record_mode(r) == mode]
    if not usable or len(models) < 2:
        return None
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

    for rec in usable:
        guesser = rec["guesser"]
        if guesser not in idx:
            continue
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
            recognized_n[author] += 1
            recognized[author] += int(correct)
            if guess in idx:
                matrix[idx[author]][idx[guess]] += 1
            else:
                unresolved[author] += 1

    def rate(n_ok, n):
        return (n_ok / n) if n else None

    guessers = []
    recognizable = []
    for m in models:
        guessers.append({
            "model": m,
            "accuracy": rate(other_correct[m], other_n[m]),
            "correct": other_correct[m],
            "n": other_n[m],
            "self_rate": rate(self_correct[m], self_n[m]),
            "self_n": self_n[m],
        })
        recognizable.append({
            "model": m,
            "recall": rate(recognized[m], recognized_n[m]),
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
    sources = sorted({r.get("research") for r in usable if r.get("research")})
    scoring = (
        "Accuracy is the share of other models' answers named correctly. "
        "Recall is the share of other guessers who named this author. A model's own answer is left out of both, "
        f"and is reported separately as self-recognition. A random matching is right {1.0 / k:.0%} of the time."
    )
    if mode == "briefed":
        src = ", ".join(sources) if sources else "a Spectra profile"
        lead = (
            "Each guesser saw every answer labeled with shuffled letters and no author names, "
            f"plus research notes from {src}. The notes do not label these answers, "
            "and the guesser is not told which model it is. "
        )
        if answer_revision and sources == [answer_revision]:
            lead += "The notes were measured on this same set of answers, so each answer is one part of its model's average. "
        elif answer_revision and sources and answer_revision not in sources:
            lead += "The notes come from a different set of prompts than these answers. "
    else:
        lead = (
            "Each guesser saw every answer to a prompt, labeled with shuffled letters and no author names, "
            "and was not told which model it is. "
        )
    return {
        "mode": mode,
        "research": sources,
        "k": k,
        "chance": 1.0 / k,
        "n_calls": len(usable),
        "n_judgments": n_other,
        "accuracy": rate(n_hit, n_other),
        "labels": models,
        "guessers": guessers,
        "recognizable": recognizable,
        "matrix": row_rates,
        "note": lead + scoring,
    }


def print_summary(summary: dict):
    if not summary:
        print("No parsed guesses to score.")
        return
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    title = "Guess from the research" if summary.get("mode") == "briefed" else "Just guess"
    research = ", ".join(summary.get("research") or [])
    suffix = f"  (research {research})" if research else ""
    print(f"\n{title}: {summary['n_calls']} calls, {summary['n_judgments']} judgments of other models' answers{suffix}")
    print(f"Overall accuracy {pct(summary['accuracy'])}  (chance {pct(summary['chance'])})")
    print("\nBest guessers (other authors named correctly)")
    for i, row in enumerate(summary["guessers"], 1):
        print(f"  {i}. {row['model']:22} {pct(row['accuracy']):7}  ({row['correct']}/{row['n']})"
              f"   self-recognition {pct(row['self_rate'])}")
    print("\nMost recognizable (named correctly by the others)")
    for i, row in enumerate(summary["recognizable"], 1):
        print(f"  {i}. {row['model']:22} {pct(row['recall']):7}  ({row['correct']}/{row['n']})")


async def main():
    ap = argparse.ArgumentParser(description="Author-matching game across the Spectra roster")
    add_run_args(ap)
    ap.add_argument("--limit", type=int, default=None, help="only N prompts, stratified across categories")
    ap.add_argument("--concurrency", type=int, default=None, help="in-flight calls per guesser (default: config)")
    ap.add_argument("--mode", choices=MODES, default="blind",
                    help="blind: prompt and answers only. briefed: also give the Spectra research profile")
    ap.add_argument("--research", default=None,
                    help="revision whose analysis.json briefs the guessers (briefed mode; default: this revision)")
    args = ap.parse_args()
    if args.mode == "blind" and args.research:
        raise SystemExit("--research is only used with --mode briefed")

    cfg = load_config(args.config)
    run_dir = revision_dir(cfg, args.revision)
    models = [m["name"] for m in cfg["models"]]
    if len(models) < 2:
        raise SystemExit("The guessing game needs at least two models in config.yaml")
    for name in models:
        providers.resolve({"name": name, "temperature": 0})

    prompts = {p["id"]: p for p in json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))["prompts"]}
    chosen = {p["id"] for p in select_prompts(list(prompts.values()), args.limit)}
    responses = load_jsonl(run_dir / "responses.jsonl")
    rounds = {k: v for k, v in panel_answers(responses, models).items() if k[0] in chosen}

    brief = None
    research_rev = None
    if args.mode == "briefed":
        research_rev, brief = load_research(args.research or cfg["revision"], models)
        print(f"Research brief from {research_rev} ({len(brief)} characters)")

    out_path = run_dir / "guesses.jsonl"

    def _already_done(rec) -> bool:
        if not isinstance(rec.get("guesses"), dict) or _record_mode(rec) != args.mode:
            return False
        if args.mode == "briefed" and (rec.get("research") or "") != (research_rev or ""):
            return False
        return True

    done = {(r["guesser"], r["prompt_id"], int(r["sample"]))
            for r in load_jsonl(out_path) if _already_done(r)}
    jobs = [(guesser, pid, sample, round_)
            for (pid, sample), round_ in rounds.items()
            for guesser in models
            if (guesser, pid, sample) not in done]
    random.shuffle(jobs)
    print(f"{len(jobs)} guesses to collect ({len(done)} already done) "
          f"on {len(rounds)} complete prompts -> {out_path}")
    if not jobs:
        print_summary(summarize_guesses(load_jsonl(out_path), models, args.mode, cfg["revision"]))
        return

    cap = args.concurrency or cfg["concurrency_per_model"]
    sems = {m: asyncio.Semaphore(providers.max_concurrency({"name": m}, cap)) for m in models}
    lock = asyncio.Lock()
    counts = {"ok": 0, "err": 0}

    async def work(guesser, pid, sample, round_):
        order = shuffle_models(pid, sample, guesser, models)
        letters = string.ascii_uppercase[: len(order)]
        labels = dict(zip(letters, order))
        prompt = prompts[pid]["prompt"]
        message = build_message(prompt, order, round_["texts"], models, brief=brief)
        entry = {"name": guesser, "temperature": 0}
        rec = {"guesser": guesser, "prompt_id": pid, "sample": sample, "mode": args.mode,
               "category": round_.get("category"), "labels": labels,
               "timestamp": datetime.now(timezone.utc).isoformat()}
        if research_rev:
            rec["research"] = research_rev
        async with sems[guesser]:
            try:
                reply = await providers.complete(entry, message, timeout=cfg["timeout_seconds"], retries=cfg["retries"])
                parsed = parse_assignment(reply.text, letters, models)
                if parsed is None:
                    rec["error"] = "unparseable"
                    rec["raw"] = (reply.text or "")[:500]
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
    print_summary(summarize_guesses(load_jsonl(out_path), models, args.mode, cfg["revision"]))


if __name__ == "__main__":
    asyncio.run(main())
