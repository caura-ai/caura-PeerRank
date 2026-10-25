#!/usr/bin/env python3
"""Phase 2: what models know about voices beyond what they can describe, and whether voices spread.

  python -m spectra.voice imitate --revision oct-23-26 --dry-run     # E3, the impression contest
  python -m spectra.voice hide --revision oct-23-26 --dry-run        # E4, hide your own voice
  python -m spectra.voice contagion --revision oct-23-26 --dry-run   # E5, accent contagion
  python -m spectra.voice swap --revision oct-23-26 --dry-run        # contagion's style swap: content x voice of notes
  python -m spectra.voice budget --revision oct-23-26                # every phase 2 cost, from measured prices
  python -m spectra.voice score --revision oct-23-26                 # score whatever is collected; no calls

Every call uses the panel's answering settings and no system prompt. Experiments append to
data/spectra/<revision>/voice.jsonl, resumable on (experiment, condition, writer, other model, prompt); score writes
voice.json. The answers being imitated, rewritten or read as notes are this revision's; the portraits and coaching
come from --portraits-from (default oct-14-26), another run, so they are not measured on those answers.
The designs, hypotheses and tests are fixed in spectra/prereg/ (DRAFT-voice-phase2.md until registered).
"""
import argparse
import asyncio
import hashlib
import json
import math
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from spectra import DATA_ROOT, add_run_args, load_config, providers, run_dir as revision_dir, writer_lock

EXPERIMENTS = ("imitate", "hide", "contagion")
PER_CATEGORY = {"imitate": 3, "hide": 5, "contagion": 4}
N_NOTES = 5
SOURCES = ("claude-fable-5.1", "gemini-4-argon", "gpt-6-astra")  # three distinct accents: em dashes, long, short
# Fixed from oct-23-26's accent table among the style traits (no coding needed): model share >= 5% and <= 2%.
ACCENT = ("emdash", "length", "vocab", "hedging")
STEREOTYPE = ("caveats", "warm_open", "stock", "self_ref", "exuberance")
SEED = 7


# ----------------------------------------------------------------------------- prompts and texts
def everyday(prompts: list[dict]) -> list[dict]:
    """Prompts with no trap tag and no stated limit, where a model's own voice has room."""
    return [p for p in prompts if not p.get("tags") and not p.get("constraint")]


def pick_prompts(prompts: list[dict], per_category: int, salt: str, exclude=()) -> list[dict]:
    """The same everyday prompts for every model: per category, the first by a salted hash of the id."""
    by = {}
    for p in everyday(prompts):
        if p["id"] not in set(exclude):
            by.setdefault(p["category"], []).append(p)
    out = []
    for c in sorted(by):
        out += sorted(by[c], key=lambda p: hashlib.sha256(f"{salt}|{p['id']}".encode()).hexdigest())[:per_category]
    return out


def note_prompts(prompts: list[dict]) -> list[dict]:
    """The prompts whose answers are the notes in E5: five everyday prompts from five categories."""
    one_each = pick_prompts(prompts, 1, "notes")
    return sorted(one_each, key=lambda p: hashlib.sha256(f"notes|{p['category']}".encode()).hexdigest())[:N_NOTES]


def load_texts(run_dir: Path) -> dict:
    """{(model, prompt): text}, the last successful first-sample answer."""
    out = {}
    for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("text") and not r.get("error") and int(r.get("sample", 0)) == 0:
            out[(r["model"], r["prompt_id"])] = r["text"]
    return out


def portrait_lines(analysis: dict, models: list[str]) -> dict:
    """{model: one-line measured portrait} from the plain profile's portraits."""
    from spectra.profile import portraits

    lines = {}
    for line in portraits(analysis, models).splitlines():
        for m in models:
            if line.startswith(f"- {m}: "):
                lines[m] = line[len(f"- {m}: "):]
    return lines


def coaching(analysis: dict, model: str) -> list[str]:
    """The habits that give a model away: its accent card, then its three largest everyday departures."""
    habits = [c["text"] for c in (analysis.get("accent") or {}).get("cards", {}).get(model, [])]
    everyday_traits = [t for t in analysis["traits"] if not t.get("tag") and t["key"] != "adherence"]
    for t in sorted(everyday_traits, key=lambda t: -abs(t["per_model"][model]["d"]))[:3]:
        text = t["high"] if t["per_model"][model]["d"] > 0 else t["low"]
        if text not in habits:
            habits.append(text)
    return habits


# ----------------------------------------------------------------------------- messages
def imitate_message(prompt: str, target: str, portrait: str | None = None) -> str:
    msg = (f"Answer the request below the way {target} would answer it: its usual length, formatting, tone, "
           f"punctuation and wording, as if {target} itself were replying. Reply with the answer only, "
           "without any comment about this instruction.\n\n"
           f"Request:\n{prompt}")
    if portrait:
        msg += f"\n\nMeasured habits of {target}, from an earlier study of its answers: {portrait}"
    return msg


def hide_message(prompt: str, answer: str, habits: list[str] | None = None) -> str:
    msg = ("Below is a request and an answer you wrote to it earlier. Rewrite your answer so that a reader could not "
           "tell which AI model wrote it. Keep the content and the facts. Reply with the rewritten answer only.\n\n"
           f"Request:\n{prompt}\n\nYour earlier answer:\n{answer}")
    if habits:
        msg += ("\n\nAn earlier study found these habits in your answers, and they give you away: "
                + "; ".join(h[0].lower() + h[1:] for h in habits) + ". Avoid them.")
    return msg


def contagion_message(prompt: str, notes: list[tuple[str, str]] | None) -> str:
    """notes: [(request, answer)] from earlier work, or None for the plain prompt. Nothing asks for imitation."""
    if not notes:
        return prompt
    blocks = "\n\n".join(f"--- Note {i} ---\nRequest: {q}\nAnswer:\n{a}" for i, (q, a) in enumerate(notes, 1))
    return f"Notes from earlier work, for context:\n\n{blocks}\n\n--- New request ---\n{prompt}"


def rewrite_message(prompt: str, answer: str) -> str:
    """The style swap: a model rewrites an answer in its own words, keeping the content."""
    return ("Below is a request and an answer to it. Rewrite the answer in your own words and your own usual style, "
            "keeping all of its content and facts. Reply with the rewritten answer only.\n\n"
            f"Request:\n{prompt}\n\nAnswer:\n{answer}")


# The style swap crosses whose content the notes carry with whose voice they are written in. First letter: content,
# second: voice; O is the reader's own, S the source's. Every cell's notes are rewrites, the reader's own (OO) and the
# source's own (SS) included, so all four cells went through the same rewriting step.
SWAP_CELLS = ("OO", "OS", "SO", "SS")


def swap_rewrite_jobs(models: list[str], prompts: list[dict], texts: dict) -> list[dict]:
    """Stage 1 of the swap: every (rewriter, author) pair the four cells need, over the five note prompts."""
    pairs = set()
    for t in models:
        pairs.add((t, t))
        for s in SOURCES:
            if s != t:
                pairs |= {(s, t), (t, s)}  # the source voicing the reader's content, and the reverse
    return [{"experiment": "swap-notes", "condition": "REWRITE", "writer": rewriter, "other": author,
             "prompt_id": q["id"], "category": q["category"], "message": rewrite_message(q["prompt"], texts[(author, q["id"])])}
            for rewriter, author in sorted(pairs) for q in note_prompts(prompts)]


def swap_answer_jobs(models: list[str], prompts: list[dict], rewrites: dict, texts: dict | None = None) -> list[dict]:
    """Stage 2: the reader answers the contagion prompts after one of the four note sets.

    rewrites: {(rewriter, author, note prompt): text}. For a dry run before stage 1, texts stands in: each note is the
    author's original answer, which has about the same length, so the cost estimate holds.
    """
    notes_p = note_prompts(prompts)

    def notes(author, voice):
        return [(q["prompt"], rewrites.get((voice, author, q["id"])) or (texts or {}).get((author, q["id"]))) for q in notes_p]

    jobs = []
    for p in pick_prompts(prompts, PER_CATEGORY["contagion"], "contagion", exclude={q["id"] for q in notes_p}):
        def add(cell, t, s, content, voice):
            jobs.append({"experiment": "swap", "condition": cell, "writer": t, "other": s, "prompt_id": p["id"],
                         "category": p["category"], "message": contagion_message(p["prompt"], notes(content, voice))})
        for t in models:
            add("OO", t, t, t, t)
            for s in SOURCES:
                if s != t:
                    add("OS", t, s, t, s)
                    add("SO", t, s, s, t)
                    add("SS", t, s, s, s)
    return jobs


def rewrites_from(rows) -> dict:
    return {(r["writer"], r["other"], r["prompt_id"]): r["text"] for r in rows if r["experiment"] == "swap-notes"}


# ----------------------------------------------------------------------------- jobs
def jobs_for(experiment: str, models: list[str], prompts: list[dict], texts: dict, portraits_analysis: dict) -> list[dict]:
    """Every call an experiment needs: {experiment, condition, writer, other, prompt_id, category, message}."""
    jobs = []

    def add(condition, writer, other, p, message):
        jobs.append({"experiment": experiment, "condition": condition, "writer": writer, "other": other,
                     "prompt_id": p["id"], "category": p["category"], "message": message})

    if experiment == "imitate":
        lines = portrait_lines(portraits_analysis, models)
        for p in pick_prompts(prompts, PER_CATEGORY["imitate"], "imitate"):
            for m in models:
                add("SELF", m, m, p, imitate_message(p["prompt"], m))
                for t in models:
                    if t != m:
                        add("NAME", m, t, p, imitate_message(p["prompt"], t))
                        add("PROFILE", m, t, p, imitate_message(p["prompt"], t, lines[t]))
    elif experiment == "hide":
        for p in pick_prompts(prompts, PER_CATEGORY["hide"], "hide"):
            for m in models:
                answer = texts[(m, p["id"])]
                add("UNCOACHED", m, m, p, hide_message(p["prompt"], answer))
                add("COACHED", m, m, p, hide_message(p["prompt"], answer, coaching(portraits_analysis, m)))
    elif experiment == "contagion":
        notes_p = note_prompts(prompts)
        own = {m: [(q["prompt"], texts[(m, q["id"])]) for q in notes_p] for m in models}
        for p in pick_prompts(prompts, PER_CATEGORY["contagion"], "contagion", exclude={q["id"] for q in notes_p}):
            for t in models:
                add("NONE", t, None, p, contagion_message(p["prompt"], None))
                add("SELF", t, t, p, contagion_message(p["prompt"], own[t]))
                for s in SOURCES:
                    if s != t:
                        add("SOURCE", t, s, p, contagion_message(p["prompt"], own[s]))
    else:
        raise ValueError(f"unknown experiment {experiment}")
    return jobs


def job_key(j: dict) -> tuple:
    return j["experiment"], j["condition"], j["writer"], j.get("other"), j["prompt_id"]


def estimate(jobs: list[dict], analysis: dict, prompts: dict) -> tuple[float, dict]:
    """Measured cost per answer for each writer in this run, plus the added message text at its input price."""
    per_answer = {m: s["total_cost_usd"] / s["responses"] for m, s in analysis["runstats"].items() if s["responses"]}
    total, by = 0.0, {}
    for j in jobs:
        extra = max(len(j["message"]) - len(prompts[j["prompt_id"]]["prompt"]), 0)
        c = per_answer[j["writer"]] + (providers.input_cost({"name": j["writer"]}, extra / 4) or 0.0)
        total += c
        by[j["condition"]] = by.get(j["condition"], 0.0) + c
    return total, by


# ----------------------------------------------------------------------------- measuring
def trait_columns() -> list[tuple[str, str]]:
    from spectra.analyze import STYLE_TRAITS

    return [(k, col) for k, col, *_ in STYLE_TRAITS if k != "adherence"]


def vector(text: str, category: str) -> np.ndarray:
    from spectra.analyze import features

    f, _ = features(text or "", category)
    return np.array([float(f.get(col, float("nan"))) for _, col in trait_columns()], dtype=float)


class Baseline:
    """The run's real answers measured on the everyday style traits, in units of each trait's within-model SD."""

    def __init__(self, texts: dict, prompts: dict, models: list[str]):
        self.keys = [k for k, _ in trait_columns()]
        self.models = models
        self.raw = {(m, pid): vector(t, prompts[pid]["category"]) for (m, pid), t in texts.items()
                    if m in models and pid in prompts}
        with warnings.catch_warnings():  # traits a short text cannot have are NaN, and a model may lack them all
            warnings.simplefilter("ignore", RuntimeWarning)
            sds = []
            for i in range(len(self.keys)):
                per = [np.nanstd([v[i] for (m, _), v in self.raw.items() if mm == m], ddof=1) for mm in models]
                sd = float(np.nanmean(per)) if np.isfinite(per).any() else float("nan")
                sds.append(sd if np.isfinite(sd) and sd > 0 else 1.0)
            self.sd = np.array(sds)
            self.v = {k: v / self.sd for k, v in self.raw.items()}
            self.mean = {m: np.nanmean([v for (mm, _), v in self.v.items() if mm == m], axis=0) for m in models}

    def z(self, text: str, category: str) -> np.ndarray:
        return vector(text, category) / self.sd

    def idx(self, keys) -> list[int]:
        return [self.keys.index(k) for k in keys if k in self.keys]


def project(v: np.ndarray, start: np.ndarray, end: np.ndarray) -> float | None:
    """How far v sits from start toward end, as a fraction of the way (0 = start, 1 = end), over finite traits."""
    ok = np.isfinite(v) & np.isfinite(start) & np.isfinite(end)
    d = (end - start)[ok]
    if d @ d < 1e-9:
        return None
    return float((v - start)[ok] @ d / (d @ d))


def prompt_ci(by_prompt: dict, reps: int = 2000, seed: int = SEED) -> dict:
    """Mean of all values and a 95% interval resampling prompts. by_prompt: {prompt: [values]}."""
    keys = [k for k, v in by_prompt.items() if v]
    if not keys:
        return {"mean": None, "ci": None, "n": 0, "prompts": 0}
    sums = np.array([sum(by_prompt[k]) for k in keys])
    ns = np.array([len(by_prompt[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(keys), size=(reps, len(keys)))
    boots = sums[draw].sum(axis=1) / ns[draw].sum(axis=1)
    return {"mean": float(sums.sum() / ns.sum()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": int(ns.sum()), "prompts": len(keys)}


def above(stat: dict, value: float = 0.0) -> bool | None:
    return None if not stat.get("ci") else stat["ci"][0] > value


# ----------------------------------------------------------------------------- scoring
def described_sides(records: list[dict], models: list[str], samples=(1, 2, 3, 4)) -> dict:
    """{(describer, target, trait): +1/-1}: which side of itself a describer puts a target, by majority over samples."""
    from spectra.describe import ESTIMATES

    est = {}
    for r in records:
        if r.get("estimates") and int(r.get("sample", 0)) in samples:
            est.setdefault(int(r.get("sample", 0)), {})[(r["describer"], r["target"])] = r["estimates"]
    out = {}
    for key, (trait, _, _) in ESTIMATES.items():
        for m in models:
            for t in models:
                if t == m:
                    continue
                votes = []
                for got in est.values():
                    a, b = (got.get((m, t)) or {}).get(key), (got.get((m, m)) or {}).get(key)
                    if a is not None and b is not None and a != b:
                        votes.append(1 if a > b else -1)
                if votes and sum(votes) != 0:
                    out[(m, t, trait)] = 1 if sum(votes) > 0 else -1
    return out


def score_imitate(rows: list[dict], base: Baseline, clf, sides: dict) -> dict:
    models = base.models
    frac = {c: {} for c in ("NAME", "PROFILE")}
    pairs, self_shift, name_shift = {}, {}, {}
    probs = {}
    for r in rows:
        m, t, pid = r["writer"], r["other"], r["prompt_id"]
        if (m, pid) not in base.v or (t, pid) not in base.v:
            continue
        v, a = base.z(r["text"], r["category"]), base.v[(m, pid)]
        ok = np.isfinite(v) & np.isfinite(a)
        dist = float(np.sqrt((((v - a)[ok]) ** 2).mean())) if ok.any() else None
        if r["condition"] == "SELF":
            self_shift.setdefault(pid, []).append(dist)
            continue
        f = project(v, a, base.v[(t, pid)])
        if f is not None:
            frac[r["condition"]].setdefault(pid, []).append(f)
            pairs.setdefault((m, t, pid), {})[r["condition"]] = f
        if r["condition"] == "NAME":
            name_shift.setdefault(pid, []).append(dist)
        probs.setdefault(r["condition"], []).append((pid, m, t, r["text"]))
    paired = {}
    for (m, t, pid), d in pairs.items():
        if "NAME" in d and "PROFILE" in d:
            paired.setdefault(pid, []).append(d["PROFILE"] - d["NAME"])

    # I3: traits the imitator described on the wrong side, with a measured gap of half an SD or more
    moved = {}
    keys = base.keys
    for r in rows:
        if r["condition"] != "NAME":
            continue
        m, t, pid = r["writer"], r["other"], r["prompt_id"]
        if (m, pid) not in base.v:
            continue
        v, a = base.z(r["text"], r["category"]), base.v[(m, pid)]
        for i, k in enumerate(keys):
            gap = base.mean[t][i] - base.mean[m][i]
            side = sides.get((m, t, k))
            if side is None or abs(gap) < 0.5 or side == np.sign(gap):
                continue
            delta = v[i] - a[i]
            if np.isfinite(delta) and delta != 0:
                moved.setdefault(pid, []).append(float(np.sign(delta) == np.sign(gap)))

    attribution = {}
    if clf is not None:
        for cond, items in probs.items():
            P = clf.predict_proba([x[3] for x in items])
            to_target = {}
            to_writer = {}
            for (pid, m, t, _), p in zip(items, P):
                to_target.setdefault(pid, []).append(float(clf.models[int(p.argmax())] == t))
                to_writer.setdefault(pid, []).append(float(clf.models[int(p.argmax())] == m))
            attribution[cond] = {"named_target": prompt_ci(to_target), "named_imitator": prompt_ci(to_writer)}
    i1, i2, i3 = prompt_ci(frac["NAME"]), prompt_ci(paired), prompt_ci(moved)

    # Post hoc (deviation 2 of the registration): a fresh answer drifts from the imitator's single earlier answer toward
    # everyone else's, so each imitation is compared with the same model's SELF answer on the same axis.
    selfv = {(r["writer"], r["prompt_id"]): base.z(r["text"], r["category"]) for r in rows if r["condition"] == "SELF"}
    vs_self, drift, moved_self = {c: {} for c in ("NAME", "PROFILE")}, {}, {}
    for r in rows:
        m, t, pid = r["writer"], r["other"], r["prompt_id"]
        if r["condition"] not in vs_self or (m, pid) not in selfv or (m, pid) not in base.v:
            continue
        a, b = base.v[(m, pid)], base.v[(t, pid)]
        f, f0 = project(base.z(r["text"], r["category"]), a, b), project(selfv[(m, pid)], a, b)
        if f is not None and f0 is not None:
            vs_self[r["condition"]].setdefault(pid, []).append(f - f0)
            if r["condition"] == "NAME":
                drift.setdefault(pid, []).append(f0)
                for i, k in enumerate(keys):
                    gap = base.mean[t][i] - base.mean[m][i]
                    side = sides.get((m, t, k))
                    if side is None or abs(gap) < 0.5 or side == np.sign(gap):
                        continue
                    delta = selfv[(m, pid)][i] - a[i]
                    if np.isfinite(delta) and delta != 0:
                        moved_self.setdefault(pid, []).append(float(np.sign(delta) == np.sign(gap)))
    named_base = {}
    if clf is not None:
        S = [r for r in rows if r["condition"] == "SELF"]
        for r, p in zip(S, clf.predict_proba([r["text"] for r in S]) if S else []):
            for t in models:
                if t != r["writer"]:
                    named_base.setdefault(r["prompt_id"], []).append(float(clf.models[int(p.argmax())] == t))
    posthoc = {"self_drift_toward_target": prompt_ci(drift), "name_minus_self": prompt_ci(vs_self["NAME"]),
               "profile_minus_self": prompt_ci(vs_self["PROFILE"]), "I3_cells_self": prompt_ci(moved_self),
               "self_named_as_target": prompt_ci(named_base)}
    return {"posthoc_vs_self": posthoc,
            "I1_name_moves_toward_target": i1 | {"confirmed": above(i1)},
            "I2_profile_beats_name": i2 | {"confirmed": above(i2)},
            "I3_moves_right_where_described_wrong": i3 | {"confirmed": above(i3, 0.5)},
            "profile_fraction": prompt_ci(frac["PROFILE"]),
            "shift_from_own_answer": {"SELF": prompt_ci(self_shift), "NAME": prompt_ci(name_shift)},
            "I4_attribution": attribution}


def score_hide(rows: list[dict], base: Baseline, clf, texts: dict) -> dict:
    acc, gaps, paired = {}, {}, {}
    correct = {}
    acc_idx, st_idx = base.idx(ACCENT), base.idx(STEREOTYPE)
    for r in rows:
        m, pid = r["writer"], r["prompt_id"]
        if (m, pid) not in base.v:
            continue
        d = np.abs(base.z(r["text"], r["category"]) - base.v[(m, pid)])
        if r["condition"] == "UNCOACHED":
            a, s = d[acc_idx], d[st_idx]
            if np.isfinite(a).any() and np.isfinite(s).any():
                gaps.setdefault(pid, []).append(float(np.nanmean(s) - np.nanmean(a)))
    if clf is not None:
        items = [r for r in rows if (r["writer"], r["prompt_id"]) in base.v]
        P = clf.predict_proba([r["text"] for r in items])
        for r, p in zip(items, P):
            hit = float(clf.models[int(p.argmax())] == r["writer"])
            acc.setdefault(r["condition"], {}).setdefault(r["prompt_id"], []).append(hit)
            correct[(r["condition"], r["writer"], r["prompt_id"])] = hit
        originals = sorted({(r["writer"], r["prompt_id"]) for r in items})
        P0 = clf.predict_proba([texts[k] for k in originals])
        acc["ORIGINAL"] = {}
        for (m, pid), p in zip(originals, P0):
            acc["ORIGINAL"].setdefault(pid, []).append(float(clf.models[int(p.argmax())] == m))
        for (cond, m, pid), hit in correct.items():
            if cond == "UNCOACHED" and ("COACHED", m, pid) in correct:
                paired.setdefault(pid, []).append(hit - correct[("COACHED", m, pid)])
    chance = 1 / len(base.models)
    h1 = prompt_ci(acc.get("UNCOACHED", {}))
    h2, h3 = prompt_ci(gaps), prompt_ci(paired)
    return {"H1_uncoached_still_named": h1 | {"chance": chance, "confirmed": above(h1, chance)},
            "H2_stereotype_moves_more_than_accent": h2 | {"confirmed": above(h2)},
            "H3_coaching_hides_better": h3 | {"confirmed": above(h3)},
            "accuracy": {c: prompt_ci(v) for c, v in acc.items()}}


def score_contagion(rows: list[dict], base: Baseline, clf) -> dict:
    got = {(r["condition"], r["writer"], r.get("other"), r["prompt_id"]): r for r in rows}
    shift, per_trait, amp, pdiff = {}, {k: {} for k in base.keys}, {}, {}
    panel = np.nanmean([base.mean[m] for m in base.models], axis=0)
    for (cond, t, s, pid), r in got.items():
        if cond != "SOURCE" or ("SELF", t, t, pid) not in got:
            continue
        v_s = base.z(r["text"], r["category"])
        v_0 = base.z(got[("SELF", t, t, pid)]["text"], r["category"])
        axis = base.mean[s] - base.mean[t]
        f = project(v_s - v_0 + base.mean[t], base.mean[t], base.mean[s])  # the change, measured along t -> s
        if f is not None:
            shift.setdefault(pid, []).append(f)
        for i, k in enumerate(base.keys):
            d = (v_s - v_0)[i] * np.sign(axis[i])
            if np.isfinite(d) and axis[i] != 0:
                per_trait[k].setdefault(pid, []).append(float(d))
    for (cond, t, s, pid), r in got.items():
        if cond == "SELF" and ("NONE", t, None, pid) in got:
            v_self = base.z(r["text"], r["category"])
            v_none = base.z(got[("NONE", t, None, pid)]["text"], r["category"])
            f = project(v_self - v_none + panel, panel, base.mean[t])
            if f is not None:
                amp.setdefault(pid, []).append(f)
    if clf is not None:
        items = [(k, r) for k, r in got.items() if k[0] == "SOURCE" and ("SELF", k[1], k[1], k[3]) in got]
        if items:
            P = clf.predict_proba([r["text"] for _, r in items])
            P0 = clf.predict_proba([got[("SELF", k[1], k[1], k[3])]["text"] for k, _ in items])
            for ((_, t, s, pid), _), p, p0 in zip(items, P, P0):
                j = clf.models.index(s)
                pdiff.setdefault(pid, []).append(float(p[j] - p0[j]))
    c1, c2 = prompt_ci(shift), prompt_ci(pdiff)
    traits = {k: prompt_ci(v) for k, v in per_trait.items()}
    group = lambda ks: float(np.mean([traits[k]["mean"] for k in ks if traits.get(k, {}).get("mean") is not None]))  # noqa: E731
    return {"C1_moves_toward_source": c1 | {"confirmed": above(c1)},
            "C2_classifier_leans_to_source": c2 | {"confirmed": above(c2)},
            "C3_per_trait": traits, "C3_accent_vs_stereotype": {"accent": group(ACCENT), "stereotype": group(STEREOTYPE)},
            "C3_self_notes_amplify": prompt_ci(amp)}


def score_swap(rows: list[dict], base: Baseline, clf, rewrite_rows: list[dict], texts: dict) -> dict:
    """The style swap. Per (reader, source, prompt), with v the reader's answer after each note set:
    voice effect = mean of (OS - OO) and (SS - SO), content effect = mean of (SO - OO) and (SS - OS), each projected on
    the axis from the reader's accent to the source's (a fraction of the way). The classifier's probability for the
    source is compared the same way. The manipulation check asks whether cross rewrites (one model voicing another's
    content) are attributed to the rewriter more often than to the content's author.
    """
    got = {(r["condition"], r["writer"], r.get("other"), r["prompt_id"]): r for r in rows}
    stats = {k: {} for k in ("voice", "content", "voice_without_length", "content_without_length", "ss_minus_oo")}
    probs, pairs = {}, []
    length = base.keys.index("length")
    for (cell, t, s, pid), r in got.items():
        if cell != "SS":
            continue
        need = {"OO": ("OO", t, t, pid), "OS": ("OS", t, s, pid), "SO": ("SO", t, s, pid), "SS": ("SS", t, s, pid)}
        if any(k not in got for k in need.values()):
            continue
        V = {c: base.z(got[k]["text"], r["category"]) for c, k in need.items()}
        a, b = base.mean[t], base.mean[s]
        effects = {"voice": 0.5 * ((V["OS"] - V["OO"]) + (V["SS"] - V["SO"])),
                   "content": 0.5 * ((V["SO"] - V["OO"]) + (V["SS"] - V["OS"])), "ss_minus_oo": V["SS"] - V["OO"]}
        for name, d in list(effects.items()):
            f = project(d + a, a, b)
            if f is not None:
                stats[name].setdefault(pid, []).append(f)
            if name in ("voice", "content"):
                d2, a2, b2 = d.copy(), a.copy(), b.copy()
                d2[length] = a2[length] = b2[length] = np.nan
                f2 = project(d2 + a2, a2, b2)
                if f2 is not None:
                    stats[name + "_without_length"].setdefault(pid, []).append(f2)
        pairs.append((t, s, pid, need))
    out = {k: prompt_ci(v) for k, v in stats.items()}
    out["voice"]["confirmed"] = above(out["voice"])
    out["voice_without_length"]["confirmed"] = above(out["voice_without_length"])
    if clf is not None and pairs:
        cells = [(t, s, pid, c, got[k]["text"]) for t, s, pid, need in pairs for c, k in need.items()]
        P = clf.predict_proba([x[4] for x in cells])
        p_src = {(t, s, pid, c): float(p[clf.models.index(s)]) for (t, s, pid, c, _), p in zip(cells, P)}
        for t, s, pid, _ in pairs:
            q = {c: p_src[(t, s, pid, c)] for c in SWAP_CELLS}
            probs.setdefault(pid, []).append(0.5 * ((q["OS"] - q["OO"]) + (q["SS"] - q["SO"])))
        out["classifier_voice"] = prompt_ci(probs) | {"confirmed": above(prompt_ci(probs))}
        cross = [r for r in rewrite_rows if r["writer"] != r["other"]]
        if cross:
            P = clf.predict_proba([r["text"] for r in cross])
            named = [clf.models[int(p.argmax())] for p in P]
            rw = float(np.mean([n == r["writer"] for n, r in zip(named, cross)]))
            au = float(np.mean([n == r["other"] for n, r in zip(named, cross)]))
            out["manipulation_check"] = {"named_rewriter": rw, "named_author": au, "n": len(cross), "passed": rw > au}
    return out


def score_portraits(run_dir: Path, models: list[str], revision: str, guesser: str = "gemini-4-argon",
                    sample: int = 1) -> dict | None:
    """E2: the portrait games against the measured profile and blind, for the one guesser that played them."""
    from spectra.guess import game_contrast, load_jsonl

    records = load_jsonl(run_dir / "guesses.jsonl")
    self_key = ("described", f"self, sample {sample}", "permutation", "base")
    peer_key = ("described", f"peer, sample {sample}", "permutation", "base")
    r1 = game_contrast(records, models, guesser, self_key, ("heldout", revision, "permutation", "base"))
    if r1 is None:
        return None
    r2 = game_contrast(records, models, guesser, self_key, ("blind", None, "permutation", "base"))
    r3 = game_contrast(records, models, guesser, self_key, peer_key)
    return {"R1_self_below_measured": r1 | {"confirmed": r1["ci"][1] < 0},
            "R2_self_above_blind": (r2 or {}) | {"confirmed": bool(r2) and r2["ci"][0] > 0},
            "R3_self_vs_peer": r3}


# ----------------------------------------------------------------------------- running
async def collect(jobs: list[dict], cfg: dict, out_path: Path, concurrency: int | None):
    entries = {m["name"]: m for m in cfg["models"]}
    sems = providers.semaphores(list(entries.values()), concurrency or cfg["concurrency_per_model"])
    lock = asyncio.Lock()
    counts = {"ok": 0, "err": 0}

    async def work(j):
        rec = {k: j[k] for k in ("experiment", "condition", "writer", "other", "prompt_id", "category")}
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        async with sems[j["writer"]]:
            try:
                reply = await providers.complete(entries[j["writer"]], j["message"], system_prompt=cfg.get("system_prompt"),
                                                 timeout=cfg["timeout_seconds"], retries=cfg["retries"])
                rec.update(reply.fields())
                counts["ok"] += 1
            except Exception as e:
                rec.update(text=None, error=f"{type(e).__name__}: {e}"[:500])
                counts["err"] += 1
        async with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n = counts["ok"] + counts["err"]
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} done, {counts['err']} errors")

    try:
        await asyncio.gather(*(work(j) for j in jobs))
    finally:
        await providers.close()
    print(f"Finished: {counts['ok']} ok, {counts['err']} errors -> {out_path}")


def latest(out_path: Path) -> dict:
    from spectra.guess import load_jsonl

    return {job_key(r): r for r in load_jsonl(out_path) if r.get("text") and not r.get("error")}


def budget(cfg: dict, run_dir: Path, analysis: dict, prompts: dict, models: list[str], texts: dict, portraits: dict):
    """Every phase 2 experiment's estimated cost, from this run's measured prices."""
    from spectra.guess import load_jsonl

    rows = []
    desc = [r for r in load_jsonl(run_dir / "descriptions.jsonl") if r.get("cost_usd") is not None]
    per_sample = {}
    for r in desc:
        per_sample[int(r.get("sample", 0))] = per_sample.get(int(r.get("sample", 0)), 0.0) + r["cost_usd"]
    describe_cost = float(np.mean(list(per_sample.values()))) if per_sample else 1.04
    rows.append(("E1 Describe, samples 2-4", 3 * describe_cost, f"3 x ${describe_cost:.2f}, the measured cost of a sample"))
    argon = [r["cost_usd"] for p in sorted(DATA_ROOT.glob("*/guesses.jsonl")) for r in load_jsonl(p)
             if r.get("guesser") == "gemini-4-argon" and r.get("cost_usd") is not None and r.get("mode") in ("heldout", "briefed")]
    per_call = float(np.mean(argon)) if argon else 0.079
    rows.append(("E2 Recognize from portraits", 2 * 200 * per_call, f"2 games x 200 calls x ${per_call:.3f}, argon's measured mean"))
    for e, label in (("imitate", "E3 Impression contest"), ("hide", "E4 Hide your own voice"), ("contagion", "E5 Accent contagion")):
        jobs = jobs_for(e, models, list(prompts.values()), texts, portraits)
        total, _ = estimate(jobs, analysis, prompts)
        rows.append((label, total, f"{len(jobs):,} calls"))
    answers = sum(s["total_cost_usd"] for s in analysis["runstats"].values())
    coding = 2 * len(prompts) * len(models) * 0.006  # two coders, about $0.006 each by token prices
    rows.append(("E6 v2 replication", answers + coding, f"answers ${answers:.2f} as in this run, coding about ${coding:.2f}"))
    rows.append(("Optional E7 two more answers per prompt", 2 * answers, f"2 x ${answers:.2f}"))
    print("Phase 2 budget, from measured prices (estimates assume no retries):")
    for label, cost, how in rows:
        print(f"  {label:42} ${cost:7.2f}   {how}")
    core = sum(c for label, c, _ in rows if not label.startswith("Optional"))
    print(f"  {'Subtotal':42} ${core:7.2f}\n  {'With the optional samples':42} ${core + rows[-1][1]:7.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2 voice experiments: imitate, hide, contagion")
    ap.add_argument("action", choices=EXPERIMENTS + ("swap", "budget", "score"))
    add_run_args(ap)
    ap.add_argument("--portraits-from", default="oct-14-26",
                    help="revision whose analysis.json gives the measured portraits and the coaching")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="build every message and estimate the cost; no calls")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run_dir = revision_dir(cfg, args.revision)
    models = [m["name"] for m in cfg["models"]]
    prompts = {p["id"]: p for p in json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))["prompts"]}
    analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    portraits = json.loads((DATA_ROOT / args.portraits_from / "analysis.json").read_text(encoding="utf-8"))
    texts = load_texts(run_dir)
    out_path = run_dir / "voice.jsonl"

    if args.action == "budget":
        budget(cfg, run_dir, analysis, prompts, models, texts, portraits)
        return
    if args.action == "score":
        return score(run_dir, cfg["revision"], models, prompts, texts, out_path)
    if args.action == "swap":
        return swap(cfg, run_dir, analysis, prompts, models, texts, out_path, args)

    jobs = jobs_for(args.action, models, list(prompts.values()), texts, portraits)
    done = latest(out_path)
    todo = [j for j in jobs if job_key(j) not in done]
    total, by = estimate(todo, analysis, prompts)
    print(f"{args.action}: {len(jobs)} calls in all, {len(todo)} still to collect, on "
          f"{len({j['prompt_id'] for j in jobs})} prompts -> {out_path}")
    print(f"Estimated cost ${total:.2f} (" + ", ".join(f"{c} ${v:.2f}" for c, v in by.items()) +
          "): each writer's measured cost per answer in this run, plus the added text at its input price")
    if args.dry_run:
        seen = set()
        preview = []
        for j in jobs:
            if j["condition"] not in seen:
                seen.add(j["condition"])
                preview.append(f"===== {j['condition']}: {j['writer']} -> {j['other']} on {j['prompt_id']} =====\n{j['message']}")
        path = run_dir / f"voice_preview-{args.action}.txt"
        path.write_text("\n\n".join(preview), encoding="utf-8")
        print(f"One message per condition: {path}\nDry run: no model was called.")
        return
    with writer_lock(out_path):
        random.shuffle(todo)
        asyncio.run(collect(todo, cfg, out_path, args.concurrency))


def swap(cfg, run_dir, analysis, prompts, models, texts, out_path, args):
    """The style swap in two stages: the rewrites, then the answers after each of the four note sets."""
    plist = list(prompts.values())
    rewrite_jobs = swap_rewrite_jobs(models, plist, texts)
    done = latest(out_path)
    todo_rw = [j for j in rewrite_jobs if job_key(j) not in done]
    rewrites = rewrites_from(done.values())
    answer_jobs = swap_answer_jobs(models, plist, rewrites, texts)
    todo_ans = [j for j in answer_jobs if job_key(j) not in done]
    cost_rw, _ = estimate(todo_rw, analysis, prompts)
    cost_ans, by = estimate(todo_ans, analysis, prompts)
    print(f"swap: {len(rewrite_jobs)} rewrites ({len(todo_rw)} to collect) and {len(answer_jobs)} answers "
          f"({len(todo_ans)} to collect) on {len({j['prompt_id'] for j in answer_jobs})} prompts -> {out_path}")
    print(f"Estimated cost ${cost_rw + cost_ans:.2f}: rewrites ${cost_rw:.2f}, answers ${cost_ans:.2f} ("
          + ", ".join(f"{c} ${v:.2f}" for c, v in by.items()) + "); each writer's measured cost per answer in this "
          "run, plus the added text at its input price" + ("; notes not yet rewritten are priced at their originals'"
                                                           " length" if todo_rw else ""))
    if args.dry_run:
        preview, seen = [], set()
        for j in rewrite_jobs[:1] + answer_jobs:
            if j["condition"] not in seen:
                seen.add(j["condition"])
                preview.append(f"===== {j['condition']}: {j['writer']} / {j['other']} on {j['prompt_id']} =====\n{j['message']}")
        path = run_dir / "voice_preview-swap.txt"
        path.write_text("\n\n".join(preview), encoding="utf-8")
        print(f"One message per stage and cell: {path}. Answer messages show the original notes until the rewrites "
              "exist.\nDry run: no model was called.")
        return
    with writer_lock(out_path):
        if todo_rw:
            random.shuffle(todo_rw)
            asyncio.run(collect(todo_rw, cfg, out_path, args.concurrency))
        rewrites = rewrites_from(latest(out_path).values())
        missing = [j for j in rewrite_jobs if (j["writer"], j["other"], j["prompt_id"]) not in rewrites]
        if missing:
            raise SystemExit(f"{len(missing)} rewrites failed; rerun the same command before any answer is collected")
        answer_jobs = swap_answer_jobs(models, plist, rewrites)
        todo_ans = [j for j in answer_jobs if job_key(j) not in latest(out_path)]
        random.shuffle(todo_ans)
        asyncio.run(collect(todo_ans, cfg, out_path, args.concurrency))


def score(run_dir: Path, revision: str, models: list[str], prompts: dict, texts: dict, out_path: Path):
    from spectra.attribute import reference_classifier
    from spectra.guess import load_jsonl

    rows = list(latest(out_path).values())
    base = Baseline(texts, prompts, models)
    results = {"revision": revision}
    portraits = score_portraits(run_dir, models, revision)
    if portraits:
        results["E2"] = portraits
    for e, fn in (("imitate", "E3"), ("hide", "E4"), ("contagion", "E5")):
        mine = [r for r in rows if r["experiment"] == e]
        if not mine:
            continue
        used = {r["prompt_id"] for r in mine}
        if e == "contagion":
            used |= {q["id"] for q in note_prompts(list(prompts.values()))}
        clf = reference_classifier(["oct-14-26", revision], prompts, models, exclude=used)
        if e == "imitate":
            sides = described_sides(load_jsonl(run_dir / "descriptions.jsonl"), models)
            results[fn] = score_imitate(mine, base, clf, sides)
        elif e == "hide":
            results[fn] = score_hide(mine, base, clf, texts)
        else:
            results[fn] = score_contagion(mine, base, clf)
    swap_rows = [r for r in rows if r["experiment"] == "swap"]
    if swap_rows:
        used = {r["prompt_id"] for r in swap_rows} | {q["id"] for q in note_prompts(list(prompts.values()))}
        clf = reference_classifier(["oct-14-26", revision], prompts, models, exclude=used)
        results["swap"] = score_swap(swap_rows, base, clf, [r for r in rows if r["experiment"] == "swap-notes"], texts)
    (run_dir / "voice.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps(results, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else x)[:6000])
    print(f"-> {run_dir / 'voice.json'}")


if __name__ == "__main__":
    main()
