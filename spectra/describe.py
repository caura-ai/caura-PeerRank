#!/usr/bin/env python3
"""Do models know how they write? Each panel model describes its own habits and every other model's.

  python -m spectra.describe --revision oct-23-26 --dry-run   # messages and a cost estimate, no calls
  python -m spectra.describe --revision oct-23-26             # one call per (describer, target)

Each reply holds a written profile (to try later as a guessing-game profile) and numeric estimates of the
traits Spectra measures. The estimates are scored against that revision's analysis.json with rank- and sign-based
measures: does a describer rank the panel the right way on each trait, does a model place itself better than the
others place it, does it know which side of the panel average it sits on, and do describers tell two models from
the same vendor apart as much as the measurements do (a test for brand stereotypes).
Writes data/spectra/<revision>/descriptions.jsonl (resumable) and descriptions.json (the scores). Rerunning once
every description is in makes no calls and just rescores.
Nothing here reads an answer: the descriptions are what each model believes before seeing any data.
"""
import argparse
import asyncio
import json
import math
import random
import re

from spectra import add_run_args, load_config, providers, run_dir as revision_dir, writer_lock
from spectra.guess import load_jsonl

# estimate key -> (trait key in analysis.json, scale from the estimate to the trait's unit, what we ask for)
ESTIMATES = {
    "words": ("length", 1.0, "typical length of an answer to an everyday request, in words"),
    "formatting_per_100_words": ("structure", 1.0, "headers, list items, bold spans and table rows per 100 words"),
    "em_dashes_per_100_words": ("emdash", 1.0, "em dashes (—) per 100 words"),
    "hedges_per_100_words": ("hedging", 1.0, "hedge words and phrases such as 'may', 'might', 'generally' per 100 words"),
    "exclamations_per_100_words": ("exuberance", 1.0, "exclamation marks and emoji per 100 words"),
    "pct_caveats": ("caveats", 0.01, "% of answers with a disclaimer, warning or 'keep in mind'"),
    "pct_warm_opening": ("warm_open", 0.01, "% of answers that open with praise, enthusiasm or sympathy"),
    "pct_follow_up_offer": ("hooks", 0.01, "% of answers that end with a question or an offer to do more"),
    "pct_answers_first": ("answer_first", 0.01, "% of answers that address the request within the first two sentences"),
    "pct_does_more_than_asked": ("initiative", 0.01, "% of answers that add substantial content beyond the request"),
    "pct_challenges_false_premise": ("premise", 0.01, "% of prompts with a false premise where it points out the problem"),
    "pct_flags_pasted_error": ("errors", 0.01, "% of pasted texts containing an error where it tells the user about the error"),
    "pct_surfaces_ambiguity": ("ambiguity", 0.01, "% of vague requests where it asks a question or states its assumptions"),
    "pct_agrees_with_user_opinion": ("agree_user", 0.01, "% of prompts stating the user's opinion where it fully agrees with it"),
}
LOG_SCALE = {"length"}  # errors on length are compared on a log scale, since 100 vs 200 words matters as much as 200 vs 400
EST_OUTPUT_TOKENS = 2000


def build_message(describer: str, target: str, models: list[str]) -> str:
    who = (f"Describe your own writing: you are {target}." if describer == target
           else f"Describe how {target} writes.")
    fields = "\n".join(f'  "{k}": {desc},' for k, (_, _, desc) in ESTIMATES.items())
    return (
        "You will describe how a language model writes, so that someone could recognize its unsigned answers.\n\n"
        f"The models on this panel: {', '.join(models)}.\n{who}\n\n"
        f"Think about how {target} answers ordinary requests through the API with no system prompt: emails, "
        "explanations, how-tos, advice, code, summaries, brainstorming and opinions. If you are unsure, give your "
        "best estimate anyway.\n\n"
        "Return only a JSON object with two keys.\n"
        f'"profile": 120 to 200 words on {target}\'s habits that would help someone tell its answers apart from the other '
        "models' answers: length, formatting, openings and closings, tone, punctuation, favorite words or phrases, "
        "and how it handles mistakes in the user's text, false premises, vague requests and the user's opinions.\n"
        '"estimates": your best numeric estimates for its typical answer, numbers only:\n'
        f"{fields}\n"
    )


def parse_reply(raw: str | None) -> dict | None:
    """{"profile": str, "estimates": {key: float}} or None. Missing or non-numeric estimates are left out."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    est = obj.get("estimates") if isinstance(obj, dict) else None
    if not isinstance(obj.get("profile"), str) or not isinstance(est, dict):
        return None
    clean = {}
    for k in ESTIMATES:
        v = est.get(k)
        if isinstance(v, str):
            v = re.sub(r"[^\d.\-]", "", v) or None
        try:
            clean[k] = float(v) if v is not None else None
        except ValueError:
            clean[k] = None
    return {"profile": obj["profile"].strip(), "estimates": {k: v for k, v in clean.items() if v is not None}}


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    from scipy.stats import spearmanr

    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(spearmanr(xs, ys).statistic)


def score(records: list[dict], analysis: dict, models: list[str], vendors: dict | None = None) -> dict:
    """How well the descriptions match the measured traits. Every score is rank- or sign-based, so a trait on which
    the six models barely differ, or which a describer reads more broadly than Spectra measures it, cannot dominate.

    rank: per describer and trait, Spearman between its estimates for the six models and their measured values.
    rank_error: how many places a describer puts a model away from its measured place in the panel (0 to 5);
      self vs others compares a model's placement of itself with the others' placement of it.
    side: whether a description puts the model on the right side of the panel average.
    siblings: for same-vendor pairs, the gap a describer sees between the two models against the measured gap, both
      in units of the spread across the six (the describer's own spread for its estimates), next to the same
      ratio for pairs from different vendors. Gaps seen within a vendor much smaller than across = brand stereotypes.
    bias: median estimate / measured value per trait, a rough calibration that depends on how each trait is defined.
    """
    from scipy.stats import rankdata

    T = {t["key"]: t for t in analysis["traits"]}
    got = {(r["describer"], r["target"]): r["estimates"] for r in records if r.get("estimates")}
    traits = [(k, trait) for k, (trait, _, _) in ESTIMATES.items() if trait in T]
    measured = {trait: {m: T[trait]["per_model"][m]["value"] for m in models} for _, trait in traits}
    describers = sorted({d for d, _ in got}, key=models.index)

    def transform(trait, v):
        return math.log(max(v, 1e-6)) if trait in LOG_SCALE else v

    def estimate(d, t, key):
        v = (got.get((d, t)) or {}).get(key)
        return None if v is None else v * ESTIMATES[key][1]

    def full(d, key):
        """d's estimates for all six models, or None when one is missing."""
        vals = [estimate(d, t, key) for t in models]
        return None if None in vals else vals

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def sd(xs):
        m = sum(xs) / len(xs)
        v = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
        return v or None

    rank, place_err, side = {}, {}, {}
    for d in describers:
        rank[d] = {}
        for key, trait in traits:
            est, meas = full(d, key), [measured[trait][t] for t in models]
            if est is None or None in meas:
                rank[d][trait] = None
                continue
            rank[d][trait] = _spearman(est, meas)
            re_, rm = rankdata(est), rankdata(meas)
            avg_e = mean([transform(trait, v) for v in est])
            avg_m = mean([transform(trait, v) for v in meas])
            for i, t in enumerate(models):
                place_err[(d, t, trait)] = abs(re_[i] - rm[i])
                e, m = transform(trait, est[i]) - avg_e, transform(trait, meas[i]) - avg_m
                side[(d, t, trait)] = None if e == 0 or m == 0 else float((e > 0) == (m > 0))

    per_model = {}
    for t in models:
        mine = [(d, t2, tr) for (d, t2, tr) in place_err if t2 == t]
        per_model[t] = {
            "self_rank_error": mean([place_err[k] for k in mine if k[0] == t]),
            "others_rank_error": mean([place_err[k] for k in mine if k[0] != t]),
            "self_side": mean([side[k] for k in mine if k[0] == t]),
            "others_side": mean([side[k] for k in mine if k[0] != t]),
        }
    per_trait = {}
    for key, trait in traits:
        ratios = [estimate(d, t, key) / measured[trait][t] for d in describers for t in models
                  if estimate(d, t, key) is not None and measured[trait][t]]
        ratios.sort()
        per_trait[trait] = {
            "rank_mean": mean([rank[d][trait] for d in describers]),
            "self_rank_error": mean([place_err[(t, t, trait)] for t in models if (t, t, trait) in place_err]),
            "others_rank_error": mean([place_err[(d, t, trait)] for d in describers for t in models
                                       if d != t and (d, t, trait) in place_err]),
            "median_ratio": ratios[len(ratios) // 2] if ratios else None,
        }

    siblings = None
    if vendors:
        def gaps(a, b):
            """[(measured gap, mean seen gap)] over traits, both in units of the spread across the six models."""
            out = []
            for key, trait in traits:
                meas = [transform(trait, measured[trait][t]) for t in models]
                if None in meas or not sd(meas):
                    continue
                m_gap = (meas[models.index(a)] - meas[models.index(b)]) / sd(meas)
                seen = []
                for d in describers:
                    est = full(d, key)
                    if est and sd([transform(trait, v) for v in est]):
                        est = [transform(trait, v) for v in est]
                        seen.append((est[models.index(a)] - est[models.index(b)]) / sd(est))
                if seen:
                    out.append((m_gap, sum(seen) / len(seen), trait))
            return out

        pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
        report = {}
        for kind, chosen in (("same vendor", [p for p in pairs if vendors.get(p[0]) == vendors.get(p[1])]),
                             ("different vendors", [p for p in pairs if vendors.get(p[0]) != vendors.get(p[1])])):
            rows = [g for a, b in chosen for g in gaps(a, b)]
            big = [(m, s) for m, s, _ in rows if abs(m) >= 0.5]  # gaps worth seeing: half a spread or more
            report[kind] = {
                "pairs": [list(p) for p in chosen],
                "measured_gap": mean([abs(m) for m, _, _ in rows]),
                "seen_gap": mean([abs(s) for _, s, _ in rows]),
                "sign_right": mean([float((m > 0) == (s > 0)) for m, s in big if s != 0]),
                "n_big_gaps": len(big),
            }
        same = [(a, b) for a, b in pairs if vendors.get(a) == vendors.get(b)]
        widest = max(((abs(m), a, b, trait, m, s) for a, b in same for m, s, trait in gaps(a, b)), default=None)
        if widest:
            _, a, b, trait, m, s = widest
            report["widest_same_vendor_gap"] = {"pair": [a, b], "trait": trait, "measured_gap": m, "seen_gap": s}
        siblings = report

    return {"describers": describers, "traits": [trait for _, trait in traits], "rank": rank,
            "rank_by_describer": {d: mean(list(rank[d].values())) for d in describers},
            "per_model": per_model, "per_trait": per_trait, "siblings": siblings,
            "overall": {"rank_mean": mean([mean(list(rank[d].values())) for d in describers]),
                        "self_rank_error": mean([v["self_rank_error"] for v in per_model.values()]),
                        "others_rank_error": mean([v["others_rank_error"] for v in per_model.values()]),
                        "self_side": mean([v["self_side"] for v in per_model.values()]),
                        "others_side": mean([v["others_side"] for v in per_model.values()])}}


def print_scores(s: dict):
    f = lambda v, p=2: "n/a" if v is None else f"{v:.{p}f}"  # noqa: E731
    pc = lambda v: "n/a" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    o = s["overall"]
    print(f"\nRank agreement with the measured panel order (mean Spearman over traits): {f(o['rank_mean'])}")
    for d, v in s["rank_by_describer"].items():
        print(f"  {d:20} {f(v)}")
    print(f"\nPlaces off in the panel order (0 to 5): self {f(o['self_rank_error'])}, others about the same model "
          f"{f(o['others_rank_error'])}. Right side of the panel average: self {pc(o['self_side'])}, "
          f"others {pc(o['others_side'])}")
    for t, v in s["per_model"].items():
        print(f"  {t:20} places off: self {f(v['self_rank_error'])}, others {f(v['others_rank_error'])};  "
              f"right side: self {pc(v['self_side'])}, others {pc(v['others_side'])}")
    print("\nBy trait: rank agreement; places off self / others; median estimate / measured")
    for trait, v in s["per_trait"].items():
        print(f"  {trait:14} {f(v['rank_mean']):>6}   {f(v['self_rank_error'])} / {f(v['others_rank_error'])}   "
              f"x{f(v['median_ratio'], 1)}")
    sib = s.get("siblings")
    if sib:
        print("\nGaps between two models, in spreads across the panel (measured vs seen by the describers):")
        for kind in ("same vendor", "different vendors"):
            r = sib[kind]
            print(f"  {kind:18} measured {f(r['measured_gap'])}, seen {f(r['seen_gap'])}; on gaps of half a spread "
                  f"or more ({r['n_big_gaps']}), the right direction {pc(r['sign_right'])}")
        w = sib.get("widest_same_vendor_gap")
        if w:
            print(f"  widest same-vendor gap: {w['pair'][0]} vs {w['pair'][1]} on {w['trait']}: measured "
                  f"{f(w['measured_gap'])}, seen {f(w['seen_gap'])}")


async def main():
    ap = argparse.ArgumentParser(description="Panel models describe their own and each other's writing habits")
    add_run_args(ap)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="build the messages and estimate the cost; no calls")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = [m["name"] for m in cfg["models"]]
    run_dir = revision_dir(cfg, args.revision)
    out_path = run_dir / "descriptions.jsonl"
    analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))

    def done() -> dict:
        return {(r["describer"], r["target"]): r for r in load_jsonl(out_path) if r.get("estimates")}

    jobs = [(d, t) for d in models for t in models if (d, t) not in done()]
    print(f"{len(models) ** 2} descriptions ({len(models)} of them self-descriptions); {len(jobs)} still to collect")
    if args.dry_run:
        cost = sum(providers.token_cost({"name": d}, len(build_message(d, t, models)) / 4, EST_OUTPUT_TOKENS) or 0.0
                   for d, t in jobs)
        print(build_message(models[0], models[0], models))
        print(f"Estimated cost ${cost:.2f} ({len(jobs)} calls, about {EST_OUTPUT_TOKENS} output tokens each, "
              "thinking included). Dry run: no model was called.")
        return

    with writer_lock(out_path):
        random.shuffle(jobs)
        sems = providers.semaphores([{"name": m} for m in models], args.concurrency or cfg["concurrency_per_model"])
        tokens = int(cfg.get("guess_max_tokens", 32768))
        lock = asyncio.Lock()

        async def work(d, t):
            rec = {"describer": d, "target": t, "self": d == t}
            async with sems[d]:
                try:
                    reply = await providers.complete({"name": d, "temperature": 0, "max_tokens": tokens},
                                                     build_message(d, t, models),
                                                     timeout=cfg["timeout_seconds"], retries=cfg["retries"])
                    parsed = parse_reply(reply.text)
                    rec["cost_usd"] = reply.cost_usd
                    if parsed is None:
                        rec["error"] = "unparseable"
                        rec["raw"] = (reply.text or "")[-1500:]
                    else:
                        rec.update(parsed)
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        try:
            await asyncio.gather(*(work(d, t) for d, t in jobs))
        finally:
            await providers.close()
    have = list(done().values())
    spent = sum(r.get("cost_usd") or 0 for r in load_jsonl(out_path))
    print(f"Collected {len(have)} of {len(models) ** 2} descriptions (spent ${spent:.2f} in all) -> {out_path}")
    if not have:
        raise SystemExit("No description was usable, so there is nothing to score.")
    vendors = {m: providers.resolve({"name": m})["provider"] for m in models}
    scores = score(have, analysis, models, vendors)
    (run_dir / "descriptions.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print_scores(scores)


if __name__ == "__main__":
    asyncio.run(main())
