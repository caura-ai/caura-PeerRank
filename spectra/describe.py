#!/usr/bin/env python3
"""Do models know how they write? Each panel model describes its own habits and every other model's.

  python -m spectra.describe --revision oct-23-26 --dry-run   # messages and a cost estimate, no calls
  python -m spectra.describe --revision oct-23-26             # one call per (describer, target)
  python -m spectra.describe --revision oct-23-26 --sample 1  # a fresh set, panel and estimates in shuffled order
  python -m spectra.describe --revision oct-23-26 --measured oct-14-26   # rescore against another run, no calls

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

from spectra import DATA_ROOT, add_run_args, load_config, providers, run_dir as revision_dir, writer_lock
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


def build_message(describer: str, target: str, models: list[str], sample: int = 0) -> str:
    """Sample 0 is the original message. A later sample lists the panel and the estimates in an order shuffled for
    (sample, describer, target), so a replication makes fresh calls without changing a word of the request."""
    who = (f"Describe your own writing: you are {target}." if describer == target
           else f"Describe how {target} writes.")
    roster, keys = list(models), list(ESTIMATES)
    if sample:
        rng = random.Random(f"{sample}|{describer}|{target}")
        rng.shuffle(roster)
        rng.shuffle(keys)
    fields = "\n".join(f'  "{k}": {ESTIMATES[k][2]},' for k in keys)
    return (
        "You will describe how a language model writes, so that someone could recognize its unsigned answers.\n\n"
        f"The models on this panel: {', '.join(roster)}.\n{who}\n\n"
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


# The pre-registered tests (spectra/prereg/2026-10-25-self-knowledge-and-profiles.md), fixed before the replication.
STYLE = ("length", "structure", "emdash", "hedging", "exuberance")
BEHAVIOR = ("premise", "errors", "ambiguity", "agree_user")
OVERESTIMATED = ("hedging", "caveats", "warm_open", "hooks")
UNDERESTIMATED = ("premise", "errors", "ambiguity")
EQUIVALENCE = 0.25  # places off: a self-advantage smaller than this counts as none (sample 0 beat chance by 0.42)
SIBLINGS = ("claude-fable-5.1", "claude-opus-5.5", "emdash")


def _estimate_ranks(records: list[dict], analysis: dict, models: list[str]) -> dict:
    """trait -> (measured values, {describer: its six estimates}) over describers that estimated all six."""
    T = {t["key"]: t for t in analysis["traits"]}
    got = {(r["describer"], r["target"]): r["estimates"] for r in records if r.get("estimates")}
    out = {}
    for key, (trait, scale, _) in ESTIMATES.items():
        if trait not in T:
            continue
        meas = [T[trait]["per_model"][m]["value"] for m in models]
        if None in meas:
            continue
        rows = {}
        for d in models:
            est = [(got.get((d, t)) or {}).get(key) for t in models]
            if None not in est:
                rows[d] = [v * scale for v in est]
        out[trait] = (meas, rows)
    return out


def confirm(records: list[dict], analysis: dict, models: list[str], reps: int = 10000, seed: int = 7) -> dict:
    """The five pre-registered tests on one set of descriptions.

    H1 rank agreement above chance: the mean Spearman of score() against a null that permutes each trait's measured
       values across the six models (the same permutation for every describer), one-sided, p < 0.05.
    H2 no self-advantage: self minus others' places off, per model and trait; 95% interval resampling traits must lie
       within +-EQUIVALENCE (Wilcoxon over the pairs reported alongside).
    H3 behavior known better than style: mean rank agreement on BEHAVIOR minus STYLE, exact permutation of the nine
       traits' group labels, one-sided, p < 0.05.
    H4 the widest same-vendor gap is missed: the describers' mean gap between SIBLINGS on that trait is under half the
       measured gap, both in spreads across the six.
    H5 cliches overestimated and trap-catching underestimated: median estimate / measured above 1 on every
       OVERESTIMATED trait and below 1 on every UNDERESTIMATED trait.
    """
    import numpy as np
    from itertools import combinations
    from scipy.stats import rankdata, wilcoxon

    s = score(records, analysis, models)
    rng = np.random.default_rng(seed)
    cells = _estimate_ranks(records, analysis, models)
    out = {}

    # H1. Spearman = Pearson of ranks: z-score the ranks once, then each permutation is one matrix product.
    def z(v):
        r = rankdata(v)
        return (r - r.mean()) / r.std()

    sums = {d: np.zeros(reps + 1) for d in models}
    counts = {d: 0 for d in models}
    for trait, (meas, rows) in cells.items():
        if len(set(meas)) < 2:
            continue
        zm = z(meas)
        perms = np.vstack([np.arange(len(models)), rng.permuted(np.tile(np.arange(len(models)), (reps, 1)), axis=1)])
        for d, est in rows.items():
            if len(set(est)) < 2:
                continue
            sums[d] += (zm[perms] * z(est)).mean(axis=1)
            counts[d] += 1
    used = [d for d in models if counts[d]]
    stat = np.mean([sums[d] / counts[d] for d in used], axis=0)
    out["H1"] = {"rank_mean": float(stat[0]), "null_mean": float(stat[1:].mean()),
                 "p": float((1 + (stat[1:] >= stat[0] - 1e-12).sum()) / (1 + reps))}
    out["H1"]["confirmed"] = out["H1"]["p"] < 0.05

    # H2
    pairs = []  # (trait, self minus mean of others)
    for trait, (meas, rows) in cells.items():
        rm = rankdata(meas)
        err = {d: np.abs(rankdata(est) - rm) for d, est in rows.items()}
        for i, t in enumerate(models):
            if t in err and len(err) > 1:
                pairs.append((trait, float(err[t][i] - np.mean([e[i] for d, e in err.items() if d != t]))))
    traits = sorted({t for t, _ in pairs})
    by = {t: [x for tr, x in pairs if tr == t] for t in traits}
    boots = [np.mean([x for j in rng.integers(0, len(traits), len(traits)) for x in by[traits[j]]]) for _ in range(reps)]
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    diffs = [x for _, x in pairs]
    out["H2"] = {"self_minus_others": float(np.mean(diffs)), "ci": [lo, hi], "n_pairs": len(diffs),
                 "wilcoxon_p": float(wilcoxon(diffs).pvalue) if any(diffs) else 1.0,
                 "confirmed": -EQUIVALENCE < lo and hi < EQUIVALENCE,
                 "self_advantage": hi < 0}

    # H3
    r = {t: s["per_trait"][t]["rank_mean"] for t in STYLE + BEHAVIOR if t in s["per_trait"]}
    if all(t in r and r[t] is not None for t in STYLE + BEHAVIOR):
        vals = [r[t] for t in STYLE + BEHAVIOR]
        obs = np.mean([r[t] for t in BEHAVIOR]) - np.mean([r[t] for t in STYLE])
        null = [np.mean([vals[i] for i in c]) - np.mean([vals[i] for i in range(len(vals)) if i not in c])
                for c in combinations(range(len(vals)), len(BEHAVIOR))]
        p = float(np.mean([n >= obs - 1e-12 for n in null]))
        out["H3"] = {"behavior": float(np.mean([r[t] for t in BEHAVIOR])), "style": float(np.mean([r[t] for t in STYLE])),
                     "p": p, "splits": len(null), "confirmed": p < 0.05}

    # H4
    a, b, trait = SIBLINGS
    if trait in cells and a in models and b in models:
        meas, rows = cells[trait]
        ia, ib = models.index(a), models.index(b)
        m_gap = (meas[ia] - meas[ib]) / np.std(meas, ddof=1)
        seen = [(e[ia] - e[ib]) / np.std(e, ddof=1) for e in rows.values() if np.std(e) > 0]
        out["H4"] = {"pair": [a, b], "trait": trait, "measured_gap": float(m_gap),
                     "seen_gap": float(np.mean(seen)) if seen else None,
                     "confirmed": bool(seen and m_gap != 0 and float(np.mean(seen)) / m_gap < 0.5)}

    # H5
    ratio = {t: s["per_trait"][t]["median_ratio"] for t in OVERESTIMATED + UNDERESTIMATED if t in s["per_trait"]}
    out["H5"] = {"median_ratio": ratio,
                 "confirmed": all((ratio.get(t) or 0) > 1 for t in OVERESTIMATED)
                 and all(ratio.get(t) is not None and ratio[t] < 1 for t in UNDERESTIMATED)}
    return out


def stability(first: list[dict], second: list[dict], models: list[str]) -> dict:
    """How much two samples of descriptions agree: per describer, target set and estimate, Spearman across the six."""
    a = {(r["describer"], r["target"]): r["estimates"] for r in first if r.get("estimates")}
    b = {(r["describer"], r["target"]): r["estimates"] for r in second if r.get("estimates")}
    rhos = {}
    for key in ESTIMATES:
        vals = []
        for d in models:
            x = [(a.get((d, t)) or {}).get(key) for t in models]
            y = [(b.get((d, t)) or {}).get(key) for t in models]
            if None not in x and None not in y:
                rho = _spearman(x, y)
                if rho is not None:
                    vals.append(rho)
        rhos[ESTIMATES[key][0]] = sum(vals) / len(vals) if vals else None
    got = [v for v in rhos.values() if v is not None]
    return {"by_trait": rhos, "mean": sum(got) / len(got) if got else None}


def print_tests(tests: dict, stable: dict | None = None):
    yes = lambda h: "confirmed" if tests.get(h, {}).get("confirmed") else "NOT confirmed"  # noqa: E731
    if "H1" in tests:
        h = tests["H1"]
        print(f"\nH1 rank agreement above chance: {h['rank_mean']:.2f} vs {h['null_mean']:.2f} by chance, "
              f"p = {h['p']:.4f} -> {yes('H1')}")
    if "H2" in tests:
        h = tests["H2"]
        print(f"H2 no self-advantage: self minus others {h['self_minus_others']:+.2f} places "
              f"(95% {h['ci'][0]:+.2f} to {h['ci'][1]:+.2f}, bound +-{EQUIVALENCE}; Wilcoxon p = {h['wilcoxon_p']:.2f}) "
              f"-> {yes('H2')}")
    if "H3" in tests:
        h = tests["H3"]
        print(f"H3 behavior known better than style: {h['behavior']:.2f} vs {h['style']:.2f}, "
              f"p = {h['p']:.3f} over {h['splits']} splits -> {yes('H3')}")
    if "H4" in tests:
        h = tests["H4"]
        print(f"H4 {h['pair'][0]} vs {h['pair'][1]} on {h['trait']}: measured {h['measured_gap']:+.2f}, seen "
              f"{h['seen_gap']:+.2f} spreads -> {yes('H4')}")
    if "H5" in tests:
        h = tests["H5"]
        print("H5 median estimate / measured: " + ", ".join(f"{t} x{v:.1f}" for t, v in h["median_ratio"].items()
                                                          if v is not None) + f" -> {yes('H5')}")
    if stable and stable.get("mean") is not None:
        print(f"Agreement with sample 0 (mean Spearman over describers and traits): {stable['mean']:.2f}")


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
    ap.add_argument("--sample", type=int, default=0,
                    help="0 = the original descriptions; N > 0 collects a fresh set with the panel and the estimates "
                         "listed in a shuffled order (a replication)")
    ap.add_argument("--measured", default=None,
                    help="score against this revision's analysis.json instead of --revision's")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = [m["name"] for m in cfg["models"]]
    run_dir = revision_dir(cfg, args.revision)
    out_path = run_dir / "descriptions.jsonl"
    measured = args.measured or cfg["revision"]
    analysis = json.loads((DATA_ROOT / measured / "analysis.json").read_text(encoding="utf-8"))
    scores_path = run_dir / ("descriptions" + (f"-s{args.sample}" if args.sample else "")
                             + (f"-vs-{measured}" if measured != cfg["revision"] else "") + ".json")

    def done(sample: int = args.sample) -> dict:
        return {(r["describer"], r["target"]): r for r in load_jsonl(out_path)
                if r.get("estimates") and int(r.get("sample", 0)) == sample}

    jobs = [(d, t) for d in models for t in models if (d, t) not in done()]
    print(f"{len(models) ** 2} descriptions ({len(models)} of them self-descriptions) in sample {args.sample}; "
          f"{len(jobs)} still to collect")
    if args.dry_run:
        cost = sum(providers.token_cost({"name": d}, len(build_message(d, t, models, args.sample)) / 4,
                                        EST_OUTPUT_TOKENS) or 0.0
                   for d, t in jobs)
        print(build_message(models[0], models[0], models, args.sample))
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
            if args.sample:
                rec["sample"] = args.sample
            async with sems[d]:
                try:
                    reply = await providers.complete({"name": d, "temperature": 0, "max_tokens": tokens},
                                                     build_message(d, t, models, args.sample),
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
    spent = sum(r.get("cost_usd") or 0 for r in load_jsonl(out_path) if int(r.get("sample", 0)) == args.sample)
    print(f"Collected {len(have)} of {len(models) ** 2} descriptions in sample {args.sample} "
          f"(spent ${spent:.2f} on it) -> {out_path}")
    if not have:
        raise SystemExit("No description was usable, so there is nothing to score.")
    vendors = {m: providers.resolve({"name": m})["provider"] for m in models}
    scores = score(have, analysis, models, vendors)
    scores["measured"] = measured
    scores["tests"] = confirm(have, analysis, models)
    first = list(done(0).values()) if args.sample else []
    scores["stability_vs_sample_0"] = stability(first, have, models) if first else None
    scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"Scored against {measured} -> {scores_path}")
    print_scores(scores)
    print_tests(scores["tests"], scores["stability_vs_sample_0"])


if __name__ == "__main__":
    asyncio.run(main())
