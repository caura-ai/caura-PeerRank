#!/usr/bin/env python3
"""Score the eight-model generalization test (spectra/prereg/2026-10-25-panel-8.md): G1-G5 and the exploratory items.

  python -m spectra.generalize

Reads the two eight-model revisions (oct-23-26-p8, v2-oct-25-26-p8) after analyze, describe and voice score have run,
and the six-model oct-23-26 for comparison. Writes data/spectra/oct-23-26-p8/panel8.json. No API calls.
"""
import collections
import json

import numpy as np
from scipy.stats import spearmanr

from spectra import DATA_ROOT, SPECTRA_DIR, load_config
from spectra.analyze import cohen_kappa
from spectra.guess import load_jsonl

V1, V2, SIX = "oct-23-26-p8", "v2-oct-25-26-p8", "oct-23-26"
NEW = ("grok-4.7", "kimi-k3")
TRAP_OF = {"premise": "FP", "error_handling": "ERR", "user_opinion": "OP", "emotional_ack": "EMO", "clarification": "AMB"}


def tagged_kappa(codes_path, coders, tags) -> dict:
    """Cohen's kappa of two coders on each trap label, over the answers to prompts carrying that trap only."""
    codes = collections.defaultdict(dict)
    for c in load_jsonl(codes_path):
        if c.get("codes"):
            codes[(c["model"], c["prompt_id"], c.get("sample", 0))][c["coder"]] = c["codes"]
    both = [(pid, d) for (_, pid, _), d in codes.items() if all(k in d for k in coders)]
    out = {}
    for code, tag in TRAP_OF.items():
        pairs = [(d[coders[0]][code], d[coders[1]][code]) for pid, d in both if tag in tags[pid]]
        a, b = zip(*pairs)
        k = cohen_kappa(a, b) if len(set(a) | set(b)) > 1 else None
        out[code] = {"kappa": k, "n": len(pairs), "agree": sum(x == y for x, y in pairs) / len(pairs)}
    return out


def score() -> dict:
    from spectra.voice import ACCENT, Baseline, SOURCES, latest, load_texts, score_contagion

    d1, d2, d6 = DATA_ROOT / V1, DATA_ROOT / V2, DATA_ROOT / SIX
    A1, A2, A6 = (json.loads((d / "analysis.json").read_text(encoding="utf-8")) for d in (d1, d2, d6))
    assert A2["accent"]["previous"] == V1
    models = A1["meta"]["models"]
    out = {"models": models, "coders": A1["meta"]["coders"]}

    # data and cost (the coders' calls are not costed by the pipeline)
    for name, d in (("v1", d1), ("v2", d2)):
        resp = load_jsonl(d / "responses.jsonl")
        last = {}
        for c in load_jsonl(d / "codes.jsonl"):
            k = (c["coder"], c["model"], c["prompt_id"], c.get("sample", 0))
            last[k] = last.get(k) or bool(c.get("codes"))
        out[f"data_{name}"] = {
            "answers": len({(r["model"], r["prompt_id"], r.get("sample", 0)) for r in resp if r.get("text") and not r.get("error")}),
            "new_answer_cost": sum(r.get("cost_usd") or 0 for r in resp if r["model"] in NEW and r.get("text")),
            "labels": {f"{coder} {'ok' if good else 'failed'}": n
                       for (coder, good), n in collections.Counter((k[0], g) for k, g in last.items()).items()}}
    voice_rows = load_jsonl(d1 / "voice.jsonl")
    out["cost"] = {"describe": sum(r.get("cost_usd") or 0 for r in load_jsonl(d1 / "descriptions.jsonl")),
                   "contagion_new_readers": sum(r.get("cost_usd") or 0 for r in voice_rows
                                                if r["experiment"] == "contagion" and r["writer"] in NEW and r.get("text"))}

    # G1: the eight model means keep their order on v2, Spearman >= 0.8 on >= 3 of the 4 accent traits
    rows2 = {r["key"]: r for r in A2["accent"]["traits"]}
    g1 = {k: rows2[k]["run_to_run"] for k in ACCENT}
    passing = sum(v is not None and v >= 0.8 for v in g1.values())
    out["G1"] = {"spearman": g1, "passing": passing, "confirmed": passing >= 3}
    # G2: median model share on v1 between 1% and 10%
    m1 = A1["accent"]["median_model_share"]
    out["G2"] = {"median_model_share": m1, "median_prompt_share": A1["accent"]["median_prompt_share"],
                 "v2_median_model_share": A2["accent"]["median_model_share"], "confirmed": 0.01 <= m1 <= 0.10}
    # G3: each new model keeps at least one accent-card habit from v1 to v2, same trait and direction
    g3 = {}
    for m in NEW:
        c1 = {(c["key"], np.sign(c["d"])): c["text"] for c in A1["accent"]["cards"].get(m, [])}
        c2 = {(c["key"], np.sign(c["d"])): c["text"] for c in A2["accent"]["cards"].get(m, [])}
        g3[m] = {"v1": list(c1.values()), "v2": list(c2.values()), "shared": sorted(c1[k] for k in set(c1) & set(c2))}
    out["G3"] = g3 | {"confirmed": all(g3[m]["shared"] for m in NEW)}
    # G4: the 8 x 8 descriptions: H1 confirmed and H5 holds
    tests = json.loads((d1 / "descriptions-s1.json").read_text(encoding="utf-8"))["tests"]
    out["G4"] = {"tests": tests, "confirmed": bool(tests["H1"]["confirmed"]) and bool(tests["H5"]["confirmed"])}
    # G5: contagion with the new readers
    VO = json.loads((d1 / "voice.json").read_text(encoding="utf-8"))
    c1 = VO["E5_new_readers"]["C1_moves_toward_source"]
    out["G5"] = {"new_readers": VO["E5_new_readers"], "all_eight": VO["E5"]["C1_moves_toward_source"],
                 "confirmed": c1["ci"][0] > 0}

    # exploratory: contagion per reader and source, as a fraction of the way and in SD units along the axis
    cfg = load_config(SPECTRA_DIR / "config-8.yaml")
    prompts = {p["id"]: p for p in json.loads(open(cfg["prompts"], encoding="utf-8").read())["prompts"]}
    base = Baseline(load_texts(d1), prompts, models)
    rows = [r for r in latest(d1 / "voice.jsonl").values() if r["experiment"] == "contagion"]
    per = {}
    for m in models:
        for s in SOURCES:
            if s != m:
                sc = score_contagion([r for r in rows if r["writer"] == m and r["other"] in (m, s)], base, None)["C1_moves_toward_source"]
                axis = float(np.sqrt(np.nansum((base.mean[s] - base.mean[m]) ** 2)))
                per[f"{m} <- {s}"] = {"fraction": sc["mean"], "ci": sc["ci"], "axis_sd": axis, "shift_sd": sc["mean"] * axis}
    out["contagion_per_reader_source"] = per

    # exploratory: family tree
    dist = A1["accent"]["family"]["distance"]
    pairs = sorted((v, a, b) for a, r in dist.items() for b, v in r.items() if a < b)
    out["family"] = {"nearest": {m: sorted((v, o) for o, v in dist[m].items() if o != m)[:2] for m in NEW},
                     "grok_kimi": dist["grok-4.7"]["kimi-k3"], "closest_pair": pairs[0],
                     "median_pair": float(np.median([p[0] for p in pairs])),
                     "closest_pair_v2": min((v, a, b) for a, r in A2["accent"]["family"]["distance"].items()
                                            for b, v in r.items() if a < b)}
    # exploratory: the new coder pair's agreement, and the six models' coded traits under the new pair vs the old
    tags = {p["id"]: p["tags"] for p in prompts.values()}
    out["kappa_all_answers"] = {"new_pair": A1["agreement"], "old_pair": A6["agreement"]}
    out["kappa_tagged"] = {"new_pair": tagged_kappa(d1 / "codes.jsonl", A1["meta"]["coders"], tags),
                           "old_pair": tagged_kappa(d6 / "codes.jsonl", A6["meta"]["coders"], tags)}
    six = A6["meta"]["models"]
    T6 = {t["key"]: t for t in A6["traits"] if t.get("group") == "behavior"}
    T1 = {t["key"]: t for t in A1["traits"] if t.get("group") == "behavior"}
    coded = {}
    for k in T6.keys() & T1.keys():
        a, b = [T6[k]["per_model"][m]["value"] for m in six], [T1[k]["per_model"][m]["value"] for m in six]
        rho = spearmanr(a, b).statistic if len(set(a)) > 1 and len(set(b)) > 1 else None
        coded[k] = {"rho": None if rho is None or np.isnan(rho) else float(rho), "mean_old": float(np.mean(a)),
                    "mean_new": float(np.mean(b))}
    out["coded_six_old_vs_new"] = coded
    return out


def main():
    res = score()
    path = DATA_ROOT / V1 / "panel8.json"
    path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for g in ("G1", "G2", "G3", "G4", "G5"):
        print(g, "confirmed" if res[g]["confirmed"] else "NOT confirmed")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
