#!/usr/bin/env python3
"""Three answers per prompt: how big the accent really is, how consistent each model is, and the imitation baseline.

  python -m spectra.replicates --config spectra/config-s3.yaml --revision oct-23-26-s3

With one answer per prompt, a model's steady reaction to a particular prompt and plain sampling noise cannot be told
apart, so both count against the accent. With three, each style trait splits into four parts (two-way ANOVA with
replication): the model, the prompt, the model's reaction to that prompt (model x prompt), and noise. Also:
- each model's self-consistency, the share of its own variation that lies between prompts rather than between
  repeated answers to the same prompt;
- the imitation test (E3) rescored against the mean of three answers, instead of the single earlier answer it drifted
  from.
Writes data/spectra/<revision>/replicates.json. No API calls. Fixed in spectra/prereg/2026-10-25-replicates.md.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from spectra import DATA_ROOT, add_run_args, load_config, run_dir as revision_dir

PARTS = ("model", "prompt", "model_x_prompt", "noise")


def decompose(y: np.ndarray) -> dict:
    """y: models x prompts x samples, complete. Each part's share of the total sum of squares, and the F test of
    model x prompt against noise (needs two samples or more)."""
    from scipy.stats import f as fdist

    M, P, S = y.shape
    g = y.mean()
    ym, yp, ymp = y.mean(axis=(1, 2)), y.mean(axis=(0, 2)), y.mean(axis=2)
    ss = {"model": P * S * ((ym - g) ** 2).sum(), "prompt": M * S * ((yp - g) ** 2).sum(),
          "model_x_prompt": S * ((ymp - ym[:, None] - yp[None, :] + g) ** 2).sum(),
          "noise": ((y - ymp[:, :, None]) ** 2).sum()}
    total = ((y - g) ** 2).sum()
    out = {k: float(v / total) if total > 0 else None for k, v in ss.items()}
    df_i, df_e = (M - 1) * (P - 1), M * P * (S - 1)
    F = float((ss["model_x_prompt"] / df_i) / (ss["noise"] / df_e)) if df_e and ss["noise"] > 0 else None
    return out | {"F": F, "p": float(fdist.sf(F, df_i, df_e)) if F is not None else None, "prompts": P}


def components(y: np.ndarray) -> dict:
    """Post hoc: each part's share of one answer's variance, from expected mean squares (random effects, negative
    estimates set to zero). Sum-of-squares shares hand 1/S of the noise to model x prompt; these do not."""
    M, P, S = y.shape
    d = decompose(y)
    total = ((y - y.mean()) ** 2).sum()
    ms = {"model": d["model"] * total / (M - 1), "prompt": d["prompt"] * total / (P - 1),
          "model_x_prompt": d["model_x_prompt"] * total / ((M - 1) * (P - 1)), "noise": d["noise"] * total / (M * P * (S - 1))}
    var = {"noise": ms["noise"], "model_x_prompt": max(0.0, (ms["model_x_prompt"] - ms["noise"]) / S),
           "model": max(0.0, (ms["model"] - ms["model_x_prompt"]) / (P * S)),
           "prompt": max(0.0, (ms["prompt"] - ms["model_x_prompt"]) / (M * S))}
    whole = sum(var.values())
    return {k: float(var[k] / whole) for k in PARTS}


def components_ci(arrays: dict, prompt_ids: dict, reps: int = 2000, seed: int = 7) -> dict:
    """Post hoc: 95% intervals for each trait's variance components and for their medians over traits, resampling
    prompts (one draw of prompt ids per replicate, shared by every trait, as the traits are measured on the same
    answers). arrays: {trait: models x prompts x samples}; prompt_ids: {trait: the prompt id of each column}."""
    rng = np.random.default_rng(seed)
    pool = sorted({p for ids in prompt_ids.values() for p in ids})
    where = {k: {p: j for j, p in enumerate(ids)} for k, ids in prompt_ids.items()}
    per = {k: {p: [] for p in PARTS} for k in arrays}
    meds = {p: [] for p in PARTS}
    for _ in range(reps):
        draw = [pool[j] for j in rng.integers(0, len(pool), len(pool))]
        got = {}
        for k, y in arrays.items():
            cols = [where[k][p] for p in draw if p in where[k]]
            c = components(y[:, cols, :])
            got[k] = c
            for p in PARTS:
                per[k][p].append(c[p])
        for p in PARTS:
            meds[p].append(float(np.median([got[k][p] for k in got])))
    band = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"traits": {k: {p: band(v) for p, v in d.items()} for k, d in per.items()},
            "median": {p: band(v) for p, v in meds.items()}, "reps": reps}


def consistency(y: np.ndarray) -> float | None:
    """One model's prompts x samples: the share of its variance that lies between prompts (1 = fully repeatable)."""
    total = y.var()
    return None if total == 0 else float(1 - y.var(axis=1).mean() / total)


def answer_vectors(run_dir: Path, prompts: dict, models: list[str], samples: int) -> dict:
    """{(model, prompt): samples x traits}, raw style traits, for cells with every sample answered."""
    from spectra.voice import vector

    texts = {}
    for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("text") and not r.get("error") and r["model"] in models and r["prompt_id"] in prompts:
            texts[(r["model"], r["prompt_id"], int(r.get("sample", 0)))] = r["text"]
    out = {}
    for m in models:
        for pid, p in prompts.items():
            got = [texts.get((m, pid, s)) for s in range(samples)]
            if None not in got:
                out[(m, pid)] = np.array([vector(t, p["category"]) for t in got])
    return out


def imitation_recheck(rows: list[dict], vecs: dict, base) -> dict:
    """E3 against three-answer baselines, in the units E3 used (base: the Baseline of the first answers).

    For each NAME imitation, the axis runs from the imitator's answer to the target's on that prompt: the single first
    answers, as E3 scored it, or the mean of three. SELF drift is how far the same model's fresh SELF answer sits along
    that axis; if it was regression toward the mean, it shrinks with the averaged axis."""
    from spectra.voice import project, prompt_ci

    selfv = {(r["writer"], r["prompt_id"]): base.z(r["text"], r["category"]) for r in rows if r["condition"] == "SELF"}
    single, averaged, shrink, name_vs_self = {}, {}, {}, {}
    for r in rows:
        m, t, pid = r["writer"], r["other"], r["prompt_id"]
        if r["condition"] != "NAME" or (m, pid) not in vecs or (t, pid) not in vecs or (m, pid) not in selfv:
            continue
        with warnings.catch_warnings():  # a trait a short answer lacks is NaN in every sample
            warnings.simplefilter("ignore", RuntimeWarning)
            a3, b3 = np.nanmean(vecs[(m, pid)], axis=0) / base.sd, np.nanmean(vecs[(t, pid)], axis=0) / base.sd
        s1 = project(selfv[(m, pid)], vecs[(m, pid)][0] / base.sd, vecs[(t, pid)][0] / base.sd)
        s3, n3 = project(selfv[(m, pid)], a3, b3), project(base.z(r["text"], r["category"]), a3, b3)
        if s1 is not None and s3 is not None:
            single.setdefault(pid, []).append(s1)
            averaged.setdefault(pid, []).append(s3)
            shrink.setdefault(pid, []).append(s1 - s3)
        if n3 is not None and s3 is not None:
            name_vs_self.setdefault(pid, []).append(n3 - s3)
    return {"self_drift_single": prompt_ci(single), "self_drift_averaged": prompt_ci(averaged),
            "drift_shrink": prompt_ci(shrink), "name_minus_self_averaged": prompt_ci(name_vs_self)}


def analyse(run_dir: Path, prompts: dict, models: list[str], samples: int, imitation_rows=None, reps: int = 2000) -> dict:
    from scipy.stats import rankdata

    from spectra.voice import Baseline, everyday, load_texts, trait_columns

    keys = [k for k, _ in trait_columns()]
    vecs = answer_vectors(run_dir, prompts, models, samples)
    plain = {p["id"] for p in everyday(list(prompts.values()))}
    traits, arrays, columns = {}, {}, {}
    for i, k in enumerate(keys):
        cells = [pid for pid in prompts if all((m, pid) in vecs and np.isfinite(vecs[(m, pid)][:, i]).all() for m in models)]
        if len(cells) < 10:
            continue
        y = np.array([[vecs[(m, pid)][:, i] for pid in cells] for m in models])
        if y.var() == 0:
            continue
        d = decompose(y)
        d["one_answer"] = {p: decompose(y[:, :, :1])[p] for p in ("model", "prompt")}  # the same cells, first answers
        if samples >= 3:  # the first answers came from an earlier day: the later answers alone, without that gap
            later = decompose(y[:, :, 1:])
            d["later_answers"] = {p: later[p] for p in PARTS + ("p",)}
        d["consistency"] = {m: consistency(y[j]) for j, m in enumerate(models)}
        d["components"] = components(y)
        # post-hoc robustness: the same-day answers alone, everyday prompts alone, and ranks instead of raw values
        rob = {"ranks": components(rankdata(y, axis=None).reshape(y.shape))}
        if samples >= 3:
            rob["later_answers"] = components(y[:, :, 1:])
        keep = [j for j, pid in enumerate(cells) if pid in plain]
        if len(keep) >= 10:
            rob["everyday_only"] = components(y[:, keep, :])
        d["robustness"] = rob
        traits[k], arrays[k], columns[k] = d, y, cells
    if not traits:
        raise SystemExit(f"No trait has {samples} answers per model on 10 prompts or more in {run_dir}")
    med = lambda f: float(np.median([f(t) for t in traits.values()]))  # noqa: E731
    out = {"samples": samples, "n_traits": len(traits), "traits": traits,
           "median_shares": {p: med(lambda t: t[p]) for p in PARTS},
           "median_components": {p: med(lambda t: t["components"][p]) for p in PARTS},
           "median_one_answer": {p: med(lambda t: t["one_answer"][p]) for p in ("model", "prompt")},
           "median_model_with_reactions": med(lambda t: t["model"] + t["model_x_prompt"]),
           "median_reaction_share_of_rest": med(lambda t: t["model_x_prompt"] / (t["model_x_prompt"] + t["noise"])),
           "interaction_significant": sum(1 for t in traits.values() if t["p"] is not None and t["p"] < 0.05),
           "interaction_significant_later": sum(1 for t in traits.values() if (t.get("later_answers") or {}).get("p") is not None
                                                and t["later_answers"]["p"] < 0.05),
           "median_noise_later": med(lambda t: t["later_answers"]["noise"]) if samples >= 3 else None,
           "consistency": {m: float(np.mean([t["consistency"][m] for t in traits.values()
                                             if t["consistency"][m] is not None])) for m in models}}
    if samples >= 2:
        out["components_ci"] = components_ci(arrays, columns, reps)
        out["median_robustness"] = {
            check: {p: float(np.median([t["robustness"][check][p] for t in traits.values() if check in t["robustness"]]))
                    for p in PARTS}
            for check in ("later_answers", "everyday_only", "ranks") if any(check in t["robustness"] for t in traits.values())}
    if imitation_rows:
        base = Baseline(load_texts(run_dir), prompts, models)
        out["imitation"] = imitation_recheck(imitation_rows, vecs, base)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Three answers per prompt: the accent's true size and self-consistency")
    add_run_args(ap)
    ap.add_argument("--imitation-from", default="oct-23-26", help="revision whose voice.jsonl holds the E3 records")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = revision_dir(cfg, args.revision)
    models = [m["name"] for m in cfg["models"]]
    prompts = {p["id"]: p for p in json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))["prompts"]}
    from spectra.voice import latest

    rows = [r for r in latest(DATA_ROOT / args.imitation_from / "voice.jsonl").values() if r["experiment"] == "imitate"]
    res = analyse(run_dir, prompts, models, int(cfg["samples"]), rows)
    (run_dir / "replicates.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pc = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    sh, one = res["median_shares"], res["median_one_answer"]
    print(f"Median shares over {res['n_traits']} style traits, {res['samples']} answers per prompt: "
          + ", ".join(f"{p} {pc(sh[p])}" for p in PARTS))
    print(f"  the same cells from first answers alone: model {pc(one['model'])}, prompt {pc(one['prompt'])}")
    print("  post hoc, shares of one answer's variance (expected mean squares): "
          + ", ".join(f"{p} {pc(res['median_components'][p])}" for p in PARTS))
    print(f"  model plus its reactions {pc(res['median_model_with_reactions'])}; reactions are "
          f"{pc(res['median_reaction_share_of_rest'])} of what is neither model nor prompt; model x prompt significant "
          f"on {res['interaction_significant']} of {res['n_traits']} traits")
    if res["median_noise_later"] is not None:
        print(f"  later answers alone (same day): noise {pc(res['median_noise_later'])}, model x prompt significant on "
              f"{res['interaction_significant_later']} of {res['n_traits']} traits")
    print("Self-consistency: " + ", ".join(f"{m} {v:.2f}" for m, v in sorted(res["consistency"].items(), key=lambda x: -x[1])))
    if "imitation" in res:
        im = res["imitation"]
        f = lambda s: f"{pc(s['mean'])} ({pc(s['ci'][0])} to {pc(s['ci'][1])})" if s["ci"] else "n/a"  # noqa: E731
        print(f"SELF drift toward the target: single baseline {f(im['self_drift_single'])}, averaged "
              f"{f(im['self_drift_averaged'])}, shrink {f(im['drift_shrink'])}; NAME minus SELF, averaged "
              f"{f(im['name_minus_self_averaged'])}")
    print(f"-> {run_dir / 'replicates.json'}")


if __name__ == "__main__":
    main()
