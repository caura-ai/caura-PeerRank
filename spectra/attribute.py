#!/usr/bin/env python3
"""Who wrote this answer? Author classifiers on the text alone, the reference the guessers are compared with.

  python -m spectra.attribute --revision oct-23-26                           # this run, grouped by prompt
  python -m spectra.attribute --revision oct-23-26 --train-revision oct-14-26   # also train on one run, test on another

Three feature sets, each a logistic regression with settings fixed in advance (none tuned on these runs):
  words+style            word 1-2-grams and the style features: analyze.py's classifier, reproduced
  characters             character 2-5-grams, the standard stylometry baseline (punctuation, markdown, spacing)
  characters+words+style all of them
Each is scored one answer at a time and jointly per prompt (each model once, as the guessing game names them), with
5-fold cross-validation grouped by prompt, on the game's held-out halves, and, with --train-revision, trained on
half A of one run and tested on half B of another, so neither the prompts nor the answers are shared.
Writes data/spectra/<revision>/attribution.json. No API calls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectra import DATA_ROOT, add_run_args, load_config, run_dir as revision_dir
from spectra.analyze import features, match_by_prompt, numeric_matrix

FEATURE_SETS = ("words+style", "characters", "characters+words+style")


def load_answers(run_dir: Path, prompts: dict) -> pd.DataFrame:
    """One row per (model, prompt, sample): the last successful answer, its prose and its style features."""
    rows = {}
    for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("text") and not r.get("error") and r.get("prompt_id") in prompts:
            rows[(r["model"], r["prompt_id"], r.get("sample", 0))] = r
    recs = []
    for (m, pid, s), r in rows.items():
        f, prose = features(r["text"], prompts[pid]["category"])
        f.update(model=m, prompt_id=pid, sample=s, category=prompts[pid]["category"], prose=prose)
        recs.append(f)
    return pd.DataFrame(recs).sort_values(["model", "prompt_id", "sample"]).reset_index(drop=True)


def _fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, y_train, k: int) -> np.ndarray:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    blocks_tr, blocks_te = [], []
    if "characters" in name:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=200000, sublinear_tf=True)
        blocks_tr.append(vec.fit_transform(train.prose))
        blocks_te.append(vec.transform(test.prose))
    if "words" in name:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
        blocks_tr.append(vec.fit_transform(train.prose))
        blocks_te.append(vec.transform(test.prose))
    if "style" in name:
        sc = StandardScaler().fit(numeric_matrix(train).to_numpy())
        blocks_tr.append(csr_matrix(sc.transform(numeric_matrix(train).to_numpy())))
        blocks_te.append(csr_matrix(sc.transform(numeric_matrix(test).to_numpy())))
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(hstack(blocks_tr).tocsr(), y_train)
    proba = np.zeros((len(test), k))
    proba[:, clf.classes_] = clf.predict_proba(hstack(blocks_te).tocsr())
    return proba


def _scores(proba: np.ndarray, df: pd.DataFrame, y: np.ndarray, k: int, rng) -> dict:
    """Accuracy one at a time and jointly per prompt, each with a 95% interval resampling prompts."""
    codes, prompt_idx = np.unique(df.prompt_id.to_numpy(), return_inverse=True)

    def ci(correct, reps=2000):
        hits = np.bincount(prompt_idx, weights=correct.astype(float), minlength=len(codes))
        n = np.bincount(prompt_idx, minlength=len(codes))
        draw = rng.integers(0, len(codes), size=(reps, len(codes)))
        boots = hits[draw].sum(axis=1) / n[draw].sum(axis=1)
        return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    out = {}
    for kind, pred in (("accuracy", proba.argmax(axis=1)), ("matched_accuracy", match_by_prompt(proba, df, k))):
        out[kind] = float((pred == y).mean())
        out[kind + "_ci"] = ci(pred == y)
    return out


def attribute(df: pd.DataFrame, models: list[str], seed: int = 7, sets=FEATURE_SETS) -> dict:
    """Every feature set, cross-validated by prompt and on the game's halves."""
    from sklearn.model_selection import GroupKFold
    from spectra.guess import holdout_halves

    k = len(models)
    df = df[df.model.isin(models)].reset_index(drop=True)
    y = df.model.map({m: i for i, m in enumerate(models)}).to_numpy()
    half = holdout_halves(df[["prompt_id", "category"]].drop_duplicates()
                          .rename(columns={"prompt_id": "id"}).to_dict("records"))
    in_a = df.prompt_id.map(half).eq("A").to_numpy()
    a, b = np.flatnonzero(in_a), np.flatnonzero(~in_a)
    folds = list(GroupKFold(n_splits=min(5, df.prompt_id.nunique())).split(df, y, df.prompt_id))
    rng = np.random.default_rng(seed)
    out = {}
    for name in sets:
        res = {}
        for split, pairs in (("cv", folds), ("halves", [(a, b), (b, a)])):
            proba = np.zeros((len(df), k))
            for tr, te in pairs:
                proba[te] = _fit_predict(name, df.iloc[tr], df.iloc[te], y[tr], k)
            res[split] = _scores(proba, df, y, k, rng)
        out[name] = res
    return out


def across(train: pd.DataFrame, test: pd.DataFrame, models: list[str], seed: int = 7, sets=FEATURE_SETS) -> dict:
    """Trained on half A of one run, tested on half B of another: no prompt and no answer in common."""
    from spectra.guess import holdout_halves

    k = len(models)
    cats = pd.concat([train, test])[["prompt_id", "category"]].drop_duplicates()
    half = holdout_halves(cats.rename(columns={"prompt_id": "id"}).to_dict("records"))
    tr = train[train.model.isin(models) & train.prompt_id.map(half).eq("A")].reset_index(drop=True)
    te = test[test.model.isin(models) & test.prompt_id.map(half).eq("B")].reset_index(drop=True)
    idx = {m: i for i, m in enumerate(models)}
    rng = np.random.default_rng(seed)
    return {name: _scores(_fit_predict(name, tr, te, tr.model.map(idx).to_numpy(), k), te,
                          te.model.map(idx).to_numpy(), k, rng) for name in sets}


def print_attribution(res: dict):
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    span = lambda ci: f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"  # noqa: E731
    print(f"\nAuthor classifiers on the text (chance {pct(res['chance'])}): one at a time / each model once per prompt")
    blocks = [("5 folds grouped by prompt", {n: r["cv"] for n, r in res["sets"].items()}),
              ("the game's held-out halves", {n: r["halves"] for n, r in res["sets"].items()})]
    if res.get("across"):
        blocks.append((f"trained on {res['train_revision']} half A, tested on {res['revision']} half B", res["across"]))
    for label, rows in blocks:
        print(f"  {label}")
        for name, r in rows.items():
            print(f"    {name:24} {pct(r['accuracy']):>6} {span(r['accuracy_ci']):16} "
                  f"{pct(r['matched_accuracy']):>6} {span(r['matched_accuracy_ci'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Author classifiers on the answers' text, the guessers' reference")
    add_run_args(ap)
    ap.add_argument("--train-revision", default=None,
                    help="also train on half A of this revision and test on half B of --revision")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = revision_dir(cfg, args.revision)
    models = [m["name"] for m in cfg["models"]]
    prompts = {p["id"]: p for p in json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))["prompts"]}
    df = load_answers(run_dir, prompts)
    print(f"{len(df)} answers from {df.model.nunique()} models in {cfg['revision']}")
    res = {"revision": cfg["revision"], "chance": 1.0 / len(models), "labels": models,
           "sets": attribute(df, models)}
    if args.train_revision:
        res["train_revision"] = args.train_revision
        res["across"] = across(load_answers(DATA_ROOT / args.train_revision, prompts), df, models)
    (run_dir / "attribution.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print_attribution(res)
    print(f"-> {run_dir / 'attribution.json'}")


if __name__ == "__main__":
    main()
