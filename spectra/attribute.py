#!/usr/bin/env python3
"""Who wrote this answer? Author classifiers on the text alone, the reference the guessers are compared with.

  python -m spectra.attribute --revision oct-23-26                           # this run, grouped by prompt
  python -m spectra.attribute --revision oct-23-26 --train-revision oct-14-26   # also train on one run, test on another
  python -m spectra.attribute --revision oct-23-26 --what ablations curve      # what carries identity; how many answers

Three feature sets, each a logistic regression with settings fixed in advance (none tuned on these runs):
  words+style            word 1-2-grams and the style features: analyze.py's classifier, reproduced
  characters             character 2-5-grams, the standard stylometry baseline (punctuation, markdown, spacing)
  characters+words+style all of them
Each is scored one answer at a time and jointly per prompt (each model once, as the guessing game names them), with
5-fold cross-validation grouped by prompt, on the game's held-out halves, and, with --train-revision, trained on
half A of one run and tested on half B of another, so neither the prompts nor the answers are shared.

ablations retrains the character classifier on text with layers removed (markdown, punctuation, case and line
breaks, everything after the first 50 words, word order), so the drop shows what each layer carries. curve trains it
on 10 to 160 answers per model, the same prompts for every model, and tests on prompts it never saw.
Writes data/spectra/<revision>/attribution.json, keeping what earlier runs of other checks wrote. No API calls.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from spectra import DATA_ROOT, add_run_args, load_config, run_dir as revision_dir
from spectra.analyze import features, match_by_prompt, numeric_matrix

FEATURE_SETS = ("words+style", "characters", "characters+words+style")
CURVE_SIZES = (10, 20, 40, 80, 160)

_MD_LINE = [
    (re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$"), ""),  # table separator rows
    (re.compile(r"^\s*([-*_]\s*){3,}$"), ""),  # horizontal rules
    (re.compile(r"^\s{0,3}#{1,6}\s*"), ""),  # headers
    (re.compile(r"^\s*>\s?"), ""),  # block quotes
    (re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+"), ""),  # list markers
]
_MD_INLINE = [
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # links keep their text
    (re.compile(r"\*\*|__|`"), ""),  # bold and inline code
    (re.compile(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)"), ""),  # italics
    (re.compile(r"\s*\|\s*"), " "),  # table cells
]


def strip_markdown(text: str) -> str:
    """The words and line breaks without markdown: headers, lists, tables, emphasis, quotes, rules and links."""
    lines = []
    for line in text.split("\n"):
        for rx, sub in _MD_LINE:
            line = rx.sub(sub, line)
        for rx, sub in _MD_INLINE:
            line = rx.sub(sub, line)
        lines.append(line.strip())
    return "\n".join(lines).strip()


def strip_punctuation(text: str) -> str:
    """Letters, digits and whitespace only: punctuation, dashes, symbols and emoji go."""
    return "\n".join(" ".join(line.split()) for line in re.sub(r"[^\w\s]|_", " ", text).split("\n"))


def bare_words(text: str) -> str:
    """No markdown, punctuation, case or line breaks: only the words, in order."""
    return " ".join(strip_punctuation(strip_markdown(text)).lower().split())


def _shuffled(text: str) -> str:
    words = bare_words(text).split()
    random.Random(text).shuffle(words)
    return " ".join(words)


# Each removes one more layer, so the drop from one row to the next is what that layer carried.
ABLATIONS = {
    "original": lambda t: t,
    "no markdown": strip_markdown,
    "no punctuation": strip_punctuation,
    "no markdown or punctuation": lambda t: strip_punctuation(strip_markdown(t)),
    "bare words (also no case or line breaks)": bare_words,
    "bare words, first 50 only": lambda t: " ".join(bare_words(t).split()[:50]),
    "bare words, shuffled": _shuffled,
}


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


def _cv_folds(df: pd.DataFrame, y: np.ndarray) -> list:
    from sklearn.model_selection import GroupKFold

    return list(GroupKFold(n_splits=min(5, df.prompt_id.nunique())).split(df, y, df.prompt_id))


def ablations(df: pd.DataFrame, models: list[str], seed: int = 7, transforms: dict = ABLATIONS) -> dict:
    """The character classifier, retrained and tested on text with layers removed, 5 folds grouped by prompt."""
    k = len(models)
    df = df[df.model.isin(models)].reset_index(drop=True)
    y = df.model.map({m: i for i, m in enumerate(models)}).to_numpy()
    folds = _cv_folds(df, y)
    rng = np.random.default_rng(seed)
    out = {}
    for name, fn in transforms.items():
        view = df.assign(prose=df.prose.map(fn))
        proba = np.zeros((len(df), k))
        for tr, te in folds:
            proba[te] = _fit_predict("characters", view.iloc[tr], view.iloc[te], y[tr], k)
        out[name] = _scores(proba, df, y, k, rng)
        out[name]["mean_chars"] = float(view.prose.str.len().mean())
    return out


def learning_curve(df: pd.DataFrame, models: list[str], sizes=CURVE_SIZES, repeats: int = 5, seed: int = 7) -> dict:
    """The character classifier trained on n answers per model (the same n prompts for every model).

    Each of the 5 folds grouped by prompt draws n of its training prompts, repeats times (once when n is all of them),
    and tests on the fold's unseen prompts. Reported per n: the mean over draws and the lowest and highest draw.
    """
    k = len(models)
    df = df[df.model.isin(models)].reset_index(drop=True)
    y = df.model.map({m: i for i, m in enumerate(models)}).to_numpy()
    folds = _cv_folds(df, y)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    pools = [sorted(set(df.prompt_id.iloc[tr])) for tr, _ in folds]
    out = {}
    for n in sizes:
        full = n >= min(len(p) for p in pools)  # every training prompt, so a single draw
        used = 1 if full else repeats
        draws = [np.zeros((len(df), k)) for _ in range(used)]
        for (tr, te), pool in zip(folds, pools):
            picks = [pool] if full else [rng.sample(pool, n) for _ in range(repeats)]
            for proba, pick in zip(draws, picks):
                rows = tr[df.prompt_id.iloc[tr].isin(pick).to_numpy()]
                proba[te] = _fit_predict("characters", df.iloc[rows], df.iloc[te], y[rows], k)
        scores = [_scores(p, df, y, k, np_rng) for p in draws]
        out[str(n)] = {"answers_per_model": min(len(p) for p in pools) if full else n, "draws": used}
        for kind in ("accuracy", "matched_accuracy"):
            vals = [s[kind] for s in scores]
            out[str(n)][kind] = float(np.mean(vals))
            out[str(n)][kind + "_range"] = [float(min(vals)), float(max(vals))]
    return out


class CharClassifier:
    """The character classifier as a reusable reference: fit on real answers, then score any new text.

    Phase 2 trains it on the real answers of one or more runs to prompts an experiment does not use, and asks it who
    wrote an imitation, a rewrite or an answer given another model's notes.
    """

    def __init__(self, models: list[str]):
        self.models = list(models)

    def fit(self, df: pd.DataFrame) -> "CharClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        df = df[df.model.isin(self.models)]
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=200000, sublinear_tf=True)
        X = self.vec.fit_transform(df.prose)
        self.clf = LogisticRegression(max_iter=3000, C=4.0).fit(X, df.model.map(self.models.index).to_numpy())
        return self

    def predict_proba(self, texts) -> np.ndarray:
        """(texts x models) probabilities. Texts are raw answers; code blocks are removed as in training."""
        from spectra.analyze import split_code

        proba = np.zeros((len(texts), len(self.models)))
        proba[:, self.clf.classes_] = self.clf.predict_proba(self.vec.transform([split_code(t or "")[0] for t in texts]))
        return proba


def reference_classifier(revisions: list[str], prompts: dict, models: list[str], exclude=()) -> CharClassifier:
    """A CharClassifier trained on the real answers of these revisions, leaving out the prompts in `exclude`."""
    frames = [load_answers(DATA_ROOT / rev, prompts) for rev in revisions]
    df = pd.concat(frames, ignore_index=True)
    return CharClassifier(models).fit(df[~df.prompt_id.isin(set(exclude))])


def print_attribution(res: dict):
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    span = lambda ci: f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"  # noqa: E731
    print(f"\nAuthor classifiers on the text (chance {pct(res['chance'])}): one at a time / each model once per prompt")
    blocks = []
    if res.get("sets"):
        blocks += [("5 folds grouped by prompt", {n: r["cv"] for n, r in res["sets"].items()}),
                   ("the game's held-out halves", {n: r["halves"] for n, r in res["sets"].items()})]
    if res.get("across"):
        blocks.append((f"trained on {res['train_revision']} half A, tested on {res['revision']} half B", res["across"]))
    if res.get("ablations"):
        blocks.append(("characters, with layers of the text removed (5 folds grouped by prompt)", res["ablations"]))
    for label, rows in blocks:
        print(f"  {label}")
        for name, r in rows.items():
            print(f"    {name:42} {pct(r['accuracy']):>6} {span(r['accuracy_ci']):16} "
                  f"{pct(r['matched_accuracy']):>6} {span(r['matched_accuracy_ci'])}")
    if res.get("curve"):
        print("  characters, trained on n answers per model (mean over draws, lowest to highest draw)")
        for r in res["curve"].values():
            print(f"    {r['answers_per_model']:4} answers per model  {pct(r['accuracy']):>6} "
                  f"{span(r['accuracy_range']):16} {pct(r['matched_accuracy']):>6} {span(r['matched_accuracy_range'])}"
                  f"  ({r['draws']} draw{'s' if r['draws'] > 1 else ''})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Author classifiers on the answers' text, the guessers' reference")
    add_run_args(ap)
    ap.add_argument("--train-revision", default=None,
                    help="also train on half A of this revision and test on half B of --revision")
    ap.add_argument("--what", nargs="+", choices=("sets", "ablations", "curve"), default=["sets"],
                    help="sets: the three feature sets; ablations: text with layers removed; curve: 10 to 160 "
                         "answers per model. Results of checks not run are kept from the last attribution.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = revision_dir(cfg, args.revision)
    models = [m["name"] for m in cfg["models"]]
    prompts = {p["id"]: p for p in json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))["prompts"]}
    df = load_answers(run_dir, prompts)
    print(f"{len(df)} answers from {df.model.nunique()} models in {cfg['revision']}")
    path = run_dir / "attribution.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    res.update(revision=cfg["revision"], chance=1.0 / len(models), labels=models)
    if "sets" in args.what:
        res["sets"] = attribute(df, models)
        if args.train_revision:
            res["train_revision"] = args.train_revision
            res["across"] = across(load_answers(DATA_ROOT / args.train_revision, prompts), df, models)
    if "ablations" in args.what:
        res["ablations"] = ablations(df, models)
    if "curve" in args.what:
        res["curve"] = learning_curve(df, models)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print_attribution(res)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
