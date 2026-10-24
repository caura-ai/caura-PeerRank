#!/usr/bin/env python3
"""Third-coder tie-break: a coder from another model family labels only the answers the two coders disagree on.

  python -m spectra.tiebreak --revision oct-23-26 --coder kimi-k3 --dry-run
  python -m spectra.tiebreak --revision oct-23-26 --coder kimi-k3          # labels answer_first and moralizing

The third coder sees exactly what the first two saw (the same instructions, tags, prompt and answer) and is
neither a panel model nor one of the configured coders. It writes data/spectra/<revision>/tiebreak.jsonl,
resumable, one record per disputed answer, and tiebreak.json with the summary: per label, which coder the third
one sided with, and each model's rate and rank under coder A, coder B and the majority of three.
codes.jsonl is not touched, so every trait in analysis.json stays the two-coder average it was.
"""
import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from spectra import add_run_args, load_config, providers, run_dir as revision_dir, writer_lock
from spectra.code_behaviors import build_message, parse_codes
from spectra.guess import load_jsonl

DEFAULT_LABELS = ("answer_first", "moralizing")
EST_OUTPUT_TOKENS = 1500  # a reasoning coder's typical reply, thinking included, for the dry-run estimate


def latest_codes(rows: list[dict]) -> dict:
    """{(model, prompt, sample): {coder: codes}}, the last usable record per coder."""
    out: dict = {}
    for c in rows:
        if c.get("codes"):
            out.setdefault((c["model"], c["prompt_id"], c.get("sample", 0)), {})[c["coder"]] = c["codes"]
    return out


def disputed(codes: dict, coders: list[str], labels: list[str]) -> dict:
    """{answer: [labels the coders disagree on]} among answers every coder labeled."""
    out = {}
    for key, by in codes.items():
        if all(c in by for c in coders):
            split = [lab for lab in labels if len({by[c].get(lab) for c in coders}) > 1]
            if split:
                out[key] = split
    return out


def summarize(codes: dict, ties: dict, coders: list[str], third: str, labels: list[str], models: list[str],
              positives: dict) -> dict:
    """Per label: how the third coder split the disputes, and model rates and ranks under each coder and the majority."""
    from scipy.stats import spearmanr

    a, b = coders
    out = {"coders": coders, "third": third, "labels": {}}
    for lab in labels:
        pos = positives.get(lab, {"yes"})
        sided, rates = Counter(), {view: {m: [0, 0] for m in models} for view in (a, b, "majority")}
        for key, by in codes.items():
            if key[0] not in models or not all(c in by for c in coders):
                continue
            va, vb = by[a].get(lab), by[b].get(lab)
            if va is None or vb is None:
                continue
            major = va
            if va != vb:
                t = (ties.get(key) or {}).get(lab)
                if t is None:
                    sided["not coded"] += 1
                    major = None
                else:
                    sided[a if t == va else b if t == vb else "neither"] += 1
                    votes = Counter([va, vb, t]).most_common()
                    major = votes[0][0] if votes[0][1] >= 2 else None
            for view, value in ((a, va), (b, vb), ("majority", major)):
                if value is not None:  # an answer without a majority counts for each coder but not for the majority
                    rates[view][key[0]][0] += value in pos
                    rates[view][key[0]][1] += 1
        share = {view: {m: (c / n if n else None) for m, (c, n) in per.items()} for view, per in rates.items()}
        rank = {}
        for view in (a, b):
            xs = [share[view][m] for m in models]
            ys = [share["majority"][m] for m in models]
            ok = None not in xs + ys and len(set(xs)) > 1 and len(set(ys)) > 1
            rank[view] = float(spearmanr(xs, ys).statistic) if ok else None
        out["labels"][lab] = {"disputed": sum(sided.values()), "sided_with": dict(sided), "rates": share,
                              "rank_vs_majority": rank}
    return out


def print_summary(s: dict):
    a, b = s["coders"]
    for lab, r in s["labels"].items():
        print(f"\n{lab}: {r['disputed']} disputed answers; {s['third']} sided with "
              + ", ".join(f"{k} {v}" for k, v in r["sided_with"].items()))
        for m, v in r["rates"]["majority"].items():
            print(f"  {m:20} majority {100 * v:5.1f}%   {a} {100 * r['rates'][a][m]:5.1f}%   "
                  f"{b} {100 * r['rates'][b][m]:5.1f}%")
        print("  rank agreement with the majority (Spearman): "
              + ", ".join(f"{k} {'n/a' if v is None else f'{v:.2f}'}" for k, v in r["rank_vs_majority"].items()))


async def main():
    ap = argparse.ArgumentParser(description="A third coder breaks ties on the labels the two coders disagree on")
    add_run_args(ap)
    ap.add_argument("--coder", required=True, help="a model from another family: not a panel model, not a coder")
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="comma-separated codebook labels")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="count the disputed answers and estimate the cost")
    args = ap.parse_args()

    from spectra.analyze import CODED_TRAITS  # positive values per code, as the traits count them

    cfg = load_config(args.config)
    models = [m["name"] for m in cfg["models"]]
    coders = [c["name"] for c in cfg["coders"]]
    if len(coders) != 2:
        raise SystemExit("The tie-break needs exactly two configured coders")
    if args.coder in models + coders:
        raise SystemExit(f"{args.coder} is a panel model or a configured coder; pick one from another family")
    third = {"name": args.coder, "temperature": 0}
    providers.resolve(third)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    positives = {code: pos for _, _, _, _, code, pos, _, _ in CODED_TRAITS}
    run_dir = revision_dir(cfg, args.revision)
    prompts = {p["id"]: p for p in json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))["prompts"]}
    answers = {(r["model"], r["prompt_id"], r.get("sample", 0)): r
               for r in load_jsonl(run_dir / "responses.jsonl") if r.get("text")}
    codes = latest_codes(load_jsonl(run_dir / "codes.jsonl"))
    split = {k: v for k, v in disputed(codes, coders, labels).items() if k in answers}
    out_path = run_dir / "tiebreak.jsonl"

    def ties() -> dict:
        return {(r["model"], r["prompt_id"], r.get("sample", 0)): r["codes"]
                for r in load_jsonl(out_path) if r.get("codes") and r.get("coder") == args.coder}

    jobs = [k for k in split if k not in ties()]
    print(f"{len(split)} answers with a disputed label ({', '.join(f'{lab} {sum(lab in v for v in split.values())}' for lab in labels)}); "
          f"{len(jobs)} still to label with {args.coder}")
    if args.dry_run:
        tokens = [len(build_message(prompts[k[1]], answers[k]["text"])) / 4 for k in jobs]
        cost = sum(providers.token_cost(third, t, EST_OUTPUT_TOKENS) or 0.0 for t in tokens)
        print(f"Estimated cost ${cost:.2f} ({len(jobs)} calls, about {EST_OUTPUT_TOKENS} output tokens each). "
              "Dry run: no model was called.")
        return

    with writer_lock(out_path):
        random.shuffle(jobs)
        sem = asyncio.Semaphore(providers.max_concurrency(third, args.concurrency))
        lock = asyncio.Lock()

        async def work(key):
            r = answers[key]
            rec = {"coder": args.coder, "model": key[0], "prompt_id": key[1], "sample": key[2], "disputed": split[key]}
            async with sem:
                try:
                    reply = await providers.complete(third, build_message(prompts[key[1]], r["text"]),
                                                     timeout=cfg["timeout_seconds"], retries=cfg["retries"])
                    rec["codes"] = parse_codes(reply.text)
                    rec["cost_usd"] = reply.cost_usd
                    if rec["codes"] is None:
                        rec["error"] = "unparseable"
                        rec["raw"] = (reply.text or "")[:500]
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        try:
            await asyncio.gather(*(work(k) for k in jobs))
        finally:
            await providers.close()
    done = ties()
    spent = sum(r.get("cost_usd") or 0 for r in load_jsonl(out_path))
    print(f"Labeled {len(done)} of {len(split)} disputed answers (spent ${spent:.2f} in all) -> {out_path}")
    if not done:
        errors = Counter(r.get("error", "")[:60] for r in load_jsonl(out_path) if not r.get("codes"))
        raise SystemExit(f"No answer was labeled, so there is no summary. Errors: {dict(errors)}")
    summary = summarize(codes, done, coders, args.coder, labels, models, positives)
    (run_dir / "tiebreak.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    asyncio.run(main())
