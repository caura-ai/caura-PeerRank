#!/usr/bin/env python3
"""Send every prompt to every model and append responses to data/spectra/<revision>/responses.jsonl.

Resumable: successful (model, prompt, sample) triples already in the file are skipped,
so you can stop and restart, or re-run to retry errors.

  python -m spectra.run --revision oct-14-26 --limit 5     # smoke test, 5 prompts
  python -m spectra.run --revision oct-14-26               # full run
"""
import argparse
import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from spectra import add_run_args, load_config, providers, run_dir as revision_dir, writer_lock


def completed_keys(path):
    keys = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):
                    keys.add((r["model"], r["prompt_id"], r["sample"]))
    return keys


def select_prompts(prompts: list[dict], limit: int | None) -> list[dict]:
    """Select up to `limit` prompts, stratified evenly across categories in round-robin order."""
    if not limit or limit >= len(prompts):
        return prompts
    by_cat: dict[str, list[dict]] = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)
    picked: list[dict] = []
    idx = 0
    while len(picked) < limit:
        added = False
        for cat_prompts in by_cat.values():
            if idx < len(cat_prompts):
                picked.append(cat_prompts[idx])
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
        idx += 1
    return picked


async def main():
    ap = argparse.ArgumentParser()
    add_run_args(ap)
    ap.add_argument("--limit", type=int, default=None, help="only N prompts, stratified across categories (smoke test)")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = revision_dir(cfg, args.revision)
    prompts = select_prompts(json.loads(Path(cfg["prompts"]).read_text(encoding="utf-8"))["prompts"], args.limit)
    models = [m for m in cfg["models"] if not args.models or m["name"] in args.models]
    resolved = {m["name"]: providers.describe(m) for m in models}  # an unknown name fails here, before any call

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "responses.jsonl"
    with writer_lock(out_path):
        (run_dir / "run_config.json").write_text(json.dumps(
            {"started": datetime.now(timezone.utc).isoformat(), "config": cfg, "models": list(resolved.values())},
            indent=2), encoding="utf-8")

        done = completed_keys(out_path)
        tasks = [(m, p, s) for m in models for p in prompts for s in range(cfg["samples"])
                 if (m["name"], p["id"], s) not in done]
        random.shuffle(tasks)  # spread load across providers and avoid order effects in rate limiting
        print(f"{len(tasks)} calls to make ({len(done)} already done) across {len(models)} models -> {run_dir}")
        if not tasks:
            return

        sems = providers.semaphores(models, cfg["concurrency_per_model"])  # shared per provider
        lock = asyncio.Lock()
        counts = {"ok": 0, "err": 0}

        async def worker(m, p, s):
            async with sems[m["name"]]:
                d = resolved[m["name"]]
                rec = {"model": m["name"], "backend": d["backend"], "model_id": d.get("model_id") or d.get("litellm_model"),
                       "prompt_id": p["id"], "category": p["category"], "sample": s,
                       "timestamp": datetime.now(timezone.utc).isoformat()}
                try:
                    reply = await providers.complete(m, p["prompt"], system_prompt=cfg.get("system_prompt"),
                                                     timeout=cfg["timeout_seconds"], retries=cfg["retries"])
                    rec.update(reply.fields())
                    counts["ok"] += 1
                except Exception as e:  # keep going; errors are retried on the next run
                    rec.update(text=None, error=f"{type(e).__name__}: {e}"[:500])
                    counts["err"] += 1
                async with lock:
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n = counts["ok"] + counts["err"]
                    if n % 25 == 0 or n == len(tasks):
                        print(f"  {n}/{len(tasks)} done, {counts['err']} errors")

        try:
            await asyncio.gather(*(worker(m, p, s) for m, p, s in tasks))
        finally:
            await providers.close()
        print(f"Finished: {counts['ok']} ok, {counts['err']} errors -> {out_path}")
        if counts["err"]:
            print("Re-run the same command to retry the failed calls.")


if __name__ == "__main__":
    asyncio.run(main())
