#!/usr/bin/env python3
"""Start a revision for a larger panel from an existing one, so only the new models are collected.

  python -m spectra.extend --from oct-23-26 --to oct-23-26-p8 --keep-coder deepseek-v4-pro

Copies into data/spectra/<to>/:
- the answers;
- the labels of the coder that stays (the other coder is replaced and recodes everything);
- the contagion records.

Answers, coding and contagion then resume on the copy: run.py, code_behaviors.py and voice.py skip what exists and
collect only the new models' calls. Writes extend.json with the source and the counts. Refuses to overwrite a
revision that already has answers.
"""
import argparse
import json
from datetime import datetime, timezone

from spectra import DATA_ROOT
from spectra.guess import load_jsonl


def extend(source: str, target: str, keep_coder: str | None) -> dict:
    src, dst = DATA_ROOT / source, DATA_ROOT / target
    if (dst / "responses.jsonl").exists():
        raise SystemExit(f"{dst} already has answers; extend a fresh revision")
    dst.mkdir(parents=True, exist_ok=True)

    def copy(name, keep=lambda r: True):
        rows = [r for r in load_jsonl(src / name) if keep(r)]
        (dst / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return len(rows)

    counts = {"responses": copy("responses.jsonl"),
              "codes": copy("codes.jsonl", lambda r: keep_coder is not None and r.get("coder") == keep_coder),
              "voice": copy("voice.jsonl", lambda r: r.get("experiment") == "contagion")}
    before = sorted({r["model"] for r in load_jsonl(dst / "responses.jsonl")})
    record = {"from": source, "to": target, "kept_coder": keep_coder, "copied": counts, "models_before": before,
              "at": datetime.now(timezone.utc).isoformat()}
    (dst / "extend.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description="Start a larger panel's revision from an existing one")
    ap.add_argument("--from", dest="source", required=True)
    ap.add_argument("--to", dest="target", required=True)
    ap.add_argument("--keep-coder", default=None, help="the coder whose labels are copied; any other coder recodes all")
    args = ap.parse_args(argv)
    print(json.dumps(extend(args.source, args.target, args.keep_coder), indent=2))


if __name__ == "__main__":
    main()
