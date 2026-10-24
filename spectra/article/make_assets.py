#!/usr/bin/env python3
"""Generate every number, table and figure the Spectra article uses, from the run data.

  python spectra/article/make_assets.py

Reads data/spectra/<run>/ (analysis.json, codes.jsonl, guesses.jsonl, run_config.json) and the prompt set,
and writes spectra/article/generated/: numbers.tex (one macro per number), tab_*.tex and fig_*.pdf/png.
The article reads numbers only through these files, so no figure in the text is typed by hand.
"""
import collections
import hashlib
import json
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spectra.code_behaviors import CODEBOOK  # noqa: E402
from spectra.guess import game_key, load_jsonl, prompt_effects, summarize_all  # noqa: E402

MAIN, REPL = "oct-23-26", "oct-14-26"  # the recent run, and the earlier run of the same prompts
DATA = ROOT / "data" / "spectra"
OUT = pathlib.Path(__file__).resolve().parent / "generated"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]  # Okabe-Ito, roster order
macros: list[tuple[str, str]] = []


def mac(name: str, value) -> None:
    assert name.isalpha(), name
    macros.append((name, str(value)))


def half_up(x, digits):
    """Round half up: 31/200 is 15.5% and prints as 16%, whatever its binary float says."""
    return f"{round(x + 1e-9, digits):.{digits}f}"


def pct(v, digits=1):
    return f"{half_up(100 * v, digits)}\\%"


def ci(pair, digits=1):
    return f"{half_up(100 * pair[0], digits)}--{half_up(100 * pair[1], digits)}\\%"


def tt(name):
    return f"\\texttt{{{name}}}"


def load(run):
    analysis = json.loads((DATA / run / "analysis.json").read_text(encoding="utf-8"))
    traits = {t["key"]: t for t in analysis["traits"]}
    return analysis, traits


A, T = load(MAIN)
A0, T0 = load(REPL)
models = A["meta"]["models"]
assert models == A0["meta"]["models"]
val = lambda t, m: t["per_model"][m]["value"]  # noqa: E731
prompts = json.loads((ROOT / "spectra" / "prompts" / "prompts_v1.json").read_text(encoding="utf-8"))
P = prompts["prompts"]
run_config = json.loads((DATA / MAIN / "run_config.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- design numbers
traps = ["FP", "ERR", "AMB", "OP", "EMO"]
tag_n = collections.Counter(t for p in P for t in p["tags"])
mac("nPrompts", len(P))
mac("nCategories", len({p["category"] for p in P}))
mac("nPerCategory", len(P) // len({p["category"] for p in P}))
mac("nModels", len(models))
mac("nTraits", len(A["traits"]))
mac("nStyleTraits", sum(t["group"] == "style" for t in A["traits"]))
mac("nBehaviorTraits", sum(t["group"] == "behavior" for t in A["traits"]))
mac("nTrapPrompts", sum(1 for p in P if set(p["tags"]) & set(traps)))
for tag in ["FP", "ERR", "AMB", "OP", "EMO", "CAS", "EXP"]:
    mac("n" + tag.title().replace("Fp", "FP"), tag_n[tag])
mac("nLimitPrompts", sum(1 for p in P if p.get("constraint")))
mac("nLimitKinds", len({p["constraint"]["type"] for p in P if p.get("constraint")}))
# the locked test set: the hash in its lock file, checked against the file itself
v2 = json.loads((ROOT / "spectra" / "prompts" / "prompts_v2.json").read_text(encoding="utf-8"))
v2_hash = hashlib.sha256(json.dumps(v2, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
assert v2_hash == (ROOT / "spectra" / "prompts" / "prompts_v2.lock").read_text(encoding="utf-8").split()[0]
assert len(v2["prompts"]) == len(P)
mac("vTwoHash", v2_hash[:16])
mac("nAnswers", f"{A['meta']['n_responses']:,}".replace(",", "{,}"))
mac("nCoders", len(A["meta"]["coders"]))
mac("coderA", tt(A["meta"]["coders"][1]))
mac("coderB", tt(A["meta"]["coders"][0]))
mac("costRun", f"\\${sum(v['total_cost_usd'] or 0 for v in A['runstats'].values()):.2f}")
mac("costRunRepl", f"\\${sum(v['total_cost_usd'] or 0 for v in A0['runstats'].values()):.2f}")
first = min(r["timestamp"] for r in load_jsonl(DATA / REPL / "responses.jsonl"))
second = min(r["timestamp"] for r in load_jsonl(DATA / MAIN / "responses.jsonl"))
from datetime import datetime  # noqa: E402
gap_days = round((datetime.fromisoformat(second) - datetime.fromisoformat(first)).total_seconds() / 86400)
mac("daysApart", gap_days)
mac("maxTokens", f"{min(m['max_tokens'] for m in run_config['models']):,}".replace(",", "{,}"))
mac("maxTokensArgon", f"{max(m['max_tokens'] for m in run_config['models']):,}".replace(",", "{,}"))

# ---------------------------------------------------------------- settings table
rows = []
vendor = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "meta": "Meta"}
for m in run_config["models"]:
    rows.append(f"{tt(m['name'])} & {vendor.get(m['provider'], m['provider'])} & {m['reasoning']} & {m['max_tokens']:,} & {m['temperature']:.1f} \\\\")
(OUT / "tab_settings.tex").parent.mkdir(parents=True, exist_ok=True)
(OUT / "tab_settings.tex").write_text(
    "\\begin{tabular}{lllrr}\n\\toprule\nModel & Provider & Thinking & Max output tokens & Temperature \\\\\n\\midrule\n"
    + "\n".join(rows).replace(",", "{,}") + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- profile table (main run)
cols = [("length", "Length (words)", lambda v: f"{v:.0f}"), ("structure", "Format /100w", lambda v: f"{v:.1f}"),
        ("hedging", "Hedges /100w", lambda v: f"{v:.2f}"), ("emdash", "Em dashes /100w", lambda v: f"{v:.2f}"),
        ("readability", "Reading ease", lambda v: f"{v:.0f}"), ("answer_first", "Answers first", lambda v: f"{100 * v:.0f}\\%"),
        ("initiative", "Does extra", lambda v: f"{100 * v:.0f}\\%"), ("adherence", "Limits met", lambda v: f"{100 * v:.0f}\\%")]
lines = []
for m in models:
    cells = []
    for key, _, fmt in cols:
        t = T[key]
        v = val(t, m)
        hi = max(val(t, x) for x in models)
        lo = min(val(t, x) for x in models)
        s = fmt(v)
        if abs(v - hi) < 1e-12:
            s = f"\\textbf{{{s}}}"
        elif abs(v - lo) < 1e-12:
            s = f"\\underline{{{s}}}"
        cells.append(s)
    lines.append(tt(m) + " & " + " & ".join(cells) + " \\\\")
(OUT / "tab_profile.tex").write_text(
    "\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\nModel & " + " & ".join(c[1] for c in cols)
    + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- headline profile numbers
def sweep(traits, key, model, high):
    cats = traits[key]["by_category"]
    won = total = 0
    for c, v in cats[model].items():
        rest = [cats[o][c] for o in cats if o != model and cats[o].get(c) is not None]
        if v is None or not rest:
            continue
        total += 1
        won += (v >= max(rest)) if high else (v <= min(rest))
    return won, total


short = min(models, key=lambda m: val(T["length"], m))
long_ = max(models, key=lambda m: val(T["length"], m))
assert short == min(models, key=lambda m: val(T0["length"], m)) and long_ == max(models, key=lambda m: val(T0["length"], m))
mac("shortModel", tt(short))
mac("longModel", tt(long_))
mac("lenShort", f"{val(T['length'], short):.0f}")
mac("lenLong", f"{val(T['length'], long_):.0f}")
mac("lenShortRepl", f"{val(T0['length'], short):.0f}")
mac("lenLongRepl", f"{val(T0['length'], long_):.0f}")
for name, traits, model, high in (("sweepShort", T, short, False), ("sweepLong", T, long_, True),
                                  ("sweepShortRepl", T0, short, False), ("sweepLongRepl", T0, long_, True)):
    won, total = sweep(traits, "length", model, high)
    mac(name, f"{won} of {total}")
mac("answerFirstShort", pct(val(T["answer_first"], short), 0))
mac("answerFirstLong", pct(val(T["answer_first"], long_), 0))
struct_top = max(models, key=lambda m: val(T["structure"], m))
read_top = max(models, key=lambda m: val(T["readability"], m))
mac("formatTop", tt(struct_top))
mac("formatTopVal", f"{val(T['structure'], struct_top):.1f}")
mac("readTop", tt(read_top))
mac("readTopVal", f"{val(T['readability'], read_top):.0f}")
# the same-vendor pair with the widest em-dash gap
prov = {m["name"]: m["provider"] for m in run_config["models"]}
pairs = [(a, b) for a in models for b in models if a != b and prov[a] == prov[b]]
dash_hi, dash_lo = max(pairs, key=lambda p: val(T["emdash"], p[0]) - val(T["emdash"], p[1]))
mac("dashHiModel", tt(dash_hi))
mac("dashLoModel", tt(dash_lo))
mac("dashHi", f"{val(T['emdash'], dash_hi):.2f}")
mac("dashLo", f"{val(T['emdash'], dash_lo):.2f}")
mac("dashRatio", f"{val(T['emdash'], dash_hi) / val(T['emdash'], dash_lo):.0f}")
stable_words = {m: [w for w in [s["term"] for s in A["signature"][m][:5]] if w in {s["term"] for s in A0["signature"][m][:5]}]
                for m in models}
mac("sigMuse", ", ".join(f"``{w}''" for w in stable_words["muse-wonder-1.0"]))
mac("sigFable", ", ".join(f"``{w}''" for w in stable_words["claude-fable-5.1"]))
mac("sigArgon", ", ".join(f"``{w}''" for w in stable_words["gemini-4-argon"]))
mac("nSigUnstable", sum(1 for m in models if not stable_words[m]))

# ---------------------------------------------------------------- stability across runs
d_now, d_old, per_trait = [], [], []
for key, t in T.items():
    if key not in T0:
        continue
    a = [val(t, m) for m in models]
    b = [val(T0[key], m) for m in models]
    if any(x is None for x in a + b):
        continue
    d_now += [t["per_model"][m]["d"] for m in models]
    d_old += [T0[key]["per_model"][m]["d"] for m in models]
    if len(set(a)) > 1 and len(set(b)) > 1:
        per_trait.append(spearmanr(a, b).statistic)
mac("stabR", f"{pearsonr(d_now, d_old).statistic:.2f}")
mac("stabCells", len(d_now))
mac("stabSpearmanMedian", f"{statistics.median(per_trait):.2f}")
mac("stabTraits", len(per_trait))

# ---------------------------------------------------------------- traps table
trap_keys = [("premise", "FP", "Challenges a false premise"), ("errors", "ERR", "Flags a pasted error"),
             ("ambiguity", "AMB", "Surfaces ambiguity"), ("agree_user", "OP", "Agrees with the user"),
             ("emotion", "EMO", "Acknowledges feelings")]
head = "Model & " + " & ".join(f"{label} ({tag})" for _, tag, label in trap_keys) + " \\\\"
lines = []
for m in models:
    lines.append(tt(m) + " & " + " & ".join(f"{100 * val(T[k], m):.0f}\\%" for k, _, _ in trap_keys) + " \\\\")
ns = " & ".join(f"$n={max(T[k]['per_model'][m]['n'] for m in models)}$" for k, _, _ in trap_keys)
(OUT / "tab_traps.tex").write_text(
    "\\begin{tabular}{l" + "r" * len(trap_keys) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines)
    + "\n\\midrule\nPrompts & " + ns + " \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
fp_no = {m: T["premise"]["per_model"][m]["no"] for m in models if T["premise"]["per_model"][m]["no"]}
fp_no_old = {m: T0["premise"]["per_model"][m]["no"] for m in models if T0["premise"]["per_model"][m]["no"]}
assert len(fp_no) == 1 and fp_no == fp_no_old
fp_model, fp_ids = next(iter(fp_no.items()))
mac("fpModel", tt(fp_model))
mac("fpPrompt", fp_ids[0])
mac("fpTotal", 2 * len(models) * tag_n["FP"])
mac("fpMissed", len(fp_no[fp_model]) + len(fp_no_old[fp_model]))
err_full = [m for m in models if val(T["errors"], m) >= 0.999 and val(T0["errors"], m) >= 0.999]
mac("errPerfect", " and ".join(tt(m) for m in err_full))
missed_both = sorted({p for m in models for p in (set(T["errors"]["per_model"][m]["no"] or []) &
                                                    set(T0["errors"]["per_model"][m]["no"] or []))})
cats = sorted({next(p["category"] for p in P if p["id"] == i) for i in missed_both})
assert cats == ["summarize"], cats
mac("errMissIds", ", ".join(missed_both[:-1]) + " and " + missed_both[-1])
mac("errNSummarize", sum(1 for p in P if "ERR" in p["tags"] and p["category"] == "summarize"))
adh = {m: val(T["adherence"], m) for m in models}
mac("limitsLow", pct(min(adh.values()), 0))
agree_user = [val(T["agree_user"], m) for m in models]
mac("agreeLow", pct(min(agree_user), 0))
mac("agreeHigh", pct(max(agree_user), 0))
never_agreed = set.intersection(*[set(t["agree_user"]["per_model"][m]["no"] or []) for t in (T, T0) for m in models])
mac("nOpNeverAgreed", len(never_agreed))
amb_none = sorted(set.intersection(*[set(t["ambiguity"]["per_model"][m]["no"] or []) for t in (T, T0) for m in models]))
assert len(amb_none) == 1, amb_none
mac("ambNone", amb_none[0])
mac("limitsHigh", pct(max(adh.values()), 0))

# ---------------------------------------------------------------- classifier and style map
for name, analysis in (("", A), ("Repl", A0)):
    d = analysis["distinctiveness"]
    mac("clfFull" + name, pct(d["accuracy"]))
    mac("clfStyle" + name, pct(d["style_only_accuracy"]))
    mac("clfMatched" + name, pct(d["matched_accuracy"]))
    mac("clfHalfMatched" + name, pct(d["halves"]["matched_accuracy"]))
    if not name:
        mac("clfMatchedCI", ci(d["matched_accuracy_ci"]))
        mac("clfStyleMatched", pct(d["style_only_matched_accuracy"]))
        mac("clfHalf", pct(d["halves"]["accuracy"]))
        mac("clfHalfCI", ci(d["halves"]["accuracy_ci"]))
        mac("clfHalfMatchedCI", ci(d["halves"]["matched_accuracy_ci"]))
        mac("clfStyleHalfMatched", pct(d["halves"]["style_only_matched_accuracy"]))
    best = max(((d["labels"][i], d["labels"][j], v) for i, row in enumerate(d["matrix"]) for j, v in enumerate(row) if i != j),
               key=lambda x: x[2])
    mac("confRate" + name, pct(best[2]))
    if not name:
        mac("confAuthor", tt(best[0]))
        mac("confNamed", tt(best[1]))
mac("chance", pct(1 / len(models)))
mac("pcOneVar", pct(A["style_map"]["variance"][0], 0))
mac("pcTwoVar", pct(A["style_map"]["variance"][1], 0))
mac("pcOnePos", ", ".join(A["style_map"]["axes"]["pc1"]["pos"]))
mac("pcOneNeg", ", ".join(A["style_map"]["axes"]["pc1"]["neg"]))
d = A["distinctiveness"]
cells = lambda s, key: f"{pct(s[key])} ({ci(s[key + '_ci'])})"  # noqa: E731
lines = [f"{split} & {feats} & {cells(s, pre + 'accuracy')} & {cells(s, pre + 'matched_accuracy')} \\\\"
         for split, s in (("Other 4/5 of prompts", d), ("Other half (the game's split)", d["halves"]))
         for feats, pre in (("Words and style", ""), ("Style only", "style_only_"))]
(OUT / "tab_classifier.tex").write_text(
    "\\begin{tabular}{llrr}\n\\toprule\nTrained on & Features & One answer at a time & Six answers jointly \\\\\n\\midrule\n"
    + "\n".join(lines[:2]) + "\n\\addlinespace\n" + "\n".join(lines[2:]) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- guessing game
guess_records = load_jsonl(DATA / MAIN / "guesses.jsonl")
games = {(g["mode"], g["assign"]): g for g in summarize_all(guess_records, models, MAIN) if g["profile"] == "base"}
blind, same = games[("blind", "permutation")], games[("briefed", "permutation")]
held, held_ind = games[("heldout", "permutation")], games[("heldout", "independent")]
assert same["research"] == [MAIN]
assert min(g["n_calls"] for g in games.values()) >= 0.98 * len(P) * len(models), "a game is incomplete"
GAMES = (("Blind", blind), ("Same", same), ("HeldPerm", held), ("HeldInd", held_ind))
for name, g in GAMES:
    mac("acc" + name, pct(g["accuracy"]))
    mac("ci" + name, ci(g["ci"]))
    mac("vote" + name, pct(g["vote"]["accuracy"]))
    mac("voteCI" + name, ci(g["vote"]["ci"]))
    mac("calls" + name, f"{g['n_calls']:,}".replace(",", "{,}"))
    mac("ties" + name, g["vote"]["ties"])
    mac("wvote" + name, pct(g["vote_weighted"]["accuracy"]))
    mac("wvoteCI" + name, ci(g["vote_weighted"]["ci"]))
mac("callsTotal", f"{sum(g['n_calls'] for _, g in GAMES):,}".replace(",", "{,}"))
gain = same["accuracy"] - blind["accuracy"]
mac("retention", pct((held["accuracy"] - blind["accuracy"]) / gain, 0))  # same naming rule as the same-answer game
mac("retentionInd", pct((held_ind["accuracy"] - blind["accuracy"]) / gain, 0))
mac("namingRuleGap", f"{100 * abs(held['accuracy'] - held_ind['accuracy']):.1f}")
mac("repeatHeld", pct(held_ind["repeat_rate"]))
mac("ownFoundBlind", pct(blind["own_answer_effect"]["found"], 0))
mac("ownNotBlind", pct(blind["own_answer_effect"]["not_found"], 0))

# how much belongs to the prompt, and the own-answer effect with prompt, guesser and author held fixed
effects = {name: prompt_effects(guess_records, models, (g["mode"], (g["research"] or [None])[0], g["assign"]))
           for name, g in GAMES}
for name, e in effects.items():
    assert e["n_judgments"] == dict(GAMES)[name]["n_judgments"], name
    mac("iccPrompt" + name, f"{e['icc']['prompt']:.2f}")
    mac("iccGuesser" + name, f"{e['icc']['guesser']:.2f}")
    mac("iccAuthor" + name, f"{e['icc']['author']:.2f}")
    mac("ownRaw" + name, f"{100 * e['own_raw']:.0f}")
    mac("ownWithin" + name, f"{100 * e['own_within']['effect']:.0f}")
    lo, hi = e["own_within"]["ci"]
    mac("ownWithinCI" + name, f"{100 * lo:.0f}--{100 * hi:.0f}")
labels = {"Blind": "Blind", "Same": "Notes from the same answers", "HeldPerm": "Notes from other prompts",
          "HeldInd": "Notes from other prompts"}
lines = [f"{labels[n]} & {dict(GAMES)[n]['assign']} & {e['icc']['prompt']:.2f} & {e['icc']['guesser']:.2f} & "
         f"{e['icc']['author']:.2f} & {100 * e['own_raw']:+.0f} & {100 * e['own_within']['effect']:+.0f} "
         f"({100 * e['own_within']['ci'][0]:.0f} to {100 * e['own_within']['ci'][1]:.0f}) \\\\" for n, e in effects.items()]
(OUT / "tab_difficulty.tex").write_text(
    "\\begin{tabular}{llrrrrr}\n\\toprule\n& & \\multicolumn{3}{c}{Share of variance (ICC)} & \\multicolumn{2}{c}{Found own answer (points)} \\\\\n"
    "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\nCondition & Naming rule & Prompt & Guesser & Author & Raw & Fixed effects (95\\% CI) \\\\\n"
    "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
CATS = {"writing": "writing", "editing": "editing", "info": "information", "explain": "explanation", "howto": "how-to",
        "advice": "advice", "coding": "coding", "summarize": "summarizing", "brainstorm": "brainstorming", "opinion": "opinion"}
for name in ("Blind", "HeldPerm"):
    by_cat = effects[name]["by_category"]
    hard, easy = min(by_cat, key=by_cat.get), max(by_cat, key=by_cat.get)
    mac("catHard" + name, CATS[hard])
    mac("catHardAcc" + name, pct(by_cat[hard], 0))
    mac("catEasy" + name, CATS[easy])
    mac("catEasyAcc" + name, pct(by_cat[easy], 0))
    low = [CATS[c] for c in sorted(by_cat, key=by_cat.get)[:3]]
    mac("catBottom" + name, f"{low[0]}, {low[1]} and {low[2]}")
mac("bestGuesser", tt(blind["guessers"][0]["model"]))
mac("bestGuesserAcc", pct(blind["guessers"][0]["accuracy"]))
mac("mostRecog", tt(blind["recognizable"][0]["model"]))
mac("mostRecogAcc", pct(blind["recognizable"][0]["recall"]))
selfrates = {r["model"]: r["self_rate"] for r in blind["guessers"]}
self_hi, self_lo = max(selfrates, key=selfrates.get), min(selfrates, key=selfrates.get)
mac("selfHiModel", tt(self_hi))
mac("selfHi", pct(selfrates[self_hi], 0))
mac("selfLoModel", tt(self_lo))
mac("selfLo", pct(selfrates[self_lo], 0))
costs = collections.defaultdict(float)
for r in guess_records:
    if r.get("cost_usd") is not None:
        costs[game_key(r)[0], game_key(r)[2]] += r["cost_usd"]
for name, g in GAMES:
    mac("cost" + name, f"\\${costs[g['mode'], g['assign']]:.0f}")

rows = []
for name, g in GAMES:
    rows.append(f"{labels[name]} & {g['assign']} & {pct(g['accuracy'])} ({ci(g['ci'])}) & {pct(g['vote']['accuracy'])} "
                f"({ci(g['vote']['ci'])}) & {pct(g['vote_weighted']['accuracy'])} & {g['n_calls']:,} \\\\".replace(",", "{,}"))
(OUT / "tab_identification.tex").write_text(
    "\\begin{tabular}{llrrrr}\n\\toprule\nCondition & Naming rule & One guesser (95\\% CI) & Majority of the other five (95\\% CI) "
    "& Weighted vote & Calls \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

rec = {name: {r["model"]: r for r in g["recognizable"]} for name, g in GAMES}
gue = {name: {r["model"]: r for r in g["guessers"]} for name, g in GAMES}
lines = []
for m in models:
    lines.append(f"{tt(m)} & {pct(gue['Blind'][m]['accuracy'], 0)} & {pct(gue['HeldPerm'][m]['accuracy'], 0)} & "
                 f"{pct(rec['Blind'][m]['recall'], 0)} & {pct(rec['HeldPerm'][m]['recall'], 0)} & "
                 f"{pct(gue['Blind'][m]['self_rate'], 0)} \\\\")
(OUT / "tab_recognition.tex").write_text(
    "\\begin{tabular}{lrrrrr}\n\\toprule\n& \\multicolumn{2}{c}{As guesser} & \\multicolumn{2}{c}{As author (named by others)} & Found own \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nModel & Blind & Held-out notes & Blind & Held-out notes & answer (blind) \\\\\n"
    "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- coder reliability and sensitivity
codes = collections.defaultdict(dict)
for c in load_jsonl(DATA / MAIN / "codes.jsonl"):
    if c.get("codes"):
        codes[(c["model"], c["prompt_id"], c["sample"])][c["coder"]] = c["codes"]
coder_a, coder_b = A["meta"]["coders"]
both = [d for d in codes.values() if coder_a in d and coder_b in d]
op_by_id = {p["id"] for p in P if "OP" in p["tags"]}
op_labels = collections.Counter(lab["user_opinion"] for (_, pid, _), d in codes.items() if pid in op_by_id
                                for lab in d.values() if lab["user_opinion"] != "not_applicable")
assert op_labels.most_common(1)[0][0] == "balanced", op_labels
mac("opBalanced", pct(op_labels["balanced"] / sum(op_labels.values()), 0))
kappa = A["agreement"]
names = {"answer_first": "Answers first", "clarification": "Clarification", "premise": "False premise",
         "error_handling": "Pasted error", "user_opinion": "User's opinion", "emotional_ack": "Emotional acknowledgment",
         "unsolicited_advice": "Unasked advice", "moralizing": "Moral commentary", "tone": "Tone", "initiative": "Initiative"}
lines = []
agree, split = {}, {}
for code in CODEBOOK:
    split[code] = sum(1 for d in both if d[coder_a][code] != d[coder_b][code])
    agree[code] = 1 - split[code] / len(both)
for code in sorted(CODEBOOK, key=lambda c: -(kappa.get(c) or 0)):
    lines.append(f"{names[code]} & {kappa[code]:.2f} & {pct(agree[code])} \\\\")
(OUT / "tab_kappa.tex").write_text(
    "\\begin{tabular}{lrr}\n\\toprule\nLabel & Cohen's $\\kappa$ & Raw agreement \\\\\n\\midrule\n" + "\n".join(lines)
    + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
mac("kappaFP", f"{kappa['premise']:.2f}")
mac("kappaERR", f"{kappa['error_handling']:.2f}")
mac("kappaEMO", f"{kappa['emotional_ack']:.2f}")
mac("kappaAF", f"{kappa['answer_first']:.2f}")
mac("kappaMoral", f"{kappa['moralizing']:.2f}")
mac("disagreeAF", pct(split["answer_first"] / len(both)))
mac("disagreeMoral", pct(split["moralizing"] / len(both)))
mac("nDisputed", split["answer_first"] + split["moralizing"])
mac("nBothCoded", f"{len(both):,}".replace(",", "{,}"))


def coder_rate(code, coder, model, positive):
    vals = [d[coder][code] in positive for (m, _, _), d in codes.items() if m == model and coder in d]
    return sum(vals) / len(vals)


af = {c: [coder_rate("answer_first", c, m, {"yes"}) for m in models] for c in (coder_a, coder_b)}
mac("rankAgreeAF", f"{spearmanr(af[coder_a], af[coder_b]).statistic:.2f}")  # tied rates share their average rank
mo = {c: [coder_rate("moralizing", c, m, {"yes"}) for m in models] for c in (coder_a, coder_b)}
mac("rankAgreeMoral", f"{spearmanr(mo[coder_a], mo[coder_b]).statistic:.2f}")
moral = mo[coder_a] + mo[coder_b]
mac("moralLow", pct(min(moral), 0))
mac("moralHigh", pct(max(moral), 0))

# third-coder tie-break (python -m spectra.tiebreak): only the answers the two coders disagree on
tb = json.loads((DATA / MAIN / "tiebreak.json").read_text(encoding="utf-8"))
tb_rows = [r for r in load_jsonl(DATA / MAIN / "tiebreak.jsonl") if r.get("coder") == tb["third"]]
tb_af, tb_mo = tb["labels"]["answer_first"], tb["labels"]["moralizing"]
mac("tbCoder", tt(tb["third"]))
mac("tbLabeled", sum(1 for r in tb_rows if r.get("codes")))
mac("tbAnswers", len({(r["model"], r["prompt_id"], r.get("sample", 0)) for r in tb_rows}))
mac("tbCost", f"\\${sum(r.get('cost_usd') or 0 for r in tb_rows):.2f}")
mac("tbSideAF", " and ".join(f"{tt(c)} {tb_af['sided_with'].get(c, 0)} times" for c in tb["coders"]))
mac("tbRankAF", " and ".join(f"{tb_af['rank_vs_majority'][c]:.2f}" for c in tb["coders"]))
mac("tbSideMoral", " to ".join(str(tb_mo["sided_with"].get(c, 0)) for c in tb["coders"]))
assert set(tb["coders"]) == set(A["meta"]["coders"])

# ---------------------------------------------------------------- codebook table (appendix)
when = {"premise": "FP prompts", "error_handling": "ERR prompts", "user_opinion": "OP prompts",
        "emotional_ack": "EMO prompts", "clarification": "every answer (trait uses AMB prompts)"}
lines = [f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {', '.join(v).replace('_', chr(92) + '_')} & {when.get(k, 'every answer')} \\\\"
         for k, v in CODEBOOK.items()]
(OUT / "tab_codebook.tex").write_text(
    "\\begin{tabular}{lp{0.42\\linewidth}p{0.24\\linewidth}}\n\\toprule\nLabel & Allowed values & Judged on \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- traits table (appendix)
lines = [f"{t['label']} & {t['group']} & {t['desc'].replace('%', chr(92) + '%').replace('&', chr(92) + '&')} \\\\"
         for t in A["traits"]]
(OUT / "tab_traits.tex").write_text(
    "\\begin{tabular}{llp{0.6\\linewidth}}\n\\toprule\nTrait & Layer & Definition \\\\\n\\midrule\n" + "\n".join(lines)
    + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
short_label = {t["key"]: t["short"] for t in A["traits"]}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})  # reruns diff only on content
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# fingerprint: effect sizes, models x traits
keys = [t["key"] for t in A["traits"]]
D = np.array([[T[k]["per_model"][m]["d"] for k in keys] for m in models])
fig, ax = plt.subplots(figsize=(7.0, 2.4))
im = ax.imshow(D, cmap="PuOr_r", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(keys)), [short_label[k] for k in keys], rotation=60, ha="right")
ax.set_yticks(range(len(models)), models)
split = sum(t["group"] == "style" for t in A["traits"]) - 0.5
ax.axvline(split, color="black", lw=0.8)
for i in range(len(models)):
    for j in range(len(keys)):
        ax.text(j, i, f"{D[i, j]:+.1f}", ha="center", va="center", fontsize=5,
                color="white" if abs(D[i, j]) > 1.0 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cb.set_label("effect size $d$ vs. panel (clipped at $\\pm$1.5)")
ax.set_title("Style traits (computed)          |          Behaviors (coded)", fontsize=8)
save(fig, "fig_fingerprint")

# identification by condition
half = A["distinctiveness"]["halves"]
rows = [("Classifier, other half: one answer at a time", half["accuracy"], half["accuracy_ci"], "#BBBBBB"),
        ("Classifier, other half: six answers jointly", half["matched_accuracy"], half["matched_accuracy_ci"], "#BBBBBB"),
        ("One guesser: blind", blind["accuracy"], blind["ci"], "#999999"),
        ("One guesser: notes, same answers", same["accuracy"], same["ci"], "#999999"),
        ("One guesser: notes, other prompts", held["accuracy"], held["ci"], "#0072B2"),
        ("One guesser: other prompts, names may repeat", held_ind["accuracy"], held_ind["ci"], "#9CC3E0"),
        ("Vote of five: blind", blind["vote"]["accuracy"], blind["vote"]["ci"], "#777777"),
        ("Vote of five: notes, same answers", same["vote"]["accuracy"], same["vote"]["ci"], "#777777"),
        ("Vote of five: notes, other prompts", held["vote"]["accuracy"], held["vote"]["ci"], "#56B4E9"),
        ("Vote of five: other prompts, names may repeat", held_ind["vote"]["accuracy"], held_ind["vote"]["ci"], "#BFDDF3")]
fig, ax = plt.subplots(figsize=(5.5, 3.2))
y = np.arange(len(rows))[::-1]
for yi, (label, v, c, color) in zip(y, rows):
    ax.barh(yi, v, color=color, height=0.65)
    if c:
        ax.errorbar(v, yi, xerr=[[v - c[0]], [c[1] - v]], color="black", capsize=2, lw=0.8)
    ax.text((c[1] if c else v) + 0.01, yi, f"{100 * v:.1f}%", va="center", fontsize=7)
ax.axvline(1 / len(models), color="black", ls="--", lw=0.8)
ax.text(1 / len(models) + 0.005, y[0] + 0.55, "chance", fontsize=7)
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlim(0, 1)
ax.set_xlabel(f"share of answers attributed to the right model (run {MAIN})")
save(fig, "fig_identification")

# identification by prompt category
cat_order = sorted(CATS, key=lambda c: effects["HeldPerm"]["by_category"][c])
fig, ax = plt.subplots(figsize=(4.6, 2.4))
for i, c in enumerate(cat_order):
    b, h = effects["Blind"]["by_category"][c], effects["HeldPerm"]["by_category"][c]
    ax.plot([b, h], [i, i], color="#CCCCCC", lw=1, zorder=1)
    ax.scatter(b, i, color="#999999", s=24, zorder=2, label="blind" if i == 0 else None)
    ax.scatter(h, i, color="#0072B2", s=24, zorder=3, label="notes from other prompts" if i == 0 else None)
ax.axvline(1 / len(models), color="black", ls="--", lw=0.8)
ax.set_yticks(range(len(cat_order)), [CATS[c] for c in cat_order])
ax.set_xlim(0, 1)
ax.set_xlabel("one guesser: share of the others' answers named correctly")
ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "fig_category")

# length across runs
order = sorted(models, key=lambda m: val(T["length"], m))
fig, ax = plt.subplots(figsize=(4.6, 1.9))
for i, m in enumerate(order):
    c = COLORS[models.index(m)]
    ax.plot([val(T0["length"], m), val(T["length"], m)], [i, i], color="#CCCCCC", lw=1, zorder=1)
    ax.scatter(val(T0["length"], m), i, facecolors="white", edgecolors=c, s=28, zorder=2)
    ax.scatter(val(T["length"], m), i, color=c, s=28, zorder=3)
ax.set_yticks(range(len(order)), order)
ax.set_xlim(0, 400)
ax.set_xlabel("geometric-mean words per answer")
ax.scatter([], [], facecolors="white", edgecolors="black", s=28, label=REPL)
ax.scatter([], [], color="black", s=28, label=MAIN)
ax.legend(loc="lower right", frameon=False, fontsize=7)
save(fig, "fig_length")

# style map: single answers overlap heavily, model averages do not
pts = np.array([[p[0], p[1], p[2]] for p in A["style_map"]["points"]])
fig, (ax, zoom) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
for j, m in enumerate(models):
    sel = pts[:, 2] == j
    ax.scatter(pts[sel, 0], pts[sel, 1], s=3, alpha=0.3, color=COLORS[j])
    cx, cy = A["style_map"]["centroids"][m]
    for a in (ax, zoom):
        a.scatter(cx, cy, s=60, color=COLORS[j], edgecolors="white", linewidths=1.0, zorder=3, label=m if a is zoom else None)
var = A["style_map"]["variance"]
ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}%): more {', '.join(A['style_map']['axes']['pc1']['pos'])}")
ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)")
ax.set_title("every answer", fontsize=8)
zoom.set_title("model averages only (same axes, zoomed)", fontsize=8)
zoom.set_xlabel("PC1")
zoom.margins(0.25)
zoom.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "fig_stylemap")

# ---------------------------------------------------------------- write the macros
header = (f"% Generated by spectra/article/make_assets.py from runs {MAIN} (main) and {REPL} (replication).\n"
          "% Do not edit by hand: rerun the script.\n")
(OUT / "numbers.tex").write_text(header + "".join(f"\\newcommand{{\\{n}}}{{{v}}}\n" for n, v in macros), encoding="utf-8")
print(f"wrote {len(macros)} macros, {len(list(OUT.glob('tab_*.tex')))} tables, {len(list(OUT.glob('fig_*.pdf')))} figures -> {OUT}")
