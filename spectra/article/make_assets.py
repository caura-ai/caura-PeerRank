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
# the repository, and each pre-registration: the commit that added it against the first record it governs (UTC)
import subprocess  # noqa: E402
from datetime import timezone  # noqa: E402


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


remote = git("remote", "get-url", "origin").removesuffix(".git")
assert remote.startswith("https://github.com/"), remote
mac("repoLink", f"\\url{{{remote}}}")


def first_record(recs):
    return min(datetime.fromisoformat(r["timestamp"]) for r in recs if r.get("timestamp")).astimezone(timezone.utc)


g_all = load_jsonl(DATA / MAIN / "guesses.jsonl")
v_all = load_jsonl(DATA / MAIN / "voice.jsonl")
v2_resp = load_jsonl(DATA / "v2-oct-25-26" / "responses.jsonl")
REGISTRATIONS = [  # file, what it governs, the records it governs
    ("spectra/prompts/prompts_v2.lock", "Second prompt set, locked by hash", v2_resp),
    ("spectra/prereg/2026-10-25-self-knowledge-and-profiles.md", "Self-knowledge replication; profile variants",
     [g for g in g_all if (g.get("profile") or "base") != "base"]),
    ("spectra/prereg/2026-10-25-voice-phase2.md", "Describe, recognize, imitate, hide, contagion; second set",
     [v for v in v_all if v["experiment"] in ("imitate", "hide", "contagion")] + [g for g in g_all if g.get("mode") == "described"] + v2_resp),
    ("spectra/prereg/2026-10-25-contagion-swap.md", "Style swap", [v for v in v_all if v["experiment"] in ("swap-notes", "swap")]),
    ("spectra/prereg/2026-10-25-panel-8.md", "Two more models",
     [r for rev in ("oct-23-26-p8", "v2-oct-25-26-p8") for r in load_jsonl(DATA / rev / "responses.jsonl") if r["model"] in ("grok-4.7", "kimi-k3")]),
    ("spectra/prereg/2026-10-25-replicates.md", "Three answers per prompt",
     [r for r in load_jsonl(DATA / "oct-23-26-s3" / "responses.jsonl") if int(r.get("sample", 0)) > 0]),
]
lines = []
for path, what, recs in REGISTRATIONS:
    h, t = git("log", "--diff-filter=A", "--format=%h %cI", "--", path).splitlines()[-1].split()
    registered, data_from = datetime.fromisoformat(t).astimezone(timezone.utc), first_record(recs)
    assert registered < data_from, (path, registered, data_from)  # registered before any record it governs
    name = path.split("/")[-1].replace("_", "\\_")
    lines.append(f"{what} & \\texttt{{{name}}} & \\texttt{{{h}}} & {registered:%m-%d %H:%M} & {data_from:%m-%d %H:%M} \\\\")
(OUT / "tab_prereg.tex").write_text(
    "\\begin{tabular}{lllrr}\n\\toprule\nRegistration & File & Commit & Committed & First record \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
mac("nRegistrations", len(REGISTRATIONS))
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
# the four full-panel games; profile variants and portrait games (one guesser each) are read separately
games = {(g["mode"], g["assign"]): g for g in summarize_all(guess_records, models, MAIN)
         if g["profile"] == "base" and g["mode"] in ("blind", "briefed", "heldout")}
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
pairs = held["confused_pairs"]
mac("pairsHeldPerm", "; ".join(f"{tt(p['pair'][0])} and {tt(p['pair'][1])} ({pct(p['share'], 0)})" for p in pairs))
mac("pairsShareHeldPerm", pct(sum(p["share"] for p in pairs), 0))
mac("nPairs", ["no", "one", "two", "three"][len(pairs)])
mac("pairsSameVendor", ["No", "One", "Two", "All three"][sum(prov[p["pair"][0]] == prov[p["pair"][1]] for p in pairs)])
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
ratio = round(effects["HeldPerm"]["icc"]["prompt"] / effects["HeldPerm"]["icc"]["guesser"])
mac("iccRatioWord", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][ratio])
labels = {"Blind": "Blind", "Same": "Profile from the same answers", "HeldPerm": "Profile from other prompts",
          "HeldInd": "Profile from other prompts"}
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
    if r.get("cost_usd") is not None and game_key(r)[3] == "base":
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
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nModel & Blind & Held-out profile & Blind & Held-out profile & answer (blind) \\\\\n"
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
agree, split = {}, {}
for code in CODEBOOK:
    split[code] = sum(1 for d in both if d[coder_a][code] != d[coder_b][code])
    agree[code] = 1 - split[code] / len(both)
# A trap label is scored only on the prompts carrying its trap: elsewhere both coders answer "not applicable", and
# counting that agreement would inflate kappa. Clarification is coded everywhere but its trap is the ambiguous prompts.
from spectra.analyze import cohen_kappa  # noqa: E402
TRAP_OF = {"premise": "FP", "error_handling": "ERR", "user_opinion": "OP", "emotional_ack": "EMO", "clarification": "AMB"}
tags_of = {p["id"]: p["tags"] for p in P}
codes_both = [(pid, d) for (_, pid, _), d in codes.items() if coder_a in d and coder_b in d]
tagged = {}
for code, tag in TRAP_OF.items():
    pairs = [(d[coder_a][code], d[coder_b][code]) for pid, d in codes_both if tag in tags_of[pid]]
    a_, b_ = zip(*pairs)
    k = cohen_kappa(a_, b_) if len(set(a_) | set(b_)) > 1 else None  # one label throughout: kappa is undefined
    tagged[code] = {"kappa": k, "agree": sum(x == y for x, y in pairs) / len(pairs), "n": len(pairs), "tag": tag,
                    "top": collections.Counter(a_ + b_).most_common(1)[0]}
lines = []
for code in sorted(CODEBOOK, key=lambda c: -((tagged[c]["kappa"] if c in tagged else kappa.get(c)) or 0)):
    if code in tagged:
        t = tagged[code]
        k = f"{t['kappa']:.2f}" if t["kappa"] is not None else "--"
        lines.append(f"{names[code]} & {t['tag']} ({t['n']}) & {k} & {pct(t['agree'])} \\\\")
    else:
        lines.append(f"{names[code]} & all ({len(both):,}) & {kappa[code]:.2f} & {pct(agree[code])} \\\\".replace(",", "{,}"))
(OUT / "tab_kappa.tex").write_text(
    "\\begin{tabular}{llrr}\n\\toprule\nLabel & Answers scored & Cohen's $\\kappa$ & Raw agreement \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
mac("kappaFP", f"{tagged['premise']['kappa']:.2f}")
mac("kappaERR", f"{tagged['error_handling']['kappa']:.2f}")
mac("kappaOP", f"{tagged['user_opinion']['kappa']:.2f}")
mac("kappaAMB", f"{tagged['clarification']['kappa']:.2f}")
mac("kappaOPAll", f"{kappa['user_opinion']:.2f}")
mac("kappaERRAll", f"{kappa['error_handling']:.2f}")
mac("nFPCoded", tagged["premise"]["n"])
mac("nFPChallenged", sum(1 for pid, d in codes_both if "FP" in tags_of[pid] and d[coder_a]["premise"] == "challenged"))
mac("nEMOCoded", tagged["emotional_ack"]["n"])
assert tagged["emotional_ack"]["kappa"] is None and tagged["emotional_ack"]["agree"] == 1.0
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
    "\\begin{tabular}{l>{\\raggedright\\arraybackslash}p{0.42\\linewidth}>{\\raggedright\\arraybackslash}p{0.24\\linewidth}}\n"
    "\\toprule\nLabel & Allowed values & Judged on \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- traits table (appendix)
lines = [f"{t['label']} & {t['group']} & {t['desc'].replace('%', chr(92) + '%').replace('&', chr(92) + '&')} \\\\"
         for t in A["traits"]]
(OUT / "tab_traits.tex").write_text(
    "\\begin{tabular}{ll>{\\raggedright\\arraybackslash}p{0.6\\linewidth}}\n\\toprule\nTrait & Layer & Definition \\\\\n\\midrule\n"
    + "\n".join(lines)
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
        ("One guesser: profile from the same answers", same["accuracy"], same["ci"], "#999999"),
        ("One guesser: profile from other prompts", held["accuracy"], held["ci"], "#0072B2"),
        ("One guesser: other prompts, names may repeat", held_ind["accuracy"], held_ind["ci"], "#9CC3E0"),
        ("Vote of five: blind", blind["vote"]["accuracy"], blind["vote"]["ci"], "#777777"),
        ("Vote of five: profile from the same answers", same["vote"]["accuracy"], same["vote"]["ci"], "#777777"),
        ("Vote of five: profile from other prompts", held["vote"]["accuracy"], held["vote"]["ci"], "#56B4E9"),
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
    ax.scatter(h, i, color="#0072B2", s=24, zorder=3, label="profile from other prompts" if i == 0 else None)
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

# ================================================================ the accent story (phase 2)
V2 = "v2-oct-25-26"  # the locked second prompt set, collected under the 2026-10-25 voice pre-registration
ACCENT_KEYS = ("emdash", "length", "vocab", "hedging")  # fixed in the registration: style traits with a model share >= 5%
STEREO_KEYS = ("caveats", "warm_open", "stock", "self_ref", "exuberance")  # and <= 2%


def num(x, digits=2):
    s = f"{half_up(x, digits)}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def signed(x, digits=2):
    s = half_up(abs(x), digits)
    return ("$-$" if x < 0 else "+") + s


def span(pair, digits=2, scale=1.0):
    return f"{signed(scale * pair[0], digits)} to {signed(scale * pair[1], digits)}"


def way(x, digits=0):
    """A fraction of the way toward a target, as a percent (0.03 -> 3%)."""
    return pct(x, digits).replace("-", "$-$")


def wayspan(pair, digits=1):
    return f"{way(pair[0], digits)} to {way(pair[1], digits)}"


label_of = {t["key"]: t["short"] for t in A["traits"]}
AC, AC2 = A["accent"], json.loads((DATA / V2 / "analysis.json").read_text(encoding="utf-8"))["accent"]
assert AC["previous"] == REPL and AC2["previous"] == MAIN
rows_ac = {r["key"]: r for r in AC["traits"]}
rows_v2 = {r["key"]: r for r in AC2["traits"]}
mac("accentMedModel", pct(AC["median_model_share"]))
mac("accentMedPrompt", pct(AC["median_prompt_share"], 0))
mac("accentNTraits", len(AC["traits"]))
top = AC["traits"][0]
mac("accentTopTrait", label_of[top["key"]].lower())
mac("accentTopShare", pct(top["model_share"]))
mac("accentLengthShare", pct(rows_ac["length"]["model_share"]))
mac("accentLengthPrompt", pct(rows_ac["length"]["prompt_share"], 0))
mac("accentCaveatShare", pct(rows_ac["caveats"]["model_share"]))
mac("accentEmdashW", num(rows_ac["emdash"]["category_w"]))
mac("accentLengthW", num(rows_ac["length"]["category_w"]))
mac("accentCaveatW", num(rows_ac["caveats"]["category_w"]))
mac("accentWarmW", num(rows_ac["warm_open"]["category_w"]))
rt = [r["run_to_run"] for r in AC["traits"] if r["run_to_run"] is not None]
mac("accentRtMedian", num(statistics.median(rt)))
mac("accentRtHigh", sum(x >= 0.89 for x in rt))
mac("accentRtN", len(rt))
v2_accent = [rows_v2[k]["run_to_run"] for k in ACCENT_KEYS]
mac("vTwoAccentList", ", ".join(f"{label_of[k].lower()} {num(rows_v2[k]['run_to_run'])}" for k in ACCENT_KEYS))
mac("vTwoAccentPass", f"{sum(x >= 0.8 for x in v2_accent)} of {len(ACCENT_KEYS)}")
mac("vTwoMedModel", pct(AC2["median_model_share"]))
mac("vTwoMedPrompt", pct(AC2["median_prompt_share"], 0))
mac("vTwoCaveatRt", num(rows_v2["caveats"]["run_to_run"]))
fam, fam2 = AC["family"], AC2["family"]


def closest(f):
    return min(((v, a, b) for a, row in f["distance"].items() for b, v in row.items() if a < b))


c1, c2 = closest(fam), closest(fam2)
assert {c1[1], c1[2]} == {c2[1], c2[2]}, "the closest pair changed on v2"
mac("familyPair", f"{tt(c1[1])} and {tt(c1[2])}")
mac("familyDist", num(c1[0]))
mac("familyDistTwo", num(c2[0]))
mac("familySame", num(fam["same_vendor"]))
mac("familyDiff", num(fam["different_vendors"]))
cards = AC["cards"]
mac("cardFable", "; ".join(c["text"].lower() for c in cards["claude-fable-5.1"]))
mac("cardAstra", "; ".join(c["text"].lower() for c in cards["gpt-6-astra"][:3]))

# classifier: which layers of the text carry the accent, and how many answers it needs
ATT = json.loads((DATA / MAIN / "attribution.json").read_text(encoding="utf-8"))
ab = ATT["ablations"]
mac("charHalf", pct(ATT["sets"]["characters"]["halves"]["accuracy"]))
mac("charHalfMatched", pct(ATT["sets"]["characters"]["halves"]["matched_accuracy"]))
mac("abOriginal", pct(ab["original"]["accuracy"]))
mac("abNoMarkdown", pct(ab["no markdown"]["accuracy"]))
mac("abNoPunct", pct(ab["no punctuation"]["accuracy"]))
mac("abBare", pct(ab["bare words (also no case or line breaks)"]["accuracy"]))
mac("abShuffled", pct(ab["bare words, shuffled"]["accuracy"]))
mac("abShuffledMatched", pct(ab["bare words, shuffled"]["matched_accuracy"]))
mac("abPunctDrop", half_up(100 * (ab["original"]["accuracy"] - ab["no punctuation"]["accuracy"]), 0))
mac("abShuffleDrop", half_up(100 * (ab["bare words (also no case or line breaks)"]["accuracy"]
                                    - ab["bare words, shuffled"]["accuracy"]), 0))
mac("curveTen", pct(ATT["curve"]["10"]["matched_accuracy"]))
mac("curveTwenty", pct(ATT["curve"]["20"]["matched_accuracy"]))

# describing: the first pre-registration's samples 0-1 and this one's 2-4
from spectra.describe import ESTIMATES  # noqa: E402
mac("nEstimates", len(ESTIMATES))
desc = {0: json.loads((DATA / MAIN / "descriptions.json").read_text(encoding="utf-8"))}
for s in (1, 2, 3, 4):
    desc[s] = json.loads((DATA / MAIN / f"descriptions-s{s}.json").read_text(encoding="utf-8"))
ranks = [desc[s]["tests"]["H1"]["rank_mean"] for s in desc]
mac("descRankLow", num(min(ranks)))
mac("descRankHigh", num(max(ranks)))
mac("descSamples", ["zero", "one", "two", "three", "four", "five"][len(desc)])
mac("descNPerSample", len(desc[1]["describers"]) ** 2)
s1 = desc[1]["tests"]
mac("descCaveat", half_up(s1["H5"]["median_ratio"]["caveats"], 0))
mac("descWarm", half_up(s1["H5"]["median_ratio"]["warm_open"], 0))
mac("descErrors", half_up(s1["H5"]["median_ratio"]["errors"], 1))
mac("descStyleRank", num(s1["H3"]["style"]))
mac("descBehaviorRank", num(s1["H3"]["behavior"]))
mac("descEmGapMeasured", num(s1["H4"]["measured_gap"]))
mac("descEmGapSeen", num(s1["H4"]["seen_gap"]))
h2 = [desc[s]["tests"]["H2"]["self_minus_others"] for s in (1, 2, 3, 4)]
mac("descSelfLow", signed(min(h2)))
mac("descSelfHigh", signed(max(h2)))
mac("descHFiveAll", ["no", "one", "two", "three", "four"][sum(desc[s]["tests"]["H5"]["confirmed"] for s in (1, 2, 3, 4))])
argon = "gemini-4-argon"
pa = desc[1]["per_model"][argon]
mac("argonSelfPlaces", num(pa["self_rank_error"]))
mac("argonOthersPlaces", num(pa["others_rank_error"]))

# the voice experiments (python -m spectra.voice score)
VO = json.loads((DATA / MAIN / "voice.json").read_text(encoding="utf-8"))
E2, E3, E4, E5 = VO["E2"], VO["E3"], VO["E4"], VO["E5"]
r1, r2, r3 = E2["R1_self_below_measured"], E2["R2_self_above_blind"], E2["R3_self_vs_peer"]
mac("portraitSelf", pct(r1["accuracy_a"]))
mac("portraitMeasured", pct(r1["accuracy_b"]))
mac("portraitBlind", pct(r2["accuracy_b"]))
mac("portraitPeer", pct(r3["accuracy_b"]))
mac("portraitVsMeasured", f"{signed(100 * r1['gain'], 1)} points ({span(r1['ci'], 1, 100)})")
mac("portraitVsBlind", f"{signed(100 * r2['gain'], 1)} points ({span(r2['ci'], 1, 100)})")
mac("portraitVsPeer", f"{signed(100 * r3['gain'], 1)} points ({span(r3['ci'], 1, 100)})")
mac("portraitPrompts", r1["prompts"])
ph = E3["posthoc_vs_self"]
mac("imDrift", way(ph["self_drift_toward_target"]["mean"]))
mac("imName", way(ph["name_minus_self"]["mean"]))
mac("imNameCI", wayspan(ph["name_minus_self"]["ci"]))
mac("imProfile", way(ph["profile_minus_self"]["mean"]))
mac("imProfileCI", wayspan(ph["profile_minus_self"]["ci"]))
mac("imRegistered", way(E3["I1_name_moves_toward_target"]["mean"]))
mac("imIThree", pct(E3["I3_moves_right_where_described_wrong"]["mean"], 0))
mac("imIThreeSelf", pct(ph["I3_cells_self"]["mean"], 0))
att = E3["I4_attribution"]
mac("imNamedTarget", pct(att["NAME"]["named_target"]["mean"], 0))
mac("imNamedTargetProfile", pct(att["PROFILE"]["named_target"]["mean"], 0))
mac("imNamedImitator", pct(att["NAME"]["named_imitator"]["mean"], 0))
mac("imNamedBase", pct(ph["self_named_as_target"]["mean"], 0))
acc4 = E4["accuracy"]
mac("hideOriginal", pct(acc4["ORIGINAL"]["mean"], 0))
mac("hideUncoached", pct(acc4["UNCOACHED"]["mean"], 0))
mac("hideUncoachedCI", ci(acc4["UNCOACHED"]["ci"], 0))
mac("hideCoached", pct(acc4["COACHED"]["mean"], 0))
mac("hideDrop", half_up(100 * E4["H3_coaching_hides_better"]["mean"], 0))
mac("hideDropCI", f"{half_up(100 * E4['H3_coaching_hides_better']['ci'][0], 0)}--{half_up(100 * E4['H3_coaching_hides_better']['ci'][1], 0)}")
mac("hideHTwo", signed(E4["H2_stereotype_moves_more_than_accent"]["mean"]))
mac("hideHTwoCI", span(E4["H2_stereotype_moves_more_than_accent"]["ci"]))
c1_ = E5["C1_moves_toward_source"]
mac("conShift", way(c1_["mean"]))
mac("conShiftCI", wayspan(c1_["ci"]))
mac("conProb", half_up(100 * E5["C2_classifier_leans_to_source"]["mean"], 1))
mac("conProbCI", f"{half_up(100 * E5['C2_classifier_leans_to_source']['ci'][0], 1)}--{half_up(100 * E5['C2_classifier_leans_to_source']['ci'][1], 1)}")
per = E5["C3_per_trait"]
mac("conLength", signed(per["length"]["mean"]))
mac("conExcl", signed(per["exuberance"]["mean"]))
mac("conDash", signed(per["emdash"]["mean"]))
mac("conCaveat", signed(per["caveats"]["mean"]))
mac("conWarm", signed(per["warm_open"]["mean"]))
mac("conAccent", signed(E5["C3_accent_vs_stereotype"]["accent"]))
mac("conStereo", signed(E5["C3_accent_vs_stereotype"]["stereotype"]))
mac("conSelf", way(E5["C3_self_notes_amplify"]["mean"]))
mac("conSelfCI", wayspan(E5["C3_self_notes_amplify"]["ci"]))
mac("conSources", ", ".join(tt(s) for s in ("claude-fable-5.1", "gemini-4-argon", "gpt-6-astra")))

# what phase 2 cost, from the records
voice_rows = load_jsonl(DATA / MAIN / "voice.jsonl")
vcost = collections.defaultdict(float)
for r in voice_rows:
    vcost[r["experiment"]] += r.get("cost_usd") or 0
desc_cost = sum(r.get("cost_usd") or 0 for r in load_jsonl(DATA / MAIN / "descriptions.jsonl"))
portrait_cost = sum(r.get("cost_usd") or 0 for r in guess_records if game_key(r)[0] == "described")
v2_cost = sum(r.get("cost_usd") or 0 for r in load_jsonl(DATA / V2 / "responses.jsonl"))
mac("costImitate", f"\\${vcost['imitate']:.0f}")
mac("costHide", f"\\${vcost['hide']:.0f}")
mac("costContagion", f"\\${vcost['contagion']:.0f}")
mac("costDescribe", f"\\${desc_cost:.0f}")
mac("costPortraits", f"\\${portrait_cost:.0f}")
mac("costVTwo", f"\\${v2_cost:.0f}")
mac("nImitate", f"{sum(1 for r in voice_rows if r['experiment'] == 'imitate' and r.get('text')):,}".replace(",", "{,}"))
mac("nHide", sum(1 for r in voice_rows if r["experiment"] == "hide" and r.get("text")))
mac("nContagion", f"{sum(1 for r in voice_rows if r['experiment'] == 'contagion' and r.get('text')):,}".replace(",", "{,}"))

# the profile variants of the first pre-registration (Part B), for the appendix
from spectra.guess import profile_contrasts  # noqa: E402
pc = profile_contrasts(guess_records, models, argon, "heldout", MAIN, "permutation")
names_v = {"portrait": "portraits", "pairs": "notes on confused pairs", "examples": "quoted openings and closings",
           "category": "category notes", "portrait+pairs+examples+category": "all four"}
lines = [f"{names_v[r['profile']]} & {pct(r['accuracy'])} & {pct(r['base_accuracy'])} & {signed(100 * r['gain'], 1)} "
         f"({span(r['ci'], 1, 100)}) \\\\" for r in sorted(pc, key=lambda r: -r["gain"])]
(OUT / "tab_variants.tex").write_text(
    "\\begin{tabular}{lrrr}\n\\toprule\nAdded to the profile & With it & Plain profile & Gain, points (95\\% CI) \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
mac("variantBest", f"{signed(100 * max(r['gain'] for r in pc), 1)}")

# accent table: model and prompt shares, consistency across categories, both replications; and the eight-model panel
A8, A8v2 = (json.loads((DATA / rev / "analysis.json").read_text(encoding="utf-8")) for rev in ("oct-23-26-p8", "v2-oct-25-26-p8"))
assert A8v2["accent"]["previous"] == "oct-23-26-p8"
rows_8 = {r["key"]: r for r in A8["accent"]["traits"]}
rows_8v2 = {r["key"]: r for r in A8v2["accent"]["traits"]}
dash = lambda v: num(v) if v is not None else "--"  # noqa: E731
lines = []
for r in AC["traits"]:
    k = r["key"]
    rt2 = rows_v2.get(k, {}).get("run_to_run")
    lines.append(f"{label_of[k]} & {pct(r['model_share'])} & {pct(r['prompt_share'], 0)} & {num(r['category_w'])} & "
                 f"{dash(r['run_to_run'])} & {dash(rt2)} & "
                 f"{pct(rows_8[k]['model_share']) if k in rows_8 else '--'} & {dash(rows_8v2.get(k, {}).get('run_to_run'))} \\\\")
(OUT / "tab_accent.tex").write_text(
    "\\begin{tabular}{lrrrrrrr}\n\\toprule\n& \\multicolumn{5}{c}{Six models} & \\multicolumn{2}{c}{Eight models} \\\\\n"
    "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}\n"
    "& \\multicolumn{2}{c}{Share per answer} & Same order & \\multicolumn{2}{c}{Order kept (Spearman)} & Model & Order kept \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\n"
    "Trait & Model & Prompt & across categories ($W$) & next run & new prompts & share & new prompts \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
rt8 = [rows_8v2[k]["run_to_run"] for k in rows_8v2 if rows_8v2[k]["run_to_run"] is not None]
mac("pEightRtHigh", sum(x >= 0.8 for x in rt8))
mac("pEightRtN", len(rt8))
mac("pEightRtMedian", num(statistics.median(rt8)))

# figure: the accent, model share against prompt share
fig, ax = plt.subplots(figsize=(4.8, 3.2))
tr = AC["traits"][::-1]
yy = np.arange(len(tr))
ax.barh(yy, [r["prompt_share"] for r in tr], color="#E3E7EE", height=0.7, label="the prompt")
ax.barh(yy, [r["model_share"] for r in tr], color="#0072B2", height=0.7, label="the model")
for yi, r in zip(yy, tr):
    ax.text(r["prompt_share"] + 0.01, yi, f"{100 * r['model_share']:.1f}%", va="center", fontsize=6.5, color="#0072B2")
ax.set_yticks(yy, [label_of[r["key"]] for r in tr])
ax.set_xlim(0, 1)
ax.set_xlabel("share of the variation between single answers explained by")
ax.legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "fig_accent")

# figure: family tree of the panel's voices
from scipy.cluster.hierarchy import dendrogram  # noqa: E402
Z = np.array([[a, b, h, 0] for a, b, h in fam["merges"]], dtype=float)
sizes = {i: 1 for i in range(len(models))}
for k, (a, b, h, _) in enumerate(Z):
    sizes[len(models) + k] = sizes[int(a)] + sizes[int(b)]
    Z[k, 3] = sizes[len(models) + k]
fig, ax = plt.subplots(figsize=(4.2, 1.9))
dendrogram(Z, labels=models, orientation="right", ax=ax, color_threshold=0, above_threshold_color="#444444")
for lab in ax.get_yticklabels():
    lab.set_color(COLORS[models.index(lab.get_text())])
ax.set_xlabel("distance over the everyday traits' effect sizes")
save(fig, "fig_family")

# figure: what the models cannot do, and what measurement lets them do
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax = axes[0]
bars = [("blind", r2["accuracy_b"], "#BBBBBB"), ("self-portraits", r1["accuracy_a"], "#D55E00"),
        ("peer portraits", r3["accuracy_b"], "#E69F00"), ("measured profile", r1["accuracy_b"], "#0072B2")]
for i, (lab, v, col) in enumerate(bars):
    ax.bar(i, v, color=col, width=0.7)
    ax.text(i, v + 0.02, f"{100 * v:.0f}%", ha="center", fontsize=6.5)
ax.set_xticks(range(len(bars)), [b[0] for b in bars], rotation=25, ha="right")
ax.set_ylim(0, 1)
ax.set_title("recognize (one guesser)", fontsize=8)
ax = axes[1]
im_rows = [("from the name", ph["name_minus_self"]), ("from the\nmeasured profile", ph["profile_minus_self"])]
for i, (lab, s) in enumerate(im_rows):
    ax.bar(i, s["mean"], color=["#D55E00", "#0072B2"][i], width=0.6)
    ax.errorbar(i, s["mean"], yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]], color="black", capsize=2, lw=0.8)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(range(2), [r[0] for r in im_rows])
ax.set_ylim(-0.1, 0.6)
ax.set_title("imitate (way to the target,\nbeyond self-imitation)", fontsize=8)
ax = axes[2]
hd = [("original", acc4["ORIGINAL"]), ("hidden", acc4["UNCOACHED"]), ("hidden,\ncoached", acc4["COACHED"])]
for i, (lab, s) in enumerate(hd):
    ax.bar(i, s["mean"], color=["#BBBBBB", "#D55E00", "#0072B2"][i], width=0.6)
    ax.errorbar(i, s["mean"], yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]], color="black", capsize=2, lw=0.8)
ax.axhline(1 / len(models), color="black", ls="--", lw=0.8)
ax.set_xticks(range(3), [h[0] for h in hd])
ax.set_ylim(0, 1)
ax.set_title("hide (classifier still\nnames the author)", fontsize=8)
save(fig, "fig_hearing")

# figure: contagion, per trait
order_c = sorted(per, key=lambda k: per[k]["mean"])
fig, ax = plt.subplots(figsize=(4.4, 2.6))
for i, k in enumerate(order_c):
    s = per[k]
    col = "#0072B2" if k in ACCENT_KEYS else ("#D55E00" if k in STEREO_KEYS else "#999999")
    ax.errorbar(s["mean"], i, xerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]], fmt="o", color=col, ms=4, capsize=2, lw=0.8)
ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(range(len(order_c)), [label_of[k] for k in order_c])
ax.set_xlabel("shift toward the notes' author (SD), against the model's own notes")
ax.scatter([], [], color="#0072B2", label="accent traits")
ax.scatter([], [], color="#D55E00", label="stereotype traits")
ax.legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "fig_contagion")

# ================================================================ the style swap and three answers per prompt
# style swap (2026-10-25-contagion-swap.md): the notes' content crossed with their voice
SW = VO["swap"]
mac("swapVoice", way(SW["voice"]["mean"], 1))
mac("swapVoiceCI", wayspan(SW["voice"]["ci"]))
mac("swapVoiceNoLen", way(SW["voice_without_length"]["mean"], 1))
mac("swapContent", way(SW["content"]["mean"], 1))
mac("swapContentCI", wayspan(SW["content"]["ci"]))
mac("swapSSOO", way(SW["ss_minus_oo"]["mean"], 1))
mac("swapProb", half_up(100 * SW["classifier_voice"]["mean"], 1))
mac("swapProbCI", f"{half_up(100 * SW['classifier_voice']['ci'][0], 1)}--{half_up(100 * SW['classifier_voice']['ci'][1], 1)}")
mc = SW["manipulation_check"]
assert mc["passed"]
mac("swapRewriter", pct(mc["named_rewriter"]))
mac("swapAuthor", pct(mc["named_author"]))
mac("nSwap", f"{sum(1 for r in voice_rows if r['experiment'] == 'swap' and r.get('text')):,}".replace(",", "{,}"))
mac("nSwapRewrites", sum(1 for r in voice_rows if r["experiment"] == "swap-notes" and r.get("text")))
cc = SW["content_check"]  # post hoc: names and numbers kept, added, and imported from the rewriter's own answer
mac("ccSelfKept", pct(cc["self"]["kept"]["mean"], 0))
mac("ccCrossKept", pct(cc["cross"]["kept"]["mean"], 0))
mac("ccSelfNovel", pct(cc["self"]["novel"]["mean"], 0))
mac("ccCrossNovel", pct(cc["cross"]["novel"]["mean"], 0))
mac("ccImported", pct(cc["cross"]["imported"]["mean"]))
mac("ccControl", pct(cc["cross"]["imported_control"]["mean"]))
mac("ccExcess", signed(100 * cc["cross"]["imported_excess"]["mean"], 1))
mac("ccExcessCI", span(cc["cross"]["imported_excess"]["ci"], 1, 100))
mac("costSwap", f"\\${vcost['swap'] + vcost['swap-notes']:.0f}")

# three answers per prompt (2026-10-25-replicates.md, python -m spectra.replicates)
S3 = "oct-23-26-s3"
RP = json.loads((DATA / S3 / "replicates.json").read_text(encoding="utf-8"))
assert RP["samples"] == 3
rp_traits, vc = RP["traits"], RP["median_components"]
mac("repNTraits", RP["n_traits"])
mac("repModel", pct(RP["median_shares"]["model"]))
mac("repModelOne", pct(RP["median_one_answer"]["model"]))
mac("repSig", RP["interaction_significant"])
mac("repSigLater", RP["interaction_significant_later"])
mac("repPExp", int(np.floor(np.log10(max(t["p"] for t in RP["traits"].values())))) + 1)  # every p < 10^this
mac("ssPrompt", pct(RP["median_shares"]["prompt"], 0))
mac("ssReact", pct(RP["median_shares"]["model_x_prompt"], 0))
mac("ssNoise", pct(RP["median_shares"]["noise"], 0))
mac("vcModel", pct(vc["model"]))
mac("vcPrompt", pct(vc["prompt"], 0))
mac("vcReact", pct(vc["model_x_prompt"], 0))
mac("vcNoise", pct(vc["noise"], 0))
mac("vcModelReact", pct(statistics.median(t["components"]["model"] + t["components"]["model_x_prompt"]
                                          for t in rp_traits.values()), 0))
words_n = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
mac("reactRatio", words_n[round(vc["model_x_prompt"] / vc["model"])])
vc_ci, rob = RP["components_ci"]["median"], RP["median_robustness"]
for part, name in (("model", "Model"), ("model_x_prompt", "React"), ("noise", "Noise"), ("prompt", "Prompt")):
    mac(f"vc{name}CI", ci(vc_ci[part], 1))
ratios = [vc["model_x_prompt"] / vc["model"]] + [r["model_x_prompt"] / r["model"] for r in rob.values()]
mac("reactRatioLow", words_n[int(min(ratios))])  # rounded down, so the stated range covers every check
mac("robRanksModel", pct(rob["ranks"]["model"]))
mac("robRanksReact", pct(rob["ranks"]["model_x_prompt"], 0))
mac("robEverydayModel", pct(rob["everyday_only"]["model"]))
mac("robLaterReact", pct(rob["later_answers"]["model_x_prompt"], 0))
mac("bootReps", f"{RP['components_ci']['reps']:,}".replace(",", "{,}"))
rob_names = {"later_answers": "The two same-day answers alone", "everyday_only": "Everyday prompts only",
             "ranks": "Ranks instead of values"}
lines = [f"All three answers (main estimate) & {pct(vc['model'])} & {pct(vc['model_x_prompt'])} & {pct(vc['noise'])} & {pct(vc['prompt'])} \\\\"]
lines += [f"{rob_names[c]} & {pct(r['model'])} & {pct(r['model_x_prompt'])} & {pct(r['noise'])} & {pct(r['prompt'])} \\\\"
          for c, r in rob.items()]
(OUT / "tab_robust.tex").write_text(
    "\\begin{tabular}{lrrrr}\n\\toprule\nMedian over the style traits & Model & Reactions & Noise & Prompt \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
comp = lambda k, part: rp_traits[k]["components"][part]  # noqa: E731
st_model = [comp(k, "model") for k in STEREO_KEYS]
st_react = [comp(k, "model_x_prompt") for k in STEREO_KEYS]
mac("stereoModelLow", pct(min(st_model)))
mac("stereoModelHigh", pct(max(st_model)))
mac("stereoReactLow", pct(min(st_react), 0))
mac("stereoReactHigh", pct(max(st_react), 0))
mac("vcWarmModel", pct(comp("warm_open", "model")))
mac("vcWarmReact", pct(comp("warm_open", "model_x_prompt"), 0))
mac("vcDashModel", pct(comp("emdash", "model"), 0))
mac("vcDashReact", pct(comp("emdash", "model_x_prompt"), 0))
mac("vcLengthPrompt", pct(comp("length", "prompt"), 0))
mac("vcLengthNoise", pct(comp("length", "noise")))
noisiest = sorted(rp_traits, key=lambda k: -comp(k, "noise"))[:3]
mac("vcNoisiest", ", ".join(f"{label_of[k].lower()} ({pct(comp(k, 'noise'), 0)})" for k in noisiest))
cons = RP["consistency"]
hi_c, lo_c = max(cons, key=cons.get), min(cons, key=cons.get)
mac("consHighModel", tt(hi_c))
mac("consHigh", num(cons[hi_c]))
mac("consLowModel", tt(lo_c))
mac("consLow", num(cons[lo_c]))
imr = RP["imitation"]
assert abs(imr["self_drift_single"]["mean"] - ph["self_drift_toward_target"]["mean"]) < 1e-9  # the same E3 baseline
mac("repDriftAvg", way(imr["self_drift_averaged"]["mean"]))
mac("repDriftAvgCI", wayspan(imr["self_drift_averaged"]["ci"]))
mac("repNameAvg", way(imr["name_minus_self_averaged"]["mean"], 1))
mac("repNameAvgCI", wayspan(imr["name_minus_self_averaged"]["ci"]))
rep_rows = [r for r in load_jsonl(DATA / S3 / "responses.jsonl") if int(r.get("sample", 0)) > 0]
day = lambda rows: statistics.median(datetime.fromisoformat(r["timestamp"]).date().toordinal() for r in rows)  # noqa: E731
rep_gap = round(day(rep_rows) - day([r for r in load_jsonl(DATA / S3 / "responses.jsonl") if int(r.get("sample", 0)) == 0]))
mac("repDaysLater", ["no", "one", "two", "three", "four", "five", "six", "seven"][rep_gap])
mac("nReplicates", f"{sum(1 for r in rep_rows if r.get('text') and not r.get('error')):,}".replace(",", "{,}"))
mac("costReplicates", f"\\${sum(r.get('cost_usd') or 0 for r in rep_rows):.0f}")

# table (appendix): each trait's variance components, and the interaction test
order_r = sorted(rp_traits, key=lambda k: -comp(k, "model"))
tci = RP["components_ci"]["traits"]
with_ci = lambda v, band: f"{pct(v)} {{\\scriptsize ({ci(band, 1)})}}"  # noqa: E731
lines = [f"{label_of[k]} & {with_ci(comp(k, 'model'), tci[k]['model'])} & {with_ci(comp(k, 'model_x_prompt'), tci[k]['model_x_prompt'])} & "
         f"{pct(comp(k, 'noise'))} & {pct(comp(k, 'prompt'))} & {num(rp_traits[k]['F'])} & {pct(rp_traits[k]['model'])} \\\\"
         for k in order_r]
lines.append(f"\\midrule\nMedian & {with_ci(vc['model'], vc_ci['model'])} & {with_ci(vc['model_x_prompt'], vc_ci['model_x_prompt'])} & "
             f"{pct(vc['noise'])} & {pct(vc['prompt'])} & & {pct(RP['median_shares']['model'])} \\\\")
(OUT / "tab_reactions.tex").write_text(
    "\\begin{tabular}{lrrrrrr}\n\\toprule\n& \\multicolumn{4}{c}{Share of one answer's variance (post hoc)} & "
    "\\multicolumn{2}{c}{Pre-registered} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\n"
    "Trait & Model & Reactions & Noise & Prompt & $F$ & Model share of SS \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# figure: what one answer is made of, trait by trait
fig, ax = plt.subplots(figsize=(5.2, 3.2))
tr = order_r[::-1]
yy = np.arange(len(tr))
left = np.zeros(len(tr))
for part, col, lab in (("model", "#0072B2", "the model's steady accent"), ("model_x_prompt", "#56B4E9", "its reactions to the prompt"),
                       ("prompt", "#E3E7EE", "the prompt"), ("noise", "#BBBBBB", "noise")):
    w = np.array([comp(k, part) for k in tr])
    ax.barh(yy, w, left=left, color=col, height=0.7, label=lab, hatch="///" if part == "noise" else None, edgecolor="white", lw=0)
    left += w
ax.set_yticks(yy, [label_of[k] for k in tr])
ax.set_xlim(0, 1)
ax.set_xlabel("share of one answer's variance")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
save(fig, "fig_reactions")

# ================================================================ eight models (2026-10-25-panel-8.md, python -m spectra.generalize)
P8 = json.loads((DATA / "oct-23-26-p8" / "panel8.json").read_text(encoding="utf-8"))
assert all(P8[g]["confirmed"] for g in ("G1", "G2", "G3", "G4", "G5"))
models8, new8 = P8["models"], [m for m in P8["models"] if m not in models]
assert new8 == ["grok-4.7", "kimi-k3"]
mac("pEightNew", " and ".join(tt(m) for m in new8))
mac("pEightCoder", tt(P8["coders"][1]))
g1 = P8["G1"]
mac("pEightGOneList", ", ".join(f"{label_of[k].lower()} {num(v)}" for k, v in g1["spearman"].items()))
mac("pEightGOnePass", f"{g1['passing']} of {len(g1['spearman'])}")
mac("pEightMedModel", pct(P8["G2"]["median_model_share"]))
mac("pEightMedModelTwo", pct(P8["G2"]["v2_median_model_share"]))
mac("pEightMedPrompt", pct(P8["G2"]["median_prompt_share"], 0))
for m, name in zip(new8, ("Grok", "Kimi")):
    g3 = P8["G3"][m]
    kept = [t for t in g3["v1"] if t in g3["shared"]]
    mac(f"pEightCard{name}", "; ".join(t.lower() for t in kept))
    mac(f"pEightCardKept{name}", f"{len(kept)} of {len(g3['v1'])}")
t8 = P8["G4"]["tests"]
mac("pEightRank", num(t8["H1"]["rank_mean"]))
mac("pEightCaveat", half_up(t8["H5"]["median_ratio"]["caveats"], 0))
mac("pEightSelf", signed(t8["H2"]["self_minus_others"]))
h4 = t8["H4"]
mac("pEightEmMeasured", num(h4["measured_gap"]))
mac("pEightEmSeen", num(h4["seen_gap"]))
c5 = P8["G5"]["new_readers"]
mac("pEightConNew", way(c5["C1_moves_toward_source"]["mean"]))
mac("pEightConNewCI", wayspan(c5["C1_moves_toward_source"]["ci"]))
mac("pEightConProb", half_up(100 * c5["C2_classifier_leans_to_source"]["mean"], 1))
mac("pEightConAll", way(P8["G5"]["all_eight"]["mean"]))
per8 = P8["contagion_per_reader_source"]
by_reader = collections.defaultdict(list)
for key, v in per8.items():
    by_reader[key.split(" <- ")[0]].append(v["shift_sd"])
most, least = max(by_reader, key=lambda m: np.mean(by_reader[m])), min(by_reader, key=lambda m: np.mean(by_reader[m]))
mac("pEightMostReader", tt(most))
mac("pEightMostRange", f"{num(min(by_reader[most]))} to {num(max(by_reader[most]))}")
mac("pEightLeastReader", tt(least))
mac("pEightLeastRange", f"{num(min(by_reader[least]))} to {num(max(by_reader[least]))}")
others_max = max(max(v) for m, v in by_reader.items() if m != most)
mac("pEightOthersMax", num(others_max))
fam8 = P8["family"]
for m, name in zip(new8, ("Grok", "Kimi")):
    d, nb = fam8["nearest"][m][0]
    mac(f"pEightNear{name}", tt(nb))
    mac(f"pEightNear{name}Dist", num(d))
mac("pEightNewPairDist", num(fam8["grok_kimi"]))
mac("pEightMedianPair", num(fam8["median_pair"]))
assert {fam8["closest_pair"][1], fam8["closest_pair"][2]} == {c1[1], c1[2]}  # still fable and muse
k_new, k_old = P8["kappa_all_answers"]["new_pair"], P8["kappa_all_answers"]["old_pair"]
mac("pEightKappaAF", num(k_new["answer_first"]))
mac("pEightKappaAFOld", num(k_old["answer_first"]))
rhos = [v["rho"] for v in P8["coded_six_old_vs_new"].values() if v["rho"] is not None]
mac("pEightCodedRhoLow", num(min(rhos)))
mac("pEightCodedRhoHigh", num(max(rhos)))
p8_cost = sum(P8[f"data_{s}"]["new_answer_cost"] for s in ("v1", "v2")) + sum(P8["cost"].values())
mac("costPanelEight", f"\\${p8_cost:.0f}")
p8_calls = (sum(1 for rev in ("oct-23-26-p8", "v2-oct-25-26-p8") for r in load_jsonl(DATA / rev / "responses.jsonl")
                if r["model"] in new8 and r.get("text") and not r.get("error"))
            + sum(1 for r in load_jsonl(DATA / "oct-23-26-p8" / "descriptions.jsonl") if r.get("estimates") and not r.get("error"))
            + sum(1 for r in load_jsonl(DATA / "oct-23-26-p8" / "voice.jsonl")
                  if r["experiment"] == "contagion" and r["writer"] in new8 and r.get("text")))
mac("nPanelEightCalls", f"{p8_calls:,}".replace(",", "{,}"))  # new answers on v1 and v2, describing, new readers

# figure: contagion by reader, in SD units along the axis from the reader's accent to the source's
srcs = sorted({k.split(" <- ")[1] for k in per8}, key=models8.index)
readers = sorted(by_reader, key=lambda m: np.mean(by_reader[m]))
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for j, s in enumerate(srcs):
    xs, ys, lo, hi = [], [], [], []
    for i, m in enumerate(readers):
        v = per8.get(f"{m} <- {s}")
        if v:
            xs.append(v["shift_sd"]); ys.append(i + (j - 1) * 0.18)
            lo.append(v["shift_sd"] - v["ci"][0] * v["axis_sd"]); hi.append(v["ci"][1] * v["axis_sd"] - v["shift_sd"])
    ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=3.5, lw=0.7, capsize=1.5, color=COLORS[models8.index(s)],
                label=f"notes by {s}")
ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(range(len(readers)), readers)
for lab in ax.get_yticklabels():
    if lab.get_text() in new8:
        lab.set_fontweight("bold")
ax.set_xlabel("shift toward the notes' author (SD along the axis)")
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
save(fig, "fig_readers")

# the character classifier on the eight models (python -m spectra.attribute --config spectra/config-8.yaml ...)
AT8 = json.loads((DATA / "oct-23-26-p8" / "attribution.json").read_text(encoding="utf-8"))
assert AT8["labels"] == models8
o8 = AT8["ablations"]["original"]
recall8 = dict(zip(models8, o8["recall"]))
mac("charEightAcc", pct(o8["accuracy"]))
mac("charEightMatched", pct(o8["matched_accuracy"]))
mac("chanceEight", pct(1 / len(models8)))
mac("charEightGrok", pct(recall8["grok-4.7"], 0))
mac("charEightKimi", pct(recall8["kimi-k3"], 0))
mac("charEightSixMedian", pct(statistics.median(recall8[m] for m in models), 0))
below_kimi = [m for m in models8 if recall8[m] < recall8["kimi-k3"]]
assert below_kimi == [P8["family"]["nearest"]["kimi-k3"][0][1]], below_kimi  # only kimi's nearest neighbor is named less often
mac("charEightBelowKimi", pct(recall8[below_kimi[0]], 0))

# figure (appendix): the eight-model fingerprint and family tree, from the eight-model analysis
COLORS8 = COLORS + ["#000000", "#882255"]  # the six in roster order, then grok-4.7 and kimi-k3
T8 = {t["key"]: t for t in A8["traits"]}
keys8 = [t["key"] for t in A8["traits"]]
D8 = np.array([[T8[k]["per_model"][m]["d"] for k in keys8] for m in models8])
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.6, 1]})
im = ax.imshow(D8, cmap="PuOr_r", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(keys8)), [short_label[k] for k in keys8], rotation=60, ha="right", fontsize=6)
ax.set_yticks(range(len(models8)), models8, fontsize=7)
for lab in ax.get_yticklabels():
    if lab.get_text() in new8:
        lab.set_fontweight("bold")
ax.axvline(sum(t["group"] == "style" for t in A8["traits"]) - 0.5, color="black", lw=0.8)
for i in range(len(models8)):
    for j in range(len(keys8)):
        ax.text(j, i, f"{D8[i, j]:+.1f}", ha="center", va="center", fontsize=4.2, color="white" if abs(D8[i, j]) > 1.0 else "black")
ax.set_title("Style traits (computed)      |      Behaviors (coded)", fontsize=7)
Z8 = np.array([[a, b, h, 0] for a, b, h in A8["accent"]["family"]["merges"]], dtype=float)
sizes8 = {i: 1 for i in range(len(models8))}
for k, (a, b, h, _) in enumerate(Z8):
    sizes8[len(models8) + k] = sizes8[int(a)] + sizes8[int(b)]
    Z8[k, 3] = sizes8[len(models8) + k]
dendrogram(Z8, labels=models8, orientation="left", ax=ax2, color_threshold=0, above_threshold_color="#444444")  # labels on the right, clear of the heatmap
for lab in ax2.get_yticklabels():
    lab.set_color(COLORS8[models8.index(lab.get_text())])
    lab.set_fontsize(7)
    if lab.get_text() in new8:
        lab.set_fontweight("bold")
ax2.set_xlabel("distance over effect sizes", fontsize=7)
ax2.tick_params(axis="x", labelsize=6)
ax2.set_title("Family tree", fontsize=7)
save(fig, "fig_eight")

# ---------------------------------------------------------------- write the macros
header = (f"% Generated by spectra/article/make_assets.py from runs {MAIN} (main) and {REPL} (replication).\n"
          "% Do not edit by hand: rerun the script.\n")
(OUT / "numbers.tex").write_text(header + "".join(f"\\newcommand{{\\{n}}}{{{v}}}\n" for n, v in macros), encoding="utf-8")
print(f"wrote {len(macros)} macros, {len(list(OUT.glob('tab_*.tex')))} tables, {len(list(OUT.glob('fig_*.pdf')))} figures -> {OUT}")
