"""Spectra unit tests: constraint checks, coded traits, guess parsing and scoring, held-out briefs,
writer locks, provider throttling, and the report's recommendation text.

  pytest tests/
"""
import hashlib
import json
import math
import re
import subprocess
import sys

import pandas as pd
import pytest

import numpy as np

from spectra import SPECTRA_DIR, providers, writer_lock
from spectra.analyze import check_constraint, coded_value, match_by_prompt, prompt_outcomes
from spectra.guess import (build_message, heldout_briefs, heldout_profiles, holdout_halves, parse_assignment,
                           prompt_effects, resolve_name, summarize_all, summarize_guesses)
from spectra.profile import parse_profile
from spectra.report import build_recommendations
from spectra.run import select_prompts

ROSTER = ["gpt-6-astra", "claude-fable-5.1", "claude-opus-5.5", "gemini-4-argon", "gemini-3.8-flash", "muse-wonder-1.0"]


def words(n, word="word"):
    return " ".join([word] * n)


# ----------------------------------------------------------------------------- constraints
class TestConstraints:
    def test_no_constraint_is_nan(self):
        assert math.isnan(check_constraint("anything", None))

    def test_max_words_plain(self):
        con = {"type": "max_words", "value": 30}
        assert check_constraint(words(30), con) == 1.0
        assert check_constraint(words(31), con) == 0.0

    def test_max_words_counts_only_the_draft_between_rules(self):
        text = f"Here's a draft:\n\n---\n\n{words(28)}\n\n---\n\nWant any changes?"
        assert check_constraint(text, {"type": "max_words", "value": 30}) == 1.0

    def test_max_words_labeled_options_each_within_limit(self):
        text = f"Option 1\n{words(20)}\n\nOption 2\n{words(20)}"
        assert check_constraint(text, {"type": "max_words", "value": 25}) == 1.0
        assert check_constraint(text, {"type": "max_words", "value": 15}) == 0.0

    def test_max_words_ignores_a_word_count_note(self):
        text = f"{words(30)}\n\n(30 words)"
        assert check_constraint(text, {"type": "max_words", "value": 30}) == 1.0

    def test_exact_items(self):
        con = {"type": "exact_items", "value": 3}
        assert check_constraint("- one\n- two\n- three", con) == 1.0
        assert check_constraint("- one\n- two\n- three\n- four", con) == 0.0
        assert check_constraint("- one\n  - nested\n- two\n- three", con) == 1.0  # nested items do not count

    def test_no_preamble(self):
        con = {"type": "no_preamble"}
        assert check_constraint("1. Preheat the oven\n2. Mix", con) == 1.0
        assert check_constraint("## Steps\n1. Preheat the oven", con) == 1.0
        assert check_constraint("Sure! Here are the steps:\n1. Preheat", con) == 0.0

    def test_yes_no_first(self):
        con = {"type": "yes_no_first"}
        assert check_constraint("**Yes** — it is safe.", con) == 1.0
        assert check_constraint("No, not really.", con) == 1.0
        assert check_constraint("Short answer: yes.", con) == 0.0

    def test_code_only(self):
        con = {"type": "code_only"}
        assert check_constraint("```python\nprint(1)\n```", con) == 1.0
        assert check_constraint(f"```python\nprint(1)\n```\n\n{words(20)}", con) == 0.0

    def test_table_and_lists(self):
        assert check_constraint("| a | b |\n|---|---|\n| 1 | 2 |", {"type": "table"}) == 1.0
        assert check_constraint("No table here.", {"type": "table"}) == 0.0
        assert check_constraint("Plain paragraph.", {"type": "no_lists"}) == 1.0
        assert check_constraint("Intro\n- item", {"type": "no_lists"}) == 0.0
        assert check_constraint("- a\n- b\n- c", {"type": "list_only"}) == 1.0

    def test_max_sentences_counts_list_items(self):
        con = {"type": "max_sentences", "value": 2}
        assert check_constraint("One. Two.", con) == 1.0
        assert check_constraint("One. Two.\n- three", con) == 0.0


# ----------------------------------------------------------------------------- prompts
def test_select_prompts_round_robins_categories():
    prompts = [{"id": f"{c}{i}", "category": c} for c, n in (("a", 3), ("b", 1), ("c", 2)) for i in range(n)]
    assert [p["id"] for p in select_prompts(prompts, 4)] == ["a0", "b0", "c0", "a1"]
    assert select_prompts(prompts, None) == prompts
    assert len(select_prompts(prompts, 99)) == 6


# ----------------------------------------------------------------------------- guessing game
class TestGuessParsing:
    @pytest.mark.parametrize("raw, expected", [
        ("gpt-6-astra", "gpt-6-astra"),
        ("Gemini 4 Argon", "gemini-4-argon"),
        ("opus", "claude-opus-5.5"),
        ("flash", "gemini-3.8-flash"),
        ("claude", None),   # two Claude models: ambiguous
        ("gemini", None),
        ("gp", None),       # too short to match anything
        (None, None),
    ])
    def test_resolve_name(self, raw, expected):
        assert resolve_name(raw, ROSTER) == expected

    def test_plain_json_in_a_fence(self):
        raw = '```json\n{"A": "gpt-6-astra", "B": "opus"}\n```'
        assert parse_assignment(raw, ["A", "B"], ROSTER) == {"A": "gpt-6-astra", "B": "claude-opus-5.5"}

    def test_prefixed_keys(self):
        raw = '{"Response A": "flash", "RESPONSE B": "muse-wonder-1.0"}'
        assert parse_assignment(raw, ["A", "B"], ROSTER) == {"A": "gemini-3.8-flash", "B": "muse-wonder-1.0"}

    def test_inverted_mapping(self):
        raw = '{"gpt-6-astra": "A", "claude-opus-5.5": "B"}'
        assert parse_assignment(raw, ["A", "B"], ROSTER) == {"A": "gpt-6-astra", "B": "claude-opus-5.5"}

    def test_unmatched_letter_is_none(self):
        assert parse_assignment('{"A": "claude"}', ["A", "B"], ROSTER) == {"A": None, "B": None}

    def test_no_json(self):
        assert parse_assignment("I think A is GPT.", ["A", "B"], ROSTER) is None

    def test_object_sent_back_as_a_json_string(self):
        raw = json.dumps(json.dumps({"A": "gpt-6-astra", "B": "muse-wonder-1.0"}))
        assert parse_assignment(raw, ["A", "B"], ROSTER) == {"A": "gpt-6-astra", "B": "muse-wonder-1.0"}

    def test_escaped_object_inside_prose(self):
        raw = 'My answer: {\\"A\\": \\"gemini-4-argon\\", \\"B\\": \\"gpt-6-astra\\"}'
        assert parse_assignment(raw, ["A", "B"], ROSTER) == {"A": "gemini-4-argon", "B": "gpt-6-astra"}


class TestMessages:
    ORDER = ["gpt-6-astra", "muse-wonder-1.0"]
    TEXTS = {"gpt-6-astra": "one", "muse-wonder-1.0": "two"}

    def test_permutation_wording(self):
        text = build_message("Q?", self.ORDER, self.TEXTS, ROSTER)
        assert "written by a different model" in text and "Use each model once." in text
        assert "Judge only by how the answer is written." in text

    def test_independent_wording(self):
        text = build_message("Q?", self.ORDER, self.TEXTS, ROSTER, independent=True)
        assert "Use each model once" not in text and "different model" not in text
        assert "may be named more than once" in text

    def test_brief_is_included(self):
        text = build_message("Q?", self.ORDER, self.TEXTS, ROSTER, brief="NOTES HERE")
        assert "<research>\nNOTES HERE\n</research>" in text


def test_holdout_halves_split_every_category_evenly():
    prompts = [{"id": f"p{i:03d}", "category": f"c{i % 4}"} for i in range(40)]
    halves = holdout_halves(prompts)
    assert set(halves) == {p["id"] for p in prompts}
    for c in range(4):
        got = [halves[p["id"]] for p in prompts if p["category"] == f"c{c}"]
        assert got.count("A") == got.count("B") == 5
    assert holdout_halves(list(reversed(prompts))) == halves  # fixed by the ids, not the order


# ----------------------------------------------------------------------------- the locked v2 test set
V2_SHA256 = "2c45a7f708fa55b991eaa4425730e59315202eedb865100b81e5c66282c29280"


def test_v2_prompt_set_is_locked():
    data = json.loads((SPECTRA_DIR / "prompts" / "prompts_v2.json").read_text(encoding="utf-8"))
    canon = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    lock = (SPECTRA_DIR / "prompts" / "prompts_v2.lock").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(canon).hexdigest() == lock == V2_SHA256, "prompts_v2.json is locked; do not edit it"


def test_v2_mirrors_the_design_of_v1_with_new_content():
    v1 = json.loads((SPECTRA_DIR / "prompts" / "prompts_v1.json").read_text(encoding="utf-8"))["prompts"]
    v2 = json.loads((SPECTRA_DIR / "prompts" / "prompts_v2.json").read_text(encoding="utf-8"))["prompts"]
    assert len(v2) == len(v1) == 200 and len({p["id"] for p in v2}) == 200
    for a, b in zip(v1, v2):
        assert (a["category"], a["tags"], a["constraint"]) == (b["category"], b["tags"], b["constraint"]), b["id"]
    assert not {p["prompt"] for p in v1} & {p["prompt"] for p in v2}
    assert not {p["id"] for p in v1} & {p["id"] for p in v2}


# ----------------------------------------------------------------------------- guess scoring
GUESSERS = ["alpha-1", "beta-1", "gamma-1"]
LABELS = {"A": "alpha-1", "B": "beta-1", "C": "gamma-1"}


def guess(guesser, a, b, c, prompt="p1", **extra):
    return {"guesser": guesser, "prompt_id": prompt, "sample": 0, "labels": LABELS,
            "guesses": {"A": a, "B": b, "C": c}, **extra}


def test_summarize_guesses_leaves_own_answer_out():
    records = [
        guess("alpha-1", "alpha-1", "beta-1", "alpha-1"),
        guess("beta-1", "gamma-1", "beta-1", "alpha-1", mode="briefed", research="r1"),
    ]
    blind = summarize_guesses(records, GUESSERS)
    assert blind["n_calls"] == 1 and blind["n_judgments"] == 2
    assert blind["chance"] == pytest.approx(1 / 3)
    alpha = next(g for g in blind["guessers"] if g["model"] == "alpha-1")
    assert (alpha["correct"], alpha["n"], alpha["self_rate"]) == (1, 2, 1.0)
    recall = {r["model"]: r["recall"] for r in blind["recognizable"]}
    assert recall == {"beta-1": 1.0, "gamma-1": 0.0, "alpha-1": None}
    briefed = summarize_guesses(records, GUESSERS, ("briefed", "r1", "permutation"))
    assert briefed["n_calls"] == 1 and briefed["research"] == ["r1"]
    assert summarize_guesses(records, GUESSERS, ("briefed", "other", "permutation")) is None


def test_games_are_kept_apart_and_titled():
    records = [
        guess("alpha-1", "alpha-1", "beta-1", "gamma-1"),
        guess("alpha-1", "beta-1", "beta-1", "gamma-1", assign="independent"),
        guess("alpha-1", "alpha-1", "beta-1", "gamma-1", mode="heldout", research="rev", assign="independent"),
        guess("alpha-1", "alpha-1", "gamma-1", "beta-1", mode="briefed", research="rev"),
    ]
    games = summarize_all(records, GUESSERS, "rev")
    assert [g["title"] for g in games] == [
        "Just guess",
        "Just guess, each answer judged on its own",
        "Guess from the research (rev notes)",
        "Guess from held-out research (notes from the other half of the prompts), each answer judged on its own",
    ]
    assert games[1]["repeat_rate"] == 1.0 and games[0]["repeat_rate"] == 0.0
    assert "same set of answers" in games[2]["note"]
    assert "no prompt or answer in front of the guesser" in games[3]["note"]
    # independent games report how often repeats were used, and never claim elimination was ruled out
    assert "100% of calls did" in games[1]["note"] and "0% of calls did" in games[3]["note"]
    assert all("did not narrow" not in g["note"] for g in games)
    assert "Every model had to be used once" in games[0]["note"]


def test_a_retried_call_replaces_the_earlier_one():
    records = [guess("alpha-1", "gamma-1", "alpha-1", "beta-1"), guess("alpha-1", "alpha-1", "beta-1", "gamma-1")]
    assert summarize_guesses(records, GUESSERS)["accuracy"] == 1.0


def test_majority_vote_splits_ties_and_ignores_own_votes():
    records = [
        guess("alpha-1", "alpha-1", "beta-1", "alpha-1"),   # votes: B -> beta, C -> alpha
        guess("beta-1", "alpha-1", "beta-1", "alpha-1"),    # votes: A -> alpha, C -> alpha
        guess("gamma-1", "beta-1", "beta-1", "gamma-1"),    # votes: A -> beta, B -> beta; its own C vote is ignored
    ]
    vote = summarize_guesses(records, GUESSERS)["vote"]
    # A (alpha): alpha vs beta tie -> 0.5; B (beta): 2 of 2 -> 1; C (gamma): both say alpha -> 0
    assert vote["accuracy"] == pytest.approx(0.5)
    assert (vote["n"], vote["ties"], vote["voters"]) == (3, 1, 2)


def test_skill_weighted_vote_trusts_the_guesser_that_is_right_on_the_other_half():
    # alpha names every other answer correctly; beta and gamma name every other answer wrongly.
    records = []
    for i in range(8):
        records += [guess("alpha-1", "alpha-1", "beta-1", "gamma-1", prompt=f"p{i}", category="c"),
                    guess("beta-1", "gamma-1", "beta-1", "alpha-1", prompt=f"p{i}", category="c"),
                    guess("gamma-1", "beta-1", "alpha-1", "gamma-1", prompt=f"p{i}", category="c")]
    s = summarize_guesses(records, GUESSERS)
    # plain vote: alpha's answer is a 1-1 tie between two wrong names, the others are ties with the right name
    assert s["vote"]["accuracy"] == pytest.approx(1 / 3)
    # weighted: beta and gamma weigh nothing, alpha places two answers and the third follows, each model once
    assert s["vote_weighted"]["accuracy"] == 1.0 and s["vote_weighted"]["n"] == 8 * 3
    assert s["by_category"] == {"c": pytest.approx(1 / 3)}
    assert s["confused_pairs"][0]["share"] > 0


def test_intervals_and_the_own_answer_effect():
    records = []
    for i in range(30):
        found = i % 2 == 0  # alpha finds its own answer on even prompts and then gets the others right
        records.append(guess("alpha-1", "alpha-1" if found else "beta-1", "beta-1" if found else "gamma-1",
                             "gamma-1" if found else "alpha-1", prompt=f"p{i}"))
    s = summarize_guesses(records, GUESSERS)
    lo, hi = s["ci"]
    assert lo <= s["accuracy"] <= hi and lo < hi
    assert s["own_answer_effect"]["found"] == 1.0 and s["own_answer_effect"]["not_found"] == 0.0
    assert "when they found their own answer" in s["note"]


def test_prompt_effects_tell_easy_prompts_from_finding_your_own():
    # Correctness depends only on the prompt; guessers just find their own answer more often on easy prompts.
    records = []
    for i in range(40):
        easy = i % 2 == 0
        for j, g in enumerate(GUESSERS):
            found = j < (2 if easy else 1)
            guesses = {}
            for letter, author in LABELS.items():
                right = found if author == g else easy
                guesses[letter] = author if right else next(m for m in GUESSERS if m != author)
            records.append({"guesser": g, "prompt_id": f"p{i}", "sample": 0, "labels": LABELS,
                            "guesses": guesses, "category": "c"})
    e = prompt_effects(records, GUESSERS, reps=50)
    assert e["n_judgments"] == 40 * 3 * 2 and e["n_prompts"] == 40
    assert e["own_raw"] == pytest.approx(1 / 3)  # looks like finding your own answer helps...
    assert e["own_within"]["effect"] == pytest.approx(0.0, abs=1e-9)  # ...until the prompt is held fixed
    assert e["icc"]["prompt"] > 0.9 and e["icc"]["guesser"] == 0.0
    assert e["by_category"] == {"c": 0.5}
    assert prompt_effects(records, GUESSERS, ("heldout", "x", "permutation")) is None


# ----------------------------------------------------------------------------- held-out briefs end to end
def test_heldout_briefs_never_see_the_prompts_they_brief(tmp_path):
    prompts = json.loads((SPECTRA_DIR / "prompts" / "prompts_v1.json").read_text(encoding="utf-8"))["prompts"]
    picked = [p for c in sorted({p["category"] for p in prompts})
              for p in [q for q in prompts if q["category"] == c][:4]]  # 40 prompts, 4 per category
    styles = {"alpha-1": lambda i: f"Short answer {i}. " + "Plain words here. " * 12,
              "beta-1": lambda i: f"## Plan {i}\n" + "".join(f"- **step {j}** do the thing carefully\n" for j in range(8)),
              "gamma-1": lambda i: f"Honestly, it depends {i} — you might perhaps try this. " * 10}
    rows = [{"model": m, "prompt_id": p["id"], "category": p["category"], "sample": 0, "text": f(i)}
            for i, p in enumerate(picked) for m, f in styles.items()]
    (tmp_path / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"revision": "t", "prompts": str(SPECTRA_DIR / "prompts" / "prompts_v1.json"),
                                  "models": list(styles)}), encoding="utf-8")  # JSON is valid YAML
    halves = holdout_halves(prompts)
    briefs = heldout_briefs(str(config), tmp_path, "t", list(styles), halves)
    assert set(briefs) == {"A", "B"} and briefs["A"] != briefs["B"]
    in_b = sum(halves[p["id"]] == "B" for p in picked)  # halves are fixed over the full prompt set
    assert f"Source: Spectra revision t-half-B, {in_b} prompts" in briefs["A"]  # half A is briefed from half B
    for half in "AB":
        analysis = json.loads((tmp_path / "holdout" / f"half-{half}" / "analysis.json").read_text(encoding="utf-8"))
        assert {p["id"] for p in analysis["viewer"]["prompts"]} <= {pid for pid, h in halves.items() if h == half}
        dist = analysis["distinctiveness"]
        assert {"matched_accuracy", "matched_accuracy_ci", "style_only_matched_accuracy"} <= set(dist)
        assert {"accuracy", "matched_accuracy"} <= set(dist["halves"])


def _two_half_run(tmp_path):
    """40 v1 prompts (4 per category) answered by three models whose text carries the prompt's index."""
    prompts = json.loads((SPECTRA_DIR / "prompts" / "prompts_v1.json").read_text(encoding="utf-8"))["prompts"]
    picked = [p for c in sorted({p["category"] for p in prompts})
              for p in [q for q in prompts if q["category"] == c][:4]]
    styles = {"alpha-1": lambda i: f"Short answer {i}. " + "Plain words here. " * 12,
              "beta-1": lambda i: f"## Plan {i}\n" + "".join(f"- **step {j}** do the thing carefully\n" for j in range(8)),
              "gamma-1": lambda i: f"Honestly, it depends {i} — you might perhaps try this. " * 10}
    rows = [{"model": m, "prompt_id": p["id"], "category": p["category"], "sample": 0, "text": f(i)}
            for i, p in enumerate(picked) for m, f in styles.items()]
    (tmp_path / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"revision": "t", "prompts": str(SPECTRA_DIR / "prompts" / "prompts_v1.json"),
                                  "models": list(styles)}), encoding="utf-8")  # JSON is valid YAML
    return str(config), picked, list(styles), holdout_halves(prompts)


def test_profile_names_are_canonical():
    assert parse_profile(None) == parse_profile("base") == "base"
    assert parse_profile("category,portrait") == parse_profile("portrait+category") == "portrait+category"
    assert parse_profile("all") == "portrait+pairs+examples+category"
    with pytest.raises(ValueError):
        parse_profile("examples,vibes")


def test_each_profile_is_its_own_game():
    records = [guess("alpha-1", "alpha-1", "beta-1", "gamma-1", mode="heldout", research="r"),
               guess("alpha-1", "alpha-1", "gamma-1", "beta-1", mode="heldout", research="r", profile="examples")]
    games = summarize_all(records, GUESSERS, "r")
    assert [(g["profile"], g["accuracy"]) for g in games] == [("base", 1.0), ("examples", 0.0)]
    assert games[1]["title"].endswith("profile with examples")
    assert summarize_guesses(records, GUESSERS, ("heldout", "r", "permutation"))["accuracy"] == 1.0  # old keys = base


def test_profile_parts_come_only_from_the_half_that_built_them(tmp_path):
    config, picked, models, halves = _two_half_run(tmp_path)
    plain = heldout_briefs(config, tmp_path, "t", models, halves)
    full = heldout_profiles(config, tmp_path, "t", models, halves, parse_profile("all"))
    for half, other in (("A", "B"), ("B", "A")):
        cat = next(p["category"] for p in picked if halves[p["id"]] == half)  # a prompt this half's profile is used on
        text = full[half].text(cat)
        assert full[half].base == plain[half] and text.startswith("Portraits, one per model")
        assert text.index("Portraits") < text.index("Measured habits") < text.index("How to tell apart")
        assert "Typical openings and closings" in text and f"In {cat} prompts specifically" in text
        quoted = {int(n) for n in re.findall(r"(?:answer|Plan|depends) (\d+)", full[half].examples)}
        assert quoted, full[half].examples
        assert all(halves[picked[i]["id"]] == other for i in quoted), "an example came from a prompt being guessed"
    assert heldout_profiles(config, tmp_path, "t", models, halves)["A"].text("writing") == plain["A"]


def test_tiebreak_takes_the_majority_of_three_on_disputed_labels_only():
    from spectra.tiebreak import disputed, latest_codes, summarize

    rows = []
    for i in range(4):  # coder x and y agree on m1; on m2 they split twice, the third coder sides with y once, x once
        for coder in ("x", "y"):
            m1, m2 = "yes", ("yes" if coder == "x" else "no") if i < 2 else "no"
            rows += [{"coder": coder, "model": "m1", "prompt_id": f"p{i}", "sample": 0, "codes": {"answer_first": m1}},
                     {"coder": coder, "model": "m2", "prompt_id": f"p{i}", "sample": 0, "codes": {"answer_first": m2}}]
    codes = latest_codes(rows)
    split = disputed(codes, ["x", "y"], ["answer_first"])
    assert split == {("m2", "p0", 0): ["answer_first"], ("m2", "p1", 0): ["answer_first"]}
    ties = {("m2", "p0", 0): {"answer_first": "no"}, ("m2", "p1", 0): {"answer_first": "yes"}}
    s = summarize(codes, ties, ["x", "y"], "z", ["answer_first"], ["m1", "m2"], {"answer_first": {"yes"}})["labels"]["answer_first"]
    assert s["disputed"] == 2 and s["sided_with"] == {"y": 1, "x": 1}
    assert s["rates"]["majority"] == {"m1": 1.0, "m2": 0.25} and s["rates"]["x"]["m2"] == 0.5 and s["rates"]["y"]["m2"] == 0.0


def test_descriptions_parse_and_score_against_measured_traits():
    from spectra.describe import parse_reply, score

    raw = '```json\n{"profile": "Short and direct.", "estimates": {"words": "120 words", "pct_answers_first": 90}}\n```'
    assert parse_reply(raw) == {"profile": "Short and direct.", "estimates": {"words": 120.0, "pct_answers_first": 90.0}}
    assert parse_reply("no json here") is None
    models = ["m1", "m2", "m3"]
    truth = {"length": {"m1": 100.0, "m2": 200.0, "m3": 400.0}, "answer_first": {"m1": 0.9, "m2": 0.7, "m3": 0.5}}
    analysis = {"traits": [{"key": k, "per_model": {m: {"value": v} for m, v in vals.items()}} for k, vals in truth.items()]}
    records = [{"describer": d, "target": t, "estimates": {"words": truth["length"][t],
                                                          "pct_answers_first": 100 * truth["answer_first"][t]}}
               for d in models for t in models]
    vendors = {"m1": "a", "m2": "a", "m3": "b"}
    full = score(records, analysis, models, vendors)
    s = full["overall"]
    assert s["rank_mean"] == pytest.approx(1.0) and s["self_rank_error"] == 0.0
    assert s["self_side"] == s["others_side"] == 1.0
    sib = full["siblings"]["same vendor"]  # perfect describers see the m1-m2 gap exactly as measured
    assert sib["seen_gap"] == pytest.approx(sib["measured_gap"]) and sib["sign_right"] == 1.0
    for r in records:  # m1 now thinks it writes the longest answers; the others still describe it correctly
        if r["describer"] == r["target"] == "m1":
            r["estimates"]["words"] = 800.0
    per = score(records, analysis, models, vendors)["per_model"]["m1"]
    assert per["self_rank_error"] > per["others_rank_error"] == 0.0 and per["self_side"] < per["others_side"]


def test_a_later_description_sample_only_reorders_the_request():
    from spectra.describe import build_message as describe_message

    models = [f"m{i}" for i in range(6)]
    first = describe_message("m0", "m3", models)
    assert describe_message("m0", "m3", models, 0) == first  # sample 0 is the original message, byte for byte
    later = describe_message("m0", "m3", models, 1)
    assert later == describe_message("m0", "m3", models, 1) and later != first
    fields = lambda text: sorted(line for line in text.splitlines() if line.startswith('  "'))  # noqa: E731
    assert fields(later) == fields(first)  # the same estimates asked for, in another order
    panel = lambda text: re.findall(r"m\d", next(l for l in text.splitlines() if l.startswith("The models")))  # noqa: E731,E741
    assert panel(later) != panel(first) and sorted(panel(later)) == sorted(panel(first)) == models


def test_preregistered_tests_on_perfect_describers():
    from spectra.describe import confirm, score

    models = [f"m{i}" for i in range(6)]
    truth = {"length": dict(zip(models, (100.0, 150.0, 200.0, 300.0, 400.0, 600.0))),
             "answer_first": dict(zip(models, (0.9, 0.4, 0.7, 0.5, 0.8, 0.3)))}
    analysis = {"traits": [{"key": k, "per_model": {m: {"value": v} for m, v in vals.items()}} for k, vals in truth.items()]}
    records = [{"describer": d, "target": t, "estimates": {"words": truth["length"][t],
                                                          "pct_answers_first": 100 * truth["answer_first"][t]}}
               for d in models for t in models]
    tests = confirm(records, analysis, models, reps=2000)
    assert tests["H1"]["rank_mean"] == pytest.approx(score(records, analysis, models)["overall"]["rank_mean"])
    assert tests["H1"]["p"] < 0.05 and tests["H1"]["confirmed"]
    assert tests["H2"]["self_minus_others"] == 0.0 and tests["H2"]["confirmed"]
    for r in records:  # every model now ranks itself last on both traits; the others still describe it correctly
        if r["describer"] == r["target"]:
            r["estimates"] = {"words": 1.0, "pct_answers_first": 0.0}
    h2 = confirm(records, analysis, models, reps=2000)["H2"]
    assert h2["self_minus_others"] > 0 and not h2["confirmed"] and not h2["self_advantage"]


def test_profile_contrasts_pair_each_variant_with_the_plain_profile():
    from spectra.guess import parse_guessers, profile_contrasts

    models = ["a", "b", "c"]
    assert parse_guessers(None, models) == models and parse_guessers("c, a", models) == ["a", "c"]
    with pytest.raises(ValueError):
        parse_guessers("a,zed", models)

    def rec(pid, profile, right):
        r = {"guesser": "a", "prompt_id": pid, "mode": "heldout", "research": "rev", "assign": "permutation",
             "labels": {"A": "a", "B": "b", "C": "c"},
             "guesses": {"A": "a", "B": "b" if right else "c", "C": "c" if right else "b"}}
        if profile != "base":
            r["profile"] = profile
        return r

    records = [rec(f"q{i}", "base", i % 2 == 0) for i in range(10)] + [rec(f"q{i}", "pairs", True) for i in range(6)]
    rows = profile_contrasts(records, models, "a", "heldout", "rev", "permutation")
    assert [r["profile"] for r in rows] == ["pairs"]
    row = rows[0]
    assert row["prompts"] == 6 and row["accuracy"] == 1.0 and row["base_accuracy"] == 0.5
    assert row["gain"] == 0.5 and row["ci"][0] > 0


def test_character_attribution_learns_punctuation_habits():
    from spectra.analyze import features
    from spectra.attribute import across, attribute

    rows = []
    for i in range(40):
        for m, text in (("a", f"Sure! Here is item {i} — quick and done."),
                        ("b", f"Certainly. Item {i}; as requested, finished.")):
            f, prose = features(text, "explain")
            f.update(model=m, prompt_id=f"q{i}", sample=0, category="explain" if i % 2 else "code", prose=prose)
            rows.append(f)
    df = pd.DataFrame(rows)
    res = attribute(df, ["a", "b"], sets=("characters",))["characters"]
    assert res["cv"]["accuracy"] == 1.0 and res["halves"]["matched_accuracy"] == 1.0
    assert across(df, df, ["a", "b"], sets=("characters",))["characters"]["accuracy"] == 1.0


def test_text_ablations_remove_one_layer_each():
    from spectra.attribute import ABLATIONS, bare_words, strip_markdown, strip_punctuation

    text = ("### Example\n**Compound interest** is *interest* on interest — see [the guide](http://x.y).\n\n"
            "| Year | Balance |\n|---|---:|\n| 1 | $1,050 |\n- one item\n2. two items\n> quoted")
    md = strip_markdown(text)
    assert md.splitlines()[:2] == ["Example", "Compound interest is interest on interest — see the guide."]
    assert "|" not in md and "**" not in md and "---" not in md and "one item" in md and "two items" in md
    assert strip_punctuation("Well — it's fine!") == "Well it s fine"
    assert bare_words("Hi!\n\nThere, *you*.") == "hi there you"
    assert ABLATIONS["bare words, shuffled"]("b a c") == ABLATIONS["bare words, shuffled"]("b a c")  # deterministic
    assert sorted(ABLATIONS["bare words, shuffled"]("b a c").split()) == ["a", "b", "c"]


def test_learning_curve_trains_on_n_prompts_per_model():
    from spectra.analyze import features
    from spectra.attribute import learning_curve

    rows = []
    for i in range(40):
        for m, text in (("a", f"Sure! Here is item {i} — quick and done."),
                        ("b", f"Certainly. Item {i}; as requested, finished.")):
            f, prose = features(text, "explain")
            f.update(model=m, prompt_id=f"q{i}", sample=0, category="explain", prose=prose)
            rows.append(f)
    curve = learning_curve(pd.DataFrame(rows), ["a", "b"], sizes=(4, 32), repeats=2)
    assert curve["4"]["draws"] == 2 and curve["4"]["answers_per_model"] == 4
    assert curve["32"]["draws"] == 1 and curve["32"]["answers_per_model"] == 32
    assert curve["32"]["matched_accuracy"] == 1.0


def test_accent_separates_the_model_from_the_prompt():
    from spectra.analyze import accent, kendall_w

    assert kendall_w([[1, 2, 3], [1, 2, 3]]) == 1.0 and kendall_w([[1, 2, 3], [3, 2, 1]]) == 0.0
    models = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        base = rng.normal(0, 3)  # each prompt sets most of the level
        for j, m in enumerate(models):
            rows.append({"model": m, "prompt_id": f"q{i}", "category": "explain" if i % 2 else "code",
                         "emdash_per100": base + 0.8 * j + rng.normal(0, 0.1),  # a small, steady accent
                         "hedges_per100": base + rng.normal(0, 1)})  # no accent at all
    df = pd.DataFrame(rows)

    def trait(key, col):
        means = df.groupby("model")[col].mean()
        sd = df.groupby("model")[col].std().mean()
        return {"key": key, "label": key, "high": "More " + key, "low": "Less " + key,
                "per_model": {m: {"value": float(means[m]), "d": float((means[m] - means.mean()) / sd)} for m in models}}

    traits = [trait("emdash", "emdash_per100"), trait("hedging", "hedges_per100"),
              dict(trait("hedging", "hedges_per100"), key="premise", tag="FP")]  # trap traits are left out
    previous = {"meta": {"revision": "old"}, "traits": [dict(t) for t in traits]}
    out = accent(df, models, traits, previous, vendors={"a": "x", "b": "x", "c": "y"})
    rows_by = {r["key"]: r for r in out["traits"]}
    assert set(rows_by) == {"emdash", "hedging"} and out["previous"] == "old"
    em, he = rows_by["emdash"], rows_by["hedging"]
    assert em["prompt_share"] > em["model_share"] > he["model_share"]  # small next to the prompt, yet far from zero
    assert em["category_w"] == 1.0 and em["run_to_run"] == 1.0
    fam = out["family"]
    assert sorted(fam["order"]) == models and len(fam["merges"]) == 2 and fam["same_vendor"] is not None
    assert [c["text"] for c in out["cards"]["c"]][:1] == ["More emdash"] and out["cards"]["c"][0]["categories"] == [2, 2]


def test_portraits_brief_self_or_peer_and_never_the_guessers_words():
    from spectra.guess import Portraits

    models = ["a", "b", "c", "d"]
    records = [{"describer": d, "target": t, "profile": f"{d} on {t}", "sample": 1} for d in models for t in models]
    records.append({"describer": "a", "target": "a", "profile": "sample zero", "sample": 0})
    own = Portraits(records, models, "self")
    assert own.authors == {m: m for m in models} and "- a: a on a" in own.text("any") and "sample zero" not in own.base
    peer = Portraits(records, models, "peer", skip=("b",))  # b guesses, so it never wrote a portrait it reads
    assert peer.authors == {"a": "c", "b": "c", "c": "d", "d": "a"}
    with pytest.raises(ValueError):
        Portraits(records, models, "self", sample=2)
    plain = build_message("q", ["a", "b"], {"a": "x", "b": "y"}, ["a", "b"], brief="notes")
    told = build_message("q", ["a", "b"], {"a": "x", "b": "y"}, ["a", "b"], brief="notes", brief_kind="portraits")
    assert "<research>" in plain and "<portraits>" in told and "not measurements" in told
    assert plain == build_message("q", ["a", "b"], {"a": "x", "b": "y"}, ["a", "b"], brief="notes", brief_kind="research")


def test_game_contrast_pairs_any_two_games_on_shared_prompts():
    from spectra.guess import game_contrast

    def rec(pid, mode, right, research=None):
        return {"guesser": "a", "prompt_id": pid, "mode": mode, "research": research, "assign": "permutation",
                "labels": {"A": "a", "B": "b", "C": "c"},
                "guesses": {"A": "a", "B": "b" if right else "c", "C": "c" if right else "b"}}

    records = ([rec(f"q{i}", "described", True, "self, sample 1") for i in range(8)]
               + [rec(f"q{i}", "blind", i < 2) for i in range(10)])
    out = game_contrast(records, ["a", "b", "c"], "a", ("described", "self, sample 1", "permutation"),
                        ("blind", None, "permutation"))
    assert out["prompts"] == 8 and out["accuracy_a"] == 1.0 and out["accuracy_b"] == 0.25 and out["ci"][0] > 0
    assert game_contrast(records, ["a", "b", "c"], "a", ("heldout", "x", "permutation"), ("blind", None, "permutation")) is None


def test_config_variant_extends_the_main_config(tmp_path):
    from spectra import load_config

    (tmp_path / "base.yaml").write_text("revision: r1\nprompts: p1.json\nmodels: [m1, m2]\nsamples: 3\n", encoding="utf-8")
    (tmp_path / "v2.yaml").write_text("extends: base.yaml\nprompts: p2.json\nrevision: null\n", encoding="utf-8")
    cfg = load_config(tmp_path / "v2.yaml")
    assert cfg["prompts"].endswith("p2.json") and cfg["revision"] is None and cfg["samples"] == 3
    assert [m["name"] for m in cfg["models"]] == ["m1", "m2"]
    real = load_config(SPECTRA_DIR / "config-v2.yaml")
    assert real["prompts"].endswith("prompts_v2.json") and real["models"] == load_config()["models"]


def _voice_fixture():
    """Six prompts per category over two categories, three models with distinct, steady habits."""
    from spectra import voice

    models = ["a", "b", "c"]
    prompts = [{"id": f"{c}{i}", "category": c, "prompt": f"Explain thing {c}{i}.", "tags": [], "constraint": None}
               for c in ("explain", "advice") for i in range(8)]
    prompts.append({"id": "trap", "category": "explain", "prompt": "Trap?", "tags": ["FP"], "constraint": None})
    style = {"a": lambda i: "Short answer here. " * 3,
             "b": lambda i: ("A longer answer — with dashes — that goes on and on. " * 8),
             "c": lambda i: "## Heading\n\n- point one\n- point two\n- point three\n\n**Bold** ending."}
    texts = {(m, p["id"]): style[m](i) + f" item {i}" for m in models for i, p in enumerate(prompts)}
    traits = [{"key": "length", "label": "Length", "high": "Writes long answers", "low": "Keeps answers short",
               "per_model": {"a": {"value": 10, "d": -1.0}, "b": {"value": 80, "d": 1.0}, "c": {"value": 20, "d": 0.1}}}]
    analysis = {"traits": traits, "cards": {}, "signature": {}, "accent": {"cards": {"a": [{"text": "Keeps answers short"}]}},
                "runstats": {m: {"total_cost_usd": 1.0, "responses": 100} for m in models}}
    return voice, models, prompts, texts, analysis


def test_voice_jobs_use_the_same_everyday_prompts_for_every_model(monkeypatch):
    voice, models, prompts, texts, analysis = _voice_fixture()
    picked = voice.pick_prompts(prompts, 3, "imitate")
    assert len(picked) == 6 and "trap" not in {p["id"] for p in picked}
    assert picked == voice.pick_prompts(prompts, 3, "imitate") != voice.pick_prompts(prompts, 3, "hide")
    monkeypatch.setattr(voice, "SOURCES", ("b",))  # the module's sources are real panel names
    monkeypatch.setattr(voice.providers, "input_cost", lambda entry, tokens: tokens * 1e-6)
    jobs = {e: voice.jobs_for(e, models, prompts, texts, analysis) for e in voice.EXPERIMENTS}
    n_imitate = 3 * 2 * len(models) * (1 + 2 * (len(models) - 1))
    assert len(jobs["imitate"]) == n_imitate and len(jobs["hide"]) == 5 * 2 * len(models) * 2
    profile_msgs = [j["message"] for j in jobs["imitate"] if j["condition"] == "PROFILE"]
    assert all("Measured habits of" in m for m in profile_msgs)
    assert all("way you" not in j["message"] for j in jobs["imitate"])
    coached = next(j for j in jobs["hide"] if j["condition"] == "COACHED" and j["writer"] == "a")["message"]
    assert "keeps answers short" in coached and texts[("a", next(j for j in jobs["hide"] if j["writer"] == "a")["prompt_id"])] in coached
    notes = {p["id"] for p in voice.note_prompts(prompts)}
    con = jobs["contagion"]
    assert not notes & {j["prompt_id"] for j in con}
    assert {j["condition"] for j in con} == {"NONE", "SELF", "SOURCE"}
    assert all("imitat" not in j["message"] and "style" not in j["message"] for j in con)
    source = next(j for j in con if j["condition"] == "SOURCE" and j["writer"] == "a")
    assert source["other"] == "b" and texts[("b", sorted(notes)[0])] in source["message"]
    cost, by = voice.estimate(jobs["hide"], analysis, {p["id"]: p for p in prompts})
    assert cost >= 0.01 * len(jobs["hide"]) and set(by) == {"UNCOACHED", "COACHED"}


def test_voice_scores_move_toward_the_target_and_the_source():
    voice, models, prompts, texts, analysis = _voice_fixture()
    P = {p["id"]: p for p in prompts}
    base = voice.Baseline(texts, P, models)
    assert voice.project(np.array([1.0, 1.0]), np.zeros(2), np.array([2.0, 2.0])) == 0.5
    assert voice.project(np.ones(2), np.ones(2), np.ones(2)) is None
    ci = voice.prompt_ci({"p1": [1.0, 0.0], "p2": [1.0]})
    assert ci["mean"] == pytest.approx(2 / 3) and ci["prompts"] == 2
    # a perfect impression: a writes b's own answer, so it lands on b
    rows = [{"experiment": "imitate", "condition": "NAME", "writer": "a", "other": "b", "prompt_id": pid,
             "category": P[pid]["category"], "text": texts[("b", pid)]} for pid in ("explain0", "advice0")]
    out = voice.score_imitate(rows, base, None, sides={("a", "b", "length"): -1})
    assert out["I1_name_moves_toward_target"]["mean"] == pytest.approx(1.0)
    assert out["I3_moves_right_where_described_wrong"]["mean"] == 1.0  # it moved length the right way anyway
    # after b's notes, a answers like b; after its own, like itself
    con = []
    for pid in ("explain1", "advice1"):
        con.append({"experiment": "contagion", "condition": "SELF", "writer": "a", "other": "a", "prompt_id": pid,
                    "category": P[pid]["category"], "text": texts[("a", pid)]})
        con.append({"experiment": "contagion", "condition": "SOURCE", "writer": "a", "other": "b", "prompt_id": pid,
                    "category": P[pid]["category"], "text": texts[("b", pid)]})
    c = voice.score_contagion(con, base, None)
    assert c["C1_moves_toward_source"]["mean"] > 0.5
    sides = voice.described_sides([{"describer": "a", "target": "b", "sample": s, "estimates": {"words": 50}} for s in (1, 2)]
                                  + [{"describer": "a", "target": "a", "sample": s, "estimates": {"words": 300}} for s in (1, 2)],
                                  models)
    assert sides[("a", "b", "length")] == -1  # a thinks b writes shorter than itself


def test_style_swap_crosses_content_with_voice(monkeypatch):
    voice, models, prompts, texts, analysis = _voice_fixture()
    monkeypatch.setattr(voice, "SOURCES", ("b",))
    notes = voice.note_prompts(prompts)
    rw = voice.swap_rewrite_jobs(models, prompts, texts)
    pairs = {(j["writer"], j["other"]) for j in rw}
    assert pairs == {("a", "a"), ("b", "b"), ("c", "c"), ("b", "a"), ("a", "b"), ("b", "c"), ("c", "b")}
    assert len(rw) == len(pairs) * len(notes) and all("Rewrite the answer" in j["message"] for j in rw)
    rewrites = {(w, a, q["id"]): f"<{w} voicing {a} on {q['id']}>" for w, a in pairs for q in notes}
    jobs = voice.swap_answer_jobs(models, prompts, rewrites)
    assert {j["condition"] for j in jobs} == set(voice.SWAP_CELLS)
    os_job = next(j for j in jobs if j["condition"] == "OS" and j["writer"] == "a")
    assert f"<b voicing a on {notes[0]['id']}>" in os_job["message"]  # the reader's content in the source's voice
    so_job = next(j for j in jobs if j["condition"] == "SO" and j["writer"] == "a")
    assert f"<a voicing b on {notes[0]['id']}>" in so_job["message"]
    assert not {j["prompt_id"] for j in jobs} & {q["id"] for q in notes}
    # a reader that writes like whoever's voice it read, whatever the content: the voice effect is the whole way
    P = {p["id"]: p for p in prompts}
    base = voice.Baseline(texts, P, models)
    rows = []
    for j in jobs:
        voiced_by = j["writer"] if j["condition"][1] == "O" else j["other"]
        rows.append(dict(j, text=texts[(voiced_by, j["prompt_id"])]))
    out = voice.score_swap(rows, base, None, [], texts)
    assert out["voice"]["mean"] == pytest.approx(1.0) and out["voice"]["confirmed"]
    assert out["content"]["mean"] == pytest.approx(0.0)


def test_extend_copies_answers_and_the_kept_coder_only(tmp_path, monkeypatch):
    from spectra import extend as ext
    from spectra import load_config

    monkeypatch.setattr(ext, "DATA_ROOT", tmp_path)
    src = tmp_path / "old"
    src.mkdir()
    (src / "responses.jsonl").write_text("".join(json.dumps({"model": m, "prompt_id": "p1", "sample": 0, "text": "x"}) + "\n"
                                                 for m in ("a", "b")), encoding="utf-8")
    (src / "codes.jsonl").write_text("".join(json.dumps({"coder": c, "model": "a", "prompt_id": "p1", "sample": 0, "codes": {}})
                                             + "\n" for c in ("keep", "drop")), encoding="utf-8")
    (src / "voice.jsonl").write_text("".join(json.dumps({"experiment": e}) + "\n" for e in ("contagion", "imitate")),
                                     encoding="utf-8")
    rec = ext.extend("old", "new", "keep")
    assert rec["copied"] == {"responses": 2, "codes": 1, "voice": 1} and rec["models_before"] == ["a", "b"]
    with pytest.raises(SystemExit):
        ext.extend("old", "new", "keep")  # never over a revision that has answers
    eight = load_config(SPECTRA_DIR / "config-8-v2.yaml")  # extends config-8, which extends config.yaml
    assert len(eight["models"]) == 8 and eight["prompts"].endswith("prompts_v2.json")
    assert [c["name"] for c in eight["coders"]] == ["deepseek-v4-pro", "mistral-large"]
    assert not {c["name"] for c in eight["coders"]} & {m["name"] for m in eight["models"]}


def test_replicates_split_the_reaction_from_the_noise(tmp_path):
    from spectra import load_config
    from spectra.replicates import analyse, consistency, decompose

    rng = np.random.default_rng(0)
    M, P, S = 3, 40, 3
    model, prompt = rng.normal(0, 1, (M, 1, 1)), rng.normal(0, 3, (1, P, 1))
    noise = rng.normal(0, 1, (M, P, S))
    flat = decompose(model + prompt + noise)
    assert sum(flat[k] for k in ("model", "prompt", "model_x_prompt", "noise")) == pytest.approx(1.0)
    assert flat["p"] > 0.01  # no reaction planted
    reacting = decompose(model + prompt + rng.normal(0, 2, (M, P, 1)) + noise)
    assert reacting["p"] < 1e-6 and reacting["model_x_prompt"] > flat["model_x_prompt"]
    assert decompose((model + prompt + noise)[:, :, :1])["noise"] == 0.0  # one answer: noise hides in the residual
    from spectra.replicates import components
    c = components(model + prompt + noise)
    assert sum(c.values()) == pytest.approx(1.0) and c["model_x_prompt"] < 0.02  # no reaction planted, none estimated
    assert c["model_x_prompt"] < flat["model_x_prompt"]  # the sum of squares hands it a third of the noise
    assert consistency(np.repeat(rng.normal(0, 1, (P, 1)), S, axis=1)) == pytest.approx(1.0)

    # end to end: b writes longer than a; each prompt sets a length; a's second and third answers wander
    prompts = {f"q{i}": {"id": f"q{i}", "category": "explain"} for i in range(12)}
    rows = []
    for i in range(12):
        for m, extra, wander in (("a", 0, (0, 9, -15)), ("b", 40, (0, 1, -1))):
            rows += [{"model": m, "prompt_id": f"q{i}", "sample": s, "text": words(30 + 5 * i + extra + w)}
                     for s, w in enumerate(wander)]
    (tmp_path / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    imitate = [{"experiment": "imitate", "condition": c, "writer": "a", "other": "b", "prompt_id": f"q{i}",
                "category": "explain", "text": words(30 + 5 * i)} for i in range(12) for c in ("SELF", "NAME")]
    out = analyse(tmp_path, prompts, ["a", "b"], S, imitate)
    length = out["traits"]["length"]
    assert length["model"] > 0.3 and length["one_answer"]["model"] > 0.3 and "noise" in length["later_answers"]
    assert out["consistency"]["b"] > out["consistency"]["a"]  # b repeats itself; a does not
    im = out["imitation"]
    assert im["self_drift_single"]["mean"] == pytest.approx(0.0)  # SELF repeats a's first answer exactly
    assert im["self_drift_averaged"]["mean"] > 0  # a's average is shorter, so the same answer now sits toward b
    assert im["drift_shrink"]["mean"] == pytest.approx(-im["self_drift_averaged"]["mean"])
    three = load_config(SPECTRA_DIR / "config-s3.yaml")
    assert three["samples"] == 3 and three["concurrency_per_model"] == 6 and three["revision"] is None
    assert len(three["models"]) == 6


def test_matching_names_each_model_once_per_prompt():
    df = pd.DataFrame({"prompt_id": ["p1"] * 3 + ["p2"] * 2, "sample": [0] * 5, "model": ["a", "b", "c", "a", "b"]})
    proba = np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    # p1: one guess at a time names "a" twice; jointly it is a, b, c. p2 lacks a model, so it keeps the best guesses.
    assert list(match_by_prompt(proba, df, 3)) == [0, 1, 2, 1, 1]


# ----------------------------------------------------------------------------- coded traits
class TestCodedTraits:
    def test_not_applicable_on_a_trap_prompt_is_left_out(self):
        labels = [{"premise": "not_applicable"}, {"premise": "challenged"}]
        assert coded_value(labels, "premise", {"challenged"}, tagged=True) == 1.0
        assert math.isnan(coded_value([{"premise": "not_applicable"}], "premise", {"challenged"}, tagged=True))

    def test_untagged_codes_are_unchanged(self):
        assert coded_value([{"tone": "warm"}, {"tone": "blunt"}], "tone", {"warm"}, tagged=False) == 0.5
        assert math.isnan(coded_value([{}], "tone", {"warm"}, tagged=False))

    def test_prompt_outcomes(self):
        df = pd.DataFrame({"model": ["m"] * 4, "prompt_id": ["p1", "p2", "p3", "p4"], "c": [1.0, 0.0, 0.5, None]})
        assert prompt_outcomes(df, "m", "c") == {"yes": ["p1"], "no": ["p2"], "split": ["p3"]}


# ----------------------------------------------------------------------------- throttling and locks
def test_models_from_one_provider_share_a_semaphore():
    sems = providers.semaphores([{"name": "gemini-4-argon"}, {"name": "gemini-3.8-flash"}, {"name": "gpt-6-astra"}], 32)
    assert sems["gemini-4-argon"] is sems["gemini-3.8-flash"]
    assert sems["gpt-6-astra"] is not sems["gemini-4-argon"]
    from peerrank.config import PROVIDER_CONCURRENCY
    assert sems["gemini-4-argon"]._value == min(32, PROVIDER_CONCURRENCY["google"])


class TestWriterLock:
    def test_a_second_writer_is_refused(self, tmp_path):
        out = tmp_path / "guesses.jsonl"
        with writer_lock(out):
            assert (tmp_path / "guesses.jsonl.lock").exists()
            with pytest.raises(SystemExit, match="already being written"):
                with writer_lock(out):
                    pass
        assert not (tmp_path / "guesses.jsonl.lock").exists()

    def test_a_stale_lock_is_taken_over(self, tmp_path):
        done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
        (tmp_path / "codes.jsonl.lock").write_text(done.stdout.strip() + " old run\n", encoding="utf-8")
        with writer_lock(tmp_path / "codes.jsonl"):
            pid = int((tmp_path / "codes.jsonl.lock").read_text(encoding="utf-8").split()[0])
            assert pid != int(done.stdout)


# ----------------------------------------------------------------------------- report recommendations
CATS = ["c1", "c2", "c3", "c4", "c5"]
MODELS = ["alpha-1", "alpha-2", "beta-1"]


def trait(key, values, by_cat=None, tag=None, n=50, unit="pct"):
    mean = sum(values.values()) / len(values)
    spread = (max(values.values()) - min(values.values())) or 1
    by_cat = by_cat or {m: {c: v for c in CATS} for m, v in values.items()}
    return {"key": key, "label": key.replace("_", " ").title(), "short": key.replace("_", " ").title(),
            "unit": unit, "group": "behavior" if tag or key in CODED else "style", "tag": tag,
            "high": f"More {key}", "low": f"Less {key}", "panel": mean,
            "per_model": {m: {"value": v, "lo": v, "hi": v, "n": n, "d": (v - mean) / spread} for m, v in values.items()},
            "by_category": by_cat}


CODED = {"answer_first", "initiative", "errors", "ambiguity", "premise", "warm_tone", "advice"}


def synthetic_analysis(coded=True):
    # alpha-1 is the shortest overall but shortest in only 3 of the 5 categories
    length_by_cat = {"alpha-1": {c: (100 if c in ("c1", "c2", "c3") else 300) for c in CATS},
                     "alpha-2": {c: (200 if c in ("c1", "c2", "c3") else 150) for c in CATS},
                     "beta-1": {c: 400 for c in CATS}}
    traits = [
        trait("length", {"alpha-1": 150, "alpha-2": 200, "beta-1": 400}, length_by_cat, unit="words"),
        trait("structure", {"alpha-1": 2.0, "alpha-2": 6.0, "beta-1": 4.0}, unit="per100"),
        trait("hedging", {"alpha-1": 0.9, "alpha-2": 0.5, "beta-1": 0.3}, unit="per100"),
        trait("caveats", {"alpha-1": 0.0, "alpha-2": 0.1, "beta-1": 0.3}),
        trait("warm_open", {"alpha-1": 0.01, "alpha-2": 0.2, "beta-1": 0.02}),
        trait("hooks", {"alpha-1": 0.1, "alpha-2": 0.4, "beta-1": 0.05}),
        trait("readability", {"alpha-1": 55.0, "alpha-2": 60.0, "beta-1": 70.0}, unit="score"),
        trait("vocab", {"alpha-1": 0.86, "alpha-2": 0.85, "beta-1": 0.80}, unit="ratio"),
        trait("emdash", {"alpha-1": 0.1, "alpha-2": 1.5, "beta-1": 0.01}, unit="per100"),
        trait("adherence", {"alpha-1": 0.9, "alpha-2": 0.8, "beta-1": 0.6}, n=5),
    ]
    if coded:
        traits += [
            trait("answer_first", {"alpha-1": 0.95, "alpha-2": 0.8, "beta-1": 0.5}),
            trait("initiative", {"alpha-1": 0.2, "alpha-2": 0.6, "beta-1": 0.8}),
            trait("errors", {"alpha-1": 1.0, "alpha-2": 0.75, "beta-1": 0.25}, tag="ERR", n=4,
                  by_cat={"alpha-1": {"c1": 1.0}, "alpha-2": {"c1": 0.75}, "beta-1": {"c1": 0.0}}),
            trait("ambiguity", {"alpha-1": 0.7, "alpha-2": 0.9, "beta-1": 0.4}, tag="AMB", n=7),
            trait("premise", {"alpha-1": 1.0, "alpha-2": 1.0, "beta-1": 0.8}, tag="FP", n=5),
            trait("warm_tone", {"alpha-1": 0.2, "alpha-2": 0.3, "beta-1": 0.6}),
            trait("advice", {"alpha-1": 0.1, "alpha-2": 0.2, "beta-1": 0.5}),
        ]
    return {
        "meta": {"revision": "test-rev", "models": MODELS, "categories": CATS, "n_prompts": 40, "n_responses": 120,
                 "samples": 1, "coders": ["c-a", "c-b"] if coded else [], "system_prompt": None},
        "traits": traits,
        "cards": {m: {"descriptors": [{"key": "length", "text": "Keeps answers short", "d": -1.0}],
                      "neighbor": next(o for o in MODELS if o != m)} for m in MODELS},
        "constraints": {"types": [{"key": "max_words", "label": "Word limit", "n_prompts": 3},
                                  {"key": "exact_items", "label": "Exact number of items", "n_prompts": 2}],
                        "per_model": {"alpha-1": {"max_words": 1.0, "exact_items": 0.5},
                                      "alpha-2": {"max_words": 1.0, "exact_items": 1.0},
                                      "beta-1": {"max_words": 0.67, "exact_items": 0.5}}},
        "signature": {m: [{"term": f"{m}-word", "z": 3.0, "count": 9}] for m in MODELS},
        "distinctiveness": None,
        "runstats": {m: {"responses": 40, "errors": 0, "median_latency_s": lat, "median_output_tokens": tok,
                         "median_reasoning_tokens": None, "total_cost_usd": cost, "median_words": w,
                         "length_cv_across_samples": None}
                     for m, lat, tok, cost, w in (("alpha-1", 5.0, 200, 1.5, 140), ("alpha-2", 9.0, 300, 3.0, 190),
                                                  ("beta-1", 30.0, 1400, 0.5, 380))},
        "agreement": None,
        "viewer": {"prompts": [], "responses": {}},
    }


def all_text(recs):
    return json.dumps(recs, ensure_ascii=False)


class TestRecommendations:
    def test_counts_come_from_the_data(self):
        recs = build_recommendations(synthetic_analysis())
        text = all_text(recs)
        short = next(u for u in recs["uses"] if u["need"] == "Short, fast first answers")
        assert "shortest in 3/5 categories" in short["evidence"]
        assert "longest answer in 5/5 categories" in short["watch"]
        errors = next(u for u in recs["uses"] if u["need"] == "Catch errors in pasted text")
        assert "n=4" in errors["evidence"] and "kappa" not in errors["evidence"]  # no agreement in this run
        assert "c1 prompts" in errors["watch"]  # beta-1 scored 0 there
        for stale in ("10/10", "n=6", "n=11", "27 constrained", "200 answers", "Flash", "Muse", "Opus", "Fable"):
            assert stale not in text, stale

    def test_no_unformatted_placeholders(self):
        text = all_text(build_recommendations(synthetic_analysis()))
        assert not re.search(r"\{[A-Za-z_]+[(\[]", text), "an f-string prefix is missing somewhere"

    def test_one_card_per_model(self):
        drive = build_recommendations(synthetic_analysis())["drive"]
        assert [d["model"] for d in drive] == MODELS  # one each, in roster order

    def test_model_without_a_role_still_gets_a_card(self):
        analysis = synthetic_analysis(coded=False)
        structure = next(t for t in analysis["traits"] if t["key"] == "structure")
        structure["per_model"]["beta-1"]["value"] = 9.0  # beta-1 now holds every role alpha-2 had
        drive = {d["model"]: d for d in build_recommendations(analysis)["drive"]}
        assert set(drive) == set(MODELS)
        assert "closest neighbor" in drive["alpha-2"]["role"]

    def test_lone_false_premise_claim_only_when_true(self):
        analysis = synthetic_analysis()
        claim = "the only model the coders agree went along with a false premise"
        assert claim + " (challenged 80%)" in all_text(build_recommendations(analysis))  # no prompt ids kept
        premise = next(t for t in analysis["traits"] if t["key"] == "premise")
        premise["per_model"]["alpha-1"]["value"] = 0.8  # now two models went along with one
        assert "the only model" not in all_text(build_recommendations(analysis))

    def test_trap_claims_name_their_prompts(self):
        analysis = synthetic_analysis()
        for t in analysis["traits"]:
            if t["key"] in ("premise", "errors"):
                for m in MODELS:
                    t["per_model"][m].update(yes=[], no=[], split=[])
        premise = next(t for t in analysis["traits"] if t["key"] == "premise")
        premise["per_model"]["beta-1"]["no"] = ["p020"]
        errors = next(t for t in analysis["traits"] if t["key"] == "errors")
        errors["per_model"]["beta-1"]["no"] = ["p145", "p157"]
        text = all_text(build_recommendations(analysis))
        assert "went along with a false premise (1 of 5: p020)" in text
        assert "Both coders agree these were missed: beta-1 on p145, p157." in text

    def test_runs_without_behavior_codes(self):
        recs = build_recommendations(synthetic_analysis(coded=False))
        needs = {u["need"] for u in recs["uses"]}
        assert "Catch errors in pasted text" not in needs and "Vague or underspecified requests" not in needs
        assert "Short, fast first answers" in needs and "Hard format or length limits" in needs
        assert "no behavior codes" in recs["method"]

    def test_report_renders_from_a_run_dir(self, tmp_path, monkeypatch):
        from spectra import report
        (tmp_path / "analysis.json").write_text(json.dumps(synthetic_analysis(coded=False)), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["report", "--run-dir", str(tmp_path)])
        report.main()
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "Usage recommendations" in html and '"drive_note"' in html and '"guessing"' not in html

        labels = {"A": "alpha-1", "B": "alpha-2", "C": "beta-1"}
        rows = [{"guesser": "alpha-1", "prompt_id": "p1", "sample": 0, "labels": labels, "guesses": labels},
                {"guesser": "alpha-1", "prompt_id": "p1", "sample": 0, "labels": labels, "guesses": labels,
                 "mode": "heldout", "research": "test-rev", "assign": "independent"}]
        (tmp_path / "guesses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        report.main()
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
        assert [g["title"] for g in data["guessing"]["games"]] == [
            "Just guess",
            "Guess from held-out research (notes from the other half of the prompts), each answer judged on its own"]
