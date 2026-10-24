"""Spectra unit tests: constraint checks, guess parsing and scoring, and the report's recommendation text.

  pytest tests/
"""
import json
import math
import re
import sys

import pytest

from spectra.analyze import check_constraint
from spectra.guess import parse_assignment, resolve_name, summarize_guesses
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


def test_summarize_guesses_leaves_own_answer_out():
    models = ["alpha-1", "beta-1", "gamma-1"]
    labels = {"A": "alpha-1", "B": "beta-1", "C": "gamma-1"}
    records = [
        {"guesser": "alpha-1", "labels": labels, "guesses": {"A": "alpha-1", "B": "beta-1", "C": "alpha-1"}},
        {"guesser": "beta-1", "labels": labels, "guesses": {"A": "gamma-1", "B": "beta-1", "C": "alpha-1"},
         "mode": "briefed", "research": "r1"},
    ]
    blind = summarize_guesses(records, models, "blind")
    assert blind["n_calls"] == 1 and blind["n_judgments"] == 2
    assert blind["chance"] == pytest.approx(1 / 3)
    alpha = next(g for g in blind["guessers"] if g["model"] == "alpha-1")
    assert (alpha["correct"], alpha["n"], alpha["self_rate"]) == (1, 2, 1.0)
    recall = {r["model"]: r["recall"] for r in blind["recognizable"]}
    assert recall == {"beta-1": 1.0, "gamma-1": 0.0, "alpha-1": None}
    briefed = summarize_guesses(records, models, "briefed")
    assert briefed["n_calls"] == 1 and briefed["research"] == ["r1"]


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
        assert "the only model that ever accepted a false premise" in all_text(build_recommendations(analysis))
        premise = next(t for t in analysis["traits"] if t["key"] == "premise")
        premise["per_model"]["alpha-1"]["value"] = 0.8  # now two models accepted one
        assert "the only model" not in all_text(build_recommendations(analysis))

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
        assert "Usage recommendations" in html and '"drive_note"' in html
