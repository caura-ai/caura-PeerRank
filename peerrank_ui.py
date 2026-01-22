"""
PeerRank Interactive UI - Streamlit app for live LLM comparison with bias testing
Run with: streamlit run peerrank_ui.py
"""

import asyncio
import random
import streamlit as st
import time
from statistics import mean

import pandas as pd

from peerrank.config import (
    MODELS, ALL_MODELS, extract_json, MAX_TOKENS_ANSWER, MAX_TOKENS_EVAL,
    TEMPERATURE_EVAL, match_model_name, BIAS_MODES, BIAS_CONFIGS, UI_DISPLAY_MODES,
    calculate_scores_from_evaluations, calculate_elo_ratings,
)
from peerrank.providers import call_llm
from peerrank_phase3 import EVAL_PROMPT, BLIND_LABELS

# BIAS_MODES and UI_DISPLAY_MODES imported from config.py

# Page config
st.set_page_config(
    page_title="PeerRank.ai - Live LLM Comparison",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PeerRank.ai - Endogenous LLM Ranking with Bias Analysis")

# Session state initialization
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}  # {config_name: {evaluator: eval_data}}
if "question" not in st.session_state:
    st.session_state.question = ""


def format_responses_for_eval(answers: list[dict], shuffle: bool, blind: bool, seed: int) -> tuple[str, dict]:
    """
    Format responses for evaluation with optional shuffling and blinding.
    Returns (formatted_text, label_to_model_map)
    """
    pairs = [(a["model"], a["answer"]) for a in answers if a.get("answer")]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pairs)

    label_to_model = {}
    lines = []

    for i, (model_name, answer) in enumerate(pairs):
        if blind:
            label = f"Response {BLIND_LABELS[i]}"
        else:
            label = model_name

        label_to_model[label] = model_name
        lines.append(f"--- {label} ---\n{answer}\n")

    return "\n".join(lines), label_to_model


def remap_scores_to_models(scores: dict, label_to_model: dict) -> dict:
    """Convert scores keyed by label back to scores keyed by actual model name."""
    remapped = {}
    for label, score_data in scores.items():
        model_name = label_to_model.get(label)
        if model_name:
            remapped[model_name] = score_data
        else:
            for full_label, model in label_to_model.items():
                if label in full_label or full_label in label:
                    remapped[model] = score_data
                    break
    return remapped


async def generate_random_question() -> str:
    """Use GPT to generate a random interesting question."""
    prompt = """Generate one interesting, thought-provoking question that would test an AI's knowledge or reasoning.
The question should be specific and answerable. Return ONLY the question, nothing else."""

    response, _, _, _ = await call_llm("openai", "gpt-5.2", prompt, max_tokens=200, use_web_search=False)
    return response.strip()


async def get_answer(provider: str, model: str, display_name: str, question: str) -> dict:
    """Get answer from a single model with timing."""
    start = time.time()
    try:
        response, duration, _, _ = await call_llm(
            provider, model, question,
            max_tokens=MAX_TOKENS_ANSWER,
            use_web_search=True
        )
        return {
            "model": display_name,
            "answer": response,
            "duration": duration,
            "error": None
        }
    except Exception as e:
        return {
            "model": display_name,
            "answer": None,
            "duration": time.time() - start,
            "error": str(e)
        }


async def get_all_answers(question: str) -> list[dict]:
    """Get answers from all models in parallel."""
    tasks = [get_answer(provider, model_id, display_name, question) for provider, model_id, display_name in MODELS]
    return await asyncio.gather(*tasks)


async def evaluate_answers_with_config(
    provider: str, evaluator: str, display_name: str,
    question: str, answers: list[dict],
    shuffle: bool, blind: bool, seed: int
) -> dict:
    """Have one model evaluate all answers with specific bias config."""
    responses_text, label_to_model = format_responses_for_eval(answers, shuffle, blind, seed)

    label_example = "Response A" if blind else list(label_to_model.keys())[0] if label_to_model else "ModelName"
    prompt = EVAL_PROMPT.format(
        question=question,
        responses=responses_text,
        label_example=label_example
    )

    try:
        response, duration, _, _ = await call_llm(
            provider, evaluator, prompt,
            max_tokens=MAX_TOKENS_EVAL,
            use_web_search=False,
            temperature=TEMPERATURE_EVAL
        )
        scores = extract_json(response)
        if isinstance(scores, dict):
            scores = remap_scores_to_models(scores, label_to_model)
        return {
            "evaluator": display_name,
            "scores": scores if isinstance(scores, dict) else {},
            "duration": duration,
            "error": None,
            "label_to_model": label_to_model
        }
    except Exception as e:
        return {
            "evaluator": display_name,
            "scores": {},
            "duration": 0,
            "error": str(e),
            "label_to_model": label_to_model
        }


async def get_all_evaluations_for_config(
    question: str, answers: list[dict],
    shuffle: bool, blind: bool, seed: int
) -> list[dict]:
    """Get evaluations from all models for a specific bias config."""
    tasks = [
        evaluate_answers_with_config(
            provider, model_id, display_name,
            question, answers, shuffle, blind, seed
        )
        for provider, model_id, display_name in MODELS
    ]
    return await asyncio.gather(*tasks)


async def run_all_bias_evaluations(question: str, answers: list[dict]) -> dict:
    """Run evaluations for all 3 bias configurations."""
    seed = random.randint(0, 2**31)
    results = {}

    for config in BIAS_CONFIGS:
        evals = await get_all_evaluations_for_config(
            question, answers,
            config["shuffle"], config["blind"], seed
        )
        results[config["name"]] = {e["evaluator"]: e for e in evals}

    return results


def calculate_rankings(evaluations: dict, model_names: list[str]) -> tuple[dict, dict, dict, dict]:
    """Calculate aggregate scores, judge scores, self scores, peer scores.
    Uses shared calculation function from config.py.
    """
    scores = calculate_scores_from_evaluations(evaluations, model_names)

    # Convert self_scores list to single value (UI only has one question)
    self_scores = {n: s[0] if s else None for n, s in scores["self_scores"].items()}

    return scores["raw_scores"], scores["judge_given"], self_scores, scores["peer_scores"]


def build_results_df(evaluations: dict, answers: dict, model_names: list[str],
                     mode: str = "peer", peer_baseline: dict = None) -> pd.DataFrame:
    """Build results dataframe for a single bias configuration.

    Args:
        mode: "peer" for Peer Score table (with Self Bias),
              "shuffle" for Shuffle table (with Name Bias)
        peer_baseline: Dict of {model: peer_score} from shuffle_blind mode (for Name Bias calc)
    """
    aggregate_scores, judge_scores, self_scores, peer_scores = calculate_rankings(evaluations, model_names)

    model_avgs = []
    for name in model_names:
        scores = peer_scores[name]
        avg = mean(scores) if scores else 0
        model_avgs.append((name, avg))
    model_avgs.sort(key=lambda x: x[1], reverse=True)

    table_data = []
    for rank, (model_name, avg_score) in enumerate(model_avgs, 1):
        score_avg = mean(peer_scores[model_name]) if peer_scores[model_name] else None

        if mode == "peer":
            # Peer Score table: Peer, Self, Self Bias
            self_score = self_scores[model_name]
            self_bias = (self_score - score_avg) if (self_score is not None and score_avg is not None) else None
            row = {
                "Rank": rank,
                "Model": model_name,
                "Peer": f"{score_avg:.1f}" if score_avg else "-",
                "Self": f"{self_score}" if self_score is not None else "-",
                "Self Bias": f"{self_bias:+.1f}" if self_bias is not None else "-",
            }
        elif mode == "shuffle":
            # Shuffle table: Score, Name Bias (Peer - Shuffle)
            peer_score = peer_baseline.get(model_name, 0) if peer_baseline else 0
            name_bias = (peer_score - score_avg) if (peer_score and score_avg) else None
            row = {
                "Rank": rank,
                "Model": model_name,
                "Score": f"{score_avg:.1f}" if score_avg else "-",
                "Name Bias": f"{name_bias:+.1f}" if name_bias is not None else "-",
            }
        else:
            row = {"Rank": rank, "Model": model_name, "Score": f"{score_avg:.1f}" if score_avg else "-"}

        table_data.append(row)

    return pd.DataFrame(table_data)


# UI Layout
label_col, random_col = st.columns([6, 1])
with label_col:
    st.markdown("**Enter your question:**")
with random_col:
    if st.button("🎲", help="Generate random question"):
        with st.spinner("Generating..."):
            st.session_state.question = asyncio.run(generate_random_question())
            st.rerun()

question_input = st.text_area(
    "question",
    value=st.session_state.question,
    height=100,
    placeholder="Ask anything...",
    label_visibility="collapsed"
)

# Action buttons
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    answer_button = st.button("📝 Get Answers", width="stretch", type="primary")
with btn_col2:
    has_answers = bool(st.session_state.answers)
    evaluate_button = st.button(
        "⚖️ Evaluate (3 bias modes)",
        width="stretch",
        disabled=not has_answers
    )

# Process: Get Answers
if answer_button and question_input:
    st.session_state.question = question_input
    st.session_state.evaluations = {}

    with st.spinner("Getting answers from all models..."):
        answers = asyncio.run(get_all_answers(question_input))
        st.session_state.answers = {a["model"]: a for a in answers}
    st.rerun()

# Process: Evaluate with all 3 bias modes
if evaluate_button and st.session_state.answers:
    answers_list = list(st.session_state.answers.values())
    with st.spinner("Running 3 bias modes (shuffle_only, blind_only, shuffle_blind)..."):
        evaluations = asyncio.run(run_all_bias_evaluations(st.session_state.question, answers_list))
        st.session_state.evaluations = evaluations
    st.rerun()

# Display results
if st.session_state.answers:
    st.divider()
    model_names = [display_name for _, _, display_name in MODELS]

    # Show bias test results if we have evaluations
    if st.session_state.evaluations:
        st.subheader("📊 Results")

        # Calculate scores for all 3 modes (needed for bias calculations)
        scores_by_config = {}
        for config in BIAS_CONFIGS:
            config_name = config["name"]
            config_evals = st.session_state.evaluations.get(config_name, {})
            if config_evals:
                _, _, _, peer_sc = calculate_rankings(config_evals, model_names)
                scores_by_config[config_name] = {n: mean(peer_sc[n]) if peer_sc[n] else 0 for n in model_names}

        # Get shuffle_blind evaluations for ELO and judge generosity
        shuffle_blind_evals = st.session_state.evaluations.get("shuffle_blind", {})

        # Display 3 columns: ELO Ratings, Name Bias, Judge Generosity
        cols = st.columns(3)

        # Column 1: ELO Ratings
        with cols[0]:
            st.markdown("### 🏆 Elo Rating")
            st.caption("Pairwise comparison ranking")

            if shuffle_blind_evals:
                # Convert UI format to phase3 format for ELO calculation
                elo_evaluations = {}
                for evaluator, eval_data in shuffle_blind_evals.items():
                    scores = eval_data.get("scores", {})
                    if scores:
                        elo_evaluations[evaluator] = {"q1": scores}

                elo_data = calculate_elo_ratings(elo_evaluations, model_names)
                if elo_data and elo_data["ratings"]:
                    # Build ELO table (Rank, Model, Elo only)
                    elo_rows = []
                    sorted_models = sorted(model_names, key=lambda m: elo_data["ratings"].get(m, 1500), reverse=True)
                    for rank, model in enumerate(sorted_models, 1):
                        elo_rows.append({
                            "Rank": rank,
                            "Model": model,
                            "Elo": int(elo_data["ratings"].get(model, 1500)),
                        })
                    df_elo = pd.DataFrame(elo_rows)
                    st.dataframe(df_elo, width="stretch", hide_index=True)
                else:
                    st.write("Could not calculate Elo ratings")
            else:
                st.write("No data")

        # Column 2: Name Bias (Shuffle table)
        with cols[1]:
            st.markdown("### 🔀 Name Bias")
            st.caption("Score change when names visible")

            shuffle_only_evals = st.session_state.evaluations.get("shuffle_only", {})
            if shuffle_only_evals:
                peer_baseline = scores_by_config.get("shuffle_blind", {})
                df = build_results_df(shuffle_only_evals, st.session_state.answers, model_names,
                                      mode="shuffle", peer_baseline=peer_baseline)
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.write("No data")

        # Column 3: Judge Generosity
        with cols[2]:
            st.markdown("### ⚖️ Judge Generosity")
            st.caption("Avg score given by each judge")

            if shuffle_blind_evals:
                # Calculate average score given by each evaluator
                judge_rows = []
                for evaluator in model_names:
                    eval_data = shuffle_blind_evals.get(evaluator, {})
                    scores = eval_data.get("scores", {})
                    if scores:
                        # Get all scores given (excluding self if desired)
                        all_given = [s.get("score", s) if isinstance(s, dict) else s
                                     for m, s in scores.items() if m != evaluator]
                        avg_given = mean(all_given) if all_given else 0
                        judge_rows.append({
                            "Judge": evaluator,
                            "Avg Given": f"{avg_given:.1f}",
                        })

                # Sort by avg given (most generous first)
                judge_rows.sort(key=lambda x: float(x["Avg Given"]), reverse=True)
                df_judge = pd.DataFrame(judge_rows)
                st.dataframe(df_judge, width="stretch", hide_index=True)
            else:
                st.write("No data")

        # Comparison summary
        st.divider()
        st.subheader("🔍 Bias Effect Analysis")

        summary_cols = st.columns(2)

        # Position Bias: shuffle_blind vs blind_only (randomizing order)
        # Show by position number (fixed order from ALL_MODELS), not model name
        with summary_cols[0]:
            st.markdown("**Position Bias**")
            st.caption("Score change when randomizing order (Peer − Blind)")
            if "shuffle_blind" in scores_by_config and "blind_only" in scores_by_config:
                fixed_order = [n for _, _, n in ALL_MODELS]
                for pos, name in enumerate(fixed_order, 1):
                    if name in model_names:
                        peer_score = scores_by_config["shuffle_blind"].get(name, 0)
                        blind_score = scores_by_config["blind_only"].get(name, 0)
                        effect = peer_score - blind_score
                        if abs(effect) > 0.1:
                            st.write(f"Position {pos}: {effect:+.2f}")
                if not any(abs(scores_by_config["shuffle_blind"].get(n, 0) - scores_by_config["blind_only"].get(n, 0)) > 0.1 for n in model_names):
                    st.write("No significant effects")

        # Self-bias comparison
        with summary_cols[1]:
            st.markdown("**Self-Bias by Mode**")
            st.caption("How much models favor themselves")
            for config in BIAS_CONFIGS:
                config_name = config["name"]
                config_evals = st.session_state.evaluations.get(config_name, {})
                if config_evals:
                    _, _, self_sc, peer_sc = calculate_rankings(config_evals, model_names)
                    biases = []
                    for name in model_names:
                        if self_sc[name] is not None and peer_sc[name]:
                            biases.append(self_sc[name] - mean(peer_sc[name]))
                    avg_bias = mean(biases) if biases else 0
                    st.write(f"{config['icon']} {config_name}: {avg_bias:+.2f}")

        st.divider()

# Footer
st.caption("PeerRank.ai - LLM Peer Evaluation with Bias Testing")
