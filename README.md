# PeerRank

**Do LLMs agree on which LLM is best?** PeerRank lets AI models evaluate each other through a structured peer review process—generating questions, answering them with web search, and cross-evaluating responses to produce ranked results with bias analysis.

The same repo includes **Spectra**, a behavioral profile of how those models write. It measures style and habits. It does not rank answer quality. See [Spectra](#spectra).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Key Features

- **5-Phase Pipeline** — Question generation → Answering → Cross-evaluation → Report → Analysis
- **25 Models, 10 Providers** — OpenAI, Anthropic, Google, xAI, DeepSeek, Meta, Together AI, Moonshot AI, Mistral, Perplexity (6 on the active roster)
- **Bias Detection** — Measures self-bias, name bias, and position bias through controlled evaluation modes
- **Standardized Web Grounding** — Tavily or SerpAPI search, same context for all models (fair comparison)
- **Cost Tracking** — Real-time token usage and cost analysis per model
- **Publication Figures** — Generate publication-quality charts and statistical analysis
- **Ground Truth Validation** — TruthfulQA (r=0.858) and GSM8K (r=0.986) correlation with accuracy
- **Spectra** — Style and behavior profiles of the same roster, plus a guessing game in two modes: the answers alone, or the answers plus the measured profile

## Quick Start

```bash
# Clone and install
git clone https://github.com/caura-ai/caura-PeerRank.git
cd caura-PeerRank
pip install -e .

# Configure API keys
# Create a .env file with your API keys (see API Keys section below)

# Test your setup
python peerrank.py --health

# Run the evaluation
python peerrank.py              # Interactive menu
python peerrank.py --all        # Run all 5 phases
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PeerRank Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1          Phase 2          Phase 3          Phase 4    Phase 5  │
│  ────────         ────────         ────────         ────────   ──────── │
│  Generate    →    Answer      →    Cross-      →    Generate → Analysis │
│  Questions        Questions        Evaluate          Report             │
│                   (+ web search)   (3 bias modes)                       │
│                                                                         │
│  Each model       All models       Each model        Markdown   Judge   │
│  creates Qs       answer all Qs    rates all         report     LLM     │
│  across           with optional    responses in      with       reviews │
│  categories       grounding        blind/shuffle     rankings   results │
│                                    conditions                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Bias Detection

Phase 3 runs three evaluation passes to detect bias:

| Bias Type | What It Measures | Formula |
|-----------|------------------|---------|
| **Self Bias** | Do models overrate their own answers? | Self Score − Peer Score |
| **Name Bias** | Does brand recognition affect scores? | Named Score − Anonymous Score |
| **Position Bias** | Does answer order matter? | Fixed Order − Shuffled Order |

## Commands

```bash
# Core commands
python peerrank.py                  # Interactive menu
python peerrank.py --all            # Run all 5 phases
python peerrank.py --phase 1        # Run specific phase (1-5)
python peerrank.py --resume         # Resume from last completed
python peerrank.py --health         # Test API connectivity

# Model selection
python peerrank.py --models gpt-6-astra,claude-opus-5.5  # Include only these
python peerrank.py --exclude gemini-3.8-flash           # Exclude these

# Configuration
python peerrank.py --web-search off           # Disable web grounding (test pure knowledge)
python peerrank.py --grounding-provider serpapi  # Use SerpAPI instead of Tavily
python peerrank.py --seed 42                  # Reproducible shuffle ordering
python peerrank.py --judge claude-opus-5.5    # Set Phase 5 judge model
python peerrank.py --rev v2                   # Set revision tag for output files

# UI and figures
streamlit run peerrank_ui.py                              # Launch web UI
python generate_figures_PeerRank.py --revision v1           # Publication figures
```

## Supported Models

| Provider | Active roster | Also available |
|----------|---------------|----------------|
| OpenAI | gpt-6-astra | gpt-5.6-terra, gpt-5.6-luna, gpt-5-nano, gpt-5-mini |
| Anthropic | claude-fable-5.1, claude-opus-5.5 | claude-sonnet-5, claude-haiku-4-5 |
| Google | gemini-4-argon, gemini-3.8-flash | gemini-3.5-flash-lite, gemini-3.1-pro-preview, gemini-3.5-flash, gemini-3.1-flash-lite |
| xAI | — | grok-4.7, grok-code-fast-1 |
| DeepSeek | — | deepseek-v4-pro, deepseek-v4.1-flash |
| Meta | muse-wonder-1.0 | muse-spark-1.3 |
| Together AI | — | llama-3.3-70b |
| Moonshot AI | — | kimi-k3 |
| Mistral | — | mistral-large |
| Perplexity | — | pplx-agent-medium |

The active roster is the 6 models with `peerrank: True` in `peerrank/models.py`; `--models` and `--exclude` filter within that roster, so set `peerrank: True` to bring another model in. Spectra's roster and coders in `spectra/config.yaml` can name any model in the table.

## API Keys

Create a `.env` file in the project root:

```bash
# Add keys for providers you want to test (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROK_API_KEY=xai-...
DEEPSEEK_API_KEY=sk-...
TOGETHER_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
KIMI_API_KEY=sk-...
MISTRAL_API_KEY=...

# Web grounding (only one required)
TAVILY_API_KEY=tvly-...       # Default: $0.008/search
SERPAPI_KEY=...               # Alternative: ~$0.01/search
```

You only need keys for providers you want to test. Use `--models` to select specific models.

## Output

Results are saved to the `data/` directory with revision tags:

```
data/
├── phase1_questions_v1.json        # Generated questions
├── phase2_answers_v1.json          # Model responses (with cost tracking)
├── phase2_web_grounding_v1.json    # Web grounding data (current events only)
├── phase3_rankings_v1.json         # Cross-evaluations (3 bias modes)
├── phase4_report_v1.md             # Markdown report with rankings
├── phase5_analysis_v1.md           # Judge analysis and insights
└── spectra/<revision>/             # Spectra run (see below)
```

### Sample Report Sections

- **Final Peer Rankings** — Scores from blind+shuffled evaluation
- **Elo Ratings** — Pairwise comparison rankings with W-L-T records
- **Bias Analysis** — Self-bias, name bias, and position bias metrics
- **Judge Generosity** — Which models rate harshly vs. leniently
- **Performance vs. Cost** — Efficiency rankings (Points²/¢)
- **Question Autopsy** — Hardest, easiest, and most controversial questions

## Project Structure

```
peerrank/                # Core package (pip installable)
  __init__.py            # Package exports
  models.py              # Model definitions and pricing
  config.py              # Settings and utilities
  providers.py           # LLM API implementations
  validation_utils.py    # Shared utilities for validation scripts
peerrank.py              # CLI entry point
peerrank_ui.py           # Streamlit web interface
peerrank_phase1-5.py     # Pipeline phases
generate_figures_*.py    # Publication figure generation
validate_truthfulqa.py   # TruthfulQA validation
validate_gsm8k.py        # GSM8K math validation
validate_mmlu.py         # MMLU benchmark validation
spectra/                 # Behavioral profiles (style, habits, guessing game)
  config.yaml            # Roster, revision, coders
  run.py                 # Collect answers
  code_behaviors.py      # Optional behavior labels from two coder models
  tiebreak.py            # A third coder on the labels the two disagree on
  guess.py               # Author-matching game (blind, briefed, or held-out notes)
  profile.py             # The guessers' profile and its optional parts (--profile)
  describe.py            # Self-knowledge probe: models describe their own and each other's habits
  attribute.py           # Author classifiers on the text (words, character n-grams), the guessers' reference
  analyze.py             # Features, traits, style map
  report.py              # HTML report
  prompts/               # prompts_v1.json (build and tune), prompts_v2.json (locked test set) + its .lock hash
  prereg/                # Pre-registrations, committed before the data they cover
  docs/                  # brochure.html, flow.html and paper-plan.html
  article/               # arXiv draft: main.tex, references.bib, make_assets.py -> generated/
pyproject.toml           # Package configuration
```

## Installation as Package

PeerRank can be installed as a Python package:

```bash
pip install -e .                    # Install in editable mode
pip install .[ui]                   # Include Streamlit UI
pip install .[figures]              # Include figure generation
pip install .[all]                  # All optional dependencies
```

## Ground Truth Validation

Validate peer rankings against objective benchmarks:

```bash
# TruthfulQA - factual accuracy
python validate_truthfulqa.py --all       # Run validation (r=0.858)
python generate_figures_TFQ.py            # Generate figures

# GSM8K - mathematical reasoning
python validate_gsm8k.py --all            # Run validation (r=0.986)
python validate_gsm8k.py --difficulty hard  # Hard questions only

# MMLU - multitask language understanding
python validate_mmlu.py --all             # Run validation
python validate_mmlu.py --subjects math,physics  # Specific subjects
```

## Spectra

Spectra profiles how each model on the PeerRank roster answers the same everyday prompts: length, formatting, hedging, vocabulary, and what it does when a prompt hides a trap. Calls go out with no system prompt, so the profile is out-of-the-box API behavior. A revision tag matches a PeerRank revision (`oct-14-26`), so the two describe the same deployment.

The roster lives in `spectra/config.yaml`. Names are PeerRank display names from `peerrank/models.py`, called through PeerRank's providers.

### Pipeline

```bash
pip install -e ".[spectra]"                            # PyYAML, pandas, scikit-learn
python -m spectra.run --revision oct-23-26 --limit 5   # smoke test: 5 prompts, every model
python -m spectra.run --revision oct-23-26             # full run (resumable; re-run retries errors)
python -m spectra.code_behaviors --revision oct-23-26  # optional behavior codes
python -m spectra.analyze --revision oct-23-26         # features, traits, style map, classifier
python -m spectra.report --revision oct-23-26          # data/spectra/oct-23-26/report.html
```

`--revision` defaults to `revision` in `spectra/config.yaml`. `--limit` picks prompts evenly across categories. The full set is 200 English prompts in 10 categories: writing, editing, info, explain, howto, advice, coding, summarize, brainstorm, and opinion.

`prompts_v2.json` is a second set of 200 prompts with the same design slot for slot (category, trap tags, stated limit) and new content. It is the locked test set: profiles are built and tuned on v1 only, and v2 is played once, with the frozen profile. Its hash is in `prompts_v2.lock` and pinned by a test, so an edit fails `pytest`. Until that test, do not read v2 answers, codes, analyses, reports or guesses.

`code_behaviors` is optional. Two coder models apply fixed labels (answer first, flag a pasted error, accept a false premise, and similar), each picked from a fixed list. They sit outside the profiled panel, and agreement is reported per label. `python -m spectra.tiebreak --revision <rev> --coder <model>` has a third coder from another family label only the answers the two disagree on (answer first and moral commentary by default) and writes `tiebreak.jsonl` and `tiebreak.json` with the majority of three; `codes.jsonl` and the traits are not changed. Style traits and limit checks are computed from the text itself.

### Report

`report.html` includes:

- **Profiles** and a **trait explorer** — departures from the panel average, as effect sizes
- **Traps** — false premise, pasted error, vague request, stated opinion, emotional framing
- **Explicit limits** — whether a stated length or format constraint was met
- **Style map** — two principal components of the measured style features
- **Can you tell them apart?** — a classifier naming the author of answers from prompts it did not train on
- **The guessing game** — best guessers and most recognizable models, scored separately for the blind mode and the research mode
- **Usage recommendations** — which model fits a need, ranked from the measured traits (no judge model)
- **Same prompt, different answers** — the raw replies side by side

### The guessing game

For each prompt that every model answered, each model sees all of the answers with the author names removed and matches each answer to the roster. The prompt does not say which model is guessing. Letter order is shuffled separately for each guesser.

Three modes share `guesses.jsonl`. Each combination of mode, research source, `--independent` and `--profile` is scored as its own game:

| Mode | What the guesser sees |
|------|------------------------|
| **No extra context** (`--mode blind`, the default) | The prompt, the answers, and the roster names |
| **Based on the results** (`--mode briefed`) | The same, plus measured habits from a Spectra `analysis.json`: length, formatting, hedging, signature words, limit-following, and the strongest departures. Cost and latency are left out. The notes do not say which letter is which. |
| **Held-out results** (`--mode heldout`) | Like `briefed`, but the prompts split in half within each category, and each half is guessed with notes measured on the other half. No prompt or answer in front of the guesser went into its notes. |

By default every model must be used once, so placing one answer narrows the rest. `--independent` asks for a name per answer, judged on its own, with repeats allowed. Guessers can still choose to use each model once, so the report shows how often a call actually repeated a name, and how guessers did on the others when they did and did not find their own answer.

`--profile` adds parts to the plain brief, comma-separated or `all`: `portrait` (a paragraph per model before the trait table), `pairs` (how to tell apart the pairs an author classifier trained on the same source confuses most), `examples` (typical opening and closing lines per model, from different categories) and `category` (each model's habits in the category of the prompt being guessed). Every part is built from the same source as the brief, a held-out half or another revision, never from the answers being guessed or from guessing results. Each profile is scored as its own game, and `--dry-run` writes the profile to `profile_preview-<profile>.txt` and estimates the cost from earlier calls in any revision.

```bash
python -m spectra.guess --revision oct-23-26 --limit 5              # blind smoke test
python -m spectra.guess --revision oct-23-26                        # no extra context
python -m spectra.guess --revision oct-23-26 --mode briefed         # profile from this same revision
python -m spectra.guess --revision oct-23-26 --mode briefed --research oct-14-26
python -m spectra.guess --revision oct-23-26 --mode heldout --dry-run   # messages + cost estimate, no calls
python -m spectra.guess --revision oct-23-26 --mode heldout             # the clean test
python -m spectra.guess --revision oct-14-26 --mode heldout --limit 100 --profile examples,pairs --dry-run
python -m spectra.guess --revision oct-23-26 --mode heldout --profile pairs --guessers gemini-4-argon
python -m spectra.guess --revision oct-23-26 --mode heldout --compare-profiles --guessers gemini-4-argon
python -m spectra.report --revision oct-23-26                       # one section per game
```

`--guessers` lets only some models guess; every model's answers are still shown, and the game's summary covers the guessers that played it. That makes profile development cheap: one strong guesser plays each variant, and `--compare-profiles` (no calls) prints its accuracy with every profile against the plain one, paired on the prompts both games share.

`--research` chooses which profile `briefed` pastes in. It defaults to the same revision as the answers, so each answer is part of the averages it is guessed with. An earlier revision avoids that, but revisions that use the same prompt file share every prompt. `heldout` is the clean test: it writes each half's answers and codes to `data/spectra/<revision>/holdout/` and runs `spectra.analyze` there. `--dry-run` builds every message and estimates the cost from earlier calls without calling a model.

A full game is one call per guesser per prompt (1,200 calls for 200 prompts and 6 models). Each game is resumable on its own, and a lock file stops a second run from writing `guesses.jsonl` at the same time. Run `spectra.analyze` before `briefed`, then regenerate the report. The report reads `guesses.jsonl` and does not rerun analysis.

| Score | Meaning |
|-------|---------|
| **Best guessers** | Share of the *other* models' answers this model named correctly |
| **Most recognizable** | Share of the *other* models who named this author |
| **Found own answer** | How often a model picks its own unlabeled answer (reported separately) |

Chance is 1/k, where k is the roster size. A model's own answer is left out of both headlines, and every score has a 95% interval that resamples prompts. Ambiguous short names such as "claude" or "gemini" count as a miss. "opus" and "flash" count when they match one model.

Each game also reports the majority vote of the other guessers, and the same votes weighted by each guesser's accuracy on the other half of the prompts with each model named once per prompt, plus the most confused pairs and accuracy by prompt category. Guess calls get their own output budget, `guess_max_tokens` in `config.yaml`.

### Author classifiers

`spectra.attribute` is the reference the guessers are measured against. It runs locally and makes no API calls. Three logistic regressions, with settings fixed in advance, name the author of each answer from its text:

- word 1–2-grams with the style features (the classifier in `analysis.json`)
- character 2–5-grams
- both together

Each is scored one answer at a time and with a prompt's answers named jointly, each model once, as a guesser does. It is tested with 5 folds grouped by prompt, on the game's held-out halves and, with `--train-revision`, on another run's half A against this run's half B. `--what ablations` retrains the character classifier on text with layers removed (markdown, punctuation, case and line breaks, word order, everything after the first 50 words). `--what curve` trains it on 10 to 160 answers per model.

```bash
python -m spectra.attribute --revision oct-23-26 --train-revision oct-14-26
python -m spectra.attribute --revision oct-23-26 --what ablations curve
```

### Self-knowledge probe

`spectra.describe` asks each panel model to describe its own writing and every other model's. Each reply holds a short written profile and numeric estimates of the traits Spectra measures. That is 36 calls, about $1. The estimates are scored against a run's `analysis.json` with rank- and sign-based measures:

- rank agreement with the measured panel order
- places off, for a model's own placement against the others' placement of it
- the right side of the panel average
- a same-vendor gap test for brand stereotypes

`--sample N` collects a fresh set with the panel and the estimates listed in a shuffled order. `--measured REV` rescores against another run without calls. `confirm()` runs the pre-registered tests.

```bash
python -m spectra.describe --revision oct-23-26 --dry-run
python -m spectra.describe --revision oct-23-26 --sample 1
python -m spectra.describe --revision oct-23-26 --sample 1 --measured oct-14-26
```

### Pre-registration

`spectra/prereg/` holds pre-registrations, each committed and pushed before the data it covers exist. Results go in a separate file next to it. The tests a pre-registration names (`describe.confirm`, `guess.profile_contrasts`) do not change once data exist, and deviations are appended to the pre-registration with their reason.

### Output

```
data/spectra/<revision>/
├── responses.jsonl     # One answer per model, prompt, and sample
├── codes.jsonl         # Present after code_behaviors
├── features.csv        # Style features per answer
├── analysis.json       # Traits, style map, classifier
├── guesses.jsonl       # Present after the guessing game
├── guess_preview.txt   # First message of the last --dry-run (profile_preview-<profile>.txt: its profile)
├── holdout/half-A|B/   # Each half's answers, codes and analysis, written by --mode heldout
├── tiebreak.jsonl      # Third-coder labels on disputed answers (spectra.tiebreak); tiebreak.json: summary
├── descriptions.jsonl  # Self-knowledge probe (spectra.describe); descriptions[-sN][-vs-<rev>].json: scores and tests
├── attribution.json    # Author classifiers on the text (spectra.attribute, local)
├── report.html         # The report
└── run_config.json     # Roster and settings for the run
```

### Caveats

The prompts are synthetic and English-only. Effect sizes are relative to the models in that run, so adding or removing a model moves everyone. Trap results rest on a small number of prompts each, which is enough to see large differences and too thin for fine ones.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run the health check (`python peerrank.py --health`)
4. Commit and push your changes
5. Open a Pull Request

### Adding a New Provider

1. Add model entry to `ALL_MODELS` in `peerrank/models.py` (includes provider, model_id, name, and cost)
2. Implement `call_{provider}()` in `peerrank/providers.py`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{peerrank2026,
  title = {PeerRank: LLM Peer Evaluation System},
  author = {Caura AI},
  year = {2026},
  url = {https://github.com/caura-ai/caura-PeerRank}
}
```
