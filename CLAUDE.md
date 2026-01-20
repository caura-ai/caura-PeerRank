# PeerRank

**LLM Peer Evaluation System** - Models generate questions, answer them with web search, cross-evaluate each other's responses, and produce a ranked report with bias analysis.

## Features

- **5-Phase Pipeline**: Question generation → Answering → Cross-evaluation → Report → Analysis
- **12 Models**: OpenAI, Anthropic, Google, xAI, DeepSeek, Together AI, Perplexity, Moonshot AI, Mistral
- **Bias Detection**: Measures self-bias, name bias, and position bias through controlled evaluation modes
- **Elo Ratings**: Alternative ranking via pairwise comparisons (K=32, excludes self-evaluations)
- **Web Search Integration**: Native search support for most providers, Tavily fallback for others
- **Cost Tracking**: Real-time token usage and cost analysis per model
- **Publication Figures**: Generate publication-quality charts and statistical analysis
- **TruthfulQA Validation**: Correlate peer rankings with ground truth accuracy
- **GSM8K Validation**: Correlate peer rankings with math accuracy (r=0.986)

## Prerequisites

- Python 3.10+
- API keys for LLM providers you want to test (see [API Keys](#api-keys))

## Installation

```bash
git clone https://github.com/yourusername/peerrank.git
cd peerrank
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Quick Start

```bash
python peerrank.py              # Interactive menu
python peerrank.py --all        # Run all 5 phases
python peerrank.py --health     # Test API connectivity
```

## Commands

```bash
python peerrank.py              # Interactive menu
python peerrank.py --phase 1    # Run specific phase (1-5)
python peerrank.py --all        # Run all phases (1-5)
python peerrank.py --resume     # Resume from last completed
python peerrank.py --models gpt-5.2,claude-opus-4-5    # Include only these models
python peerrank.py --exclude gemini-3-pro-preview      # Exclude these models
python peerrank.py --categories factual,reasoning      # Include only these categories
python peerrank.py --exclude-categories creative       # Exclude these categories
python peerrank.py --seed 42    # Reproducible shuffle ordering for Phase 3
python peerrank.py --web-search on   # Enable Phase 2 web search (default)
python peerrank.py --web-search off  # Disable Phase 2 web search (test knowledge only)
python peerrank.py --elo on          # Enable Phase 4 Elo ratings (default)
python peerrank.py --elo off         # Disable Phase 4 Elo ratings
python peerrank.py --judge gpt-5.2   # Select judge model for Phase 5
python peerrank.py --rev v2     # Set revision tag for output files
python peerrank.py --health     # API health check
streamlit run peerrank_ui.py    # Launch Streamlit UI
python generate_figures_PeerRank.py --revision v1 --output figures/  # Generate publication figures
python generate_figures_TFQ.py --output figures/              # Generate TFQ validation figures
python gsm8k.py --all --num-questions 50                      # Run GSM8K math validation
python gsm8k.py --difficulty hard --num-questions 20          # GSM8K with hard questions only
python peerrank_build_web/build.py                            # Build website
python peerrank_build_web/build.py --serve                    # Build + local server
python peerrank_build_web/build.py --watch                    # Build + server + latency monitor
```

## Interactive Menu

```
  Revision: v1
  Progress: Phase 0/5 completed
  Models: 3/10 - gpt-5.2, claude-opus-4-5, gemini-3-pro-preview
  Categories: 5/5 active
  Questions per model: 2
  Phase 2 web search: ON
  Phase 3: 3 passes (random)
  Phase 4 Elo ratings: ON
  Phase 5 judge: gpt-5.2

  [1] Phase 1 - Generate Questions
  [2] Phase 2 - Answer Questions
  [3] Phase 3 - Cross-Evaluate (3 bias modes)
  [4] Phase 4 - Generate Report
  [5] Phase 5 - Final Analysis
  [A] Run ALL phases
  [R] Resume from last completed
  [H] Health Check - Test all APIs
  [M] Models - Select which models to run
  [C] Categories - Select question categories
  [N] Number of questions per model
  [W] Web Search - Toggle Phase 2 grounding
  [D] Seed - Set random seed for Phase 3
  [E] Elo - Toggle Phase 4 Elo ratings
  [J] Judge - Select Phase 5 analysis judge
  [V] Version - Set revision tag
  [Q] Quit
```

## Architecture

```
peerrank.py          # CLI entry point
peerrank_ui.py       # Streamlit UI (live comparison)
config.py            # Settings, model configs, utilities
providers.py         # LLM API calls with web search
phase1.py            # Question generation
phase2.py            # Answer questions (web search configurable)
phase3.py            # Cross-evaluation (web search OFF, 3 bias modes)
phase4.py            # Report generation
phase5.py            # Final analysis by judge LLM
generate_figures_PeerRank.py   # Publication-quality figure generation (Figs 4-6, 10-17)
generate_figures_TFQ.py      # TruthfulQA validation figures (Figs 10-14)
truthful.py                  # TruthfulQA validation (correlate peer rankings with ground truth)
gsm8k.py                     # GSM8K validation (correlate peer rankings with math accuracy)
peerrank_build_web/          # Website source files
  build.py                   # Unified build script (generate + serve + monitor)
  generate_webpage.py        # HTML generator from Phase 4 report
  latency_monitor.py         # Live latency checker for all models
  template.html              # Jinja2 template
  styles.css                 # CSS styles
peerrank_hosted_website/     # Built output (for GitHub Pages or hosting)
  index.html                 # Generated HTML
  styles.css                 # Copied CSS
  latency.json               # Live latency data
data/
  phase1_questions_{rev}.json
  phase2_answers_{rev}.json
  phase3_rankings_{rev}.json
  phase4_report_{rev}.md
  phase5_analysis_{rev}.md
  TRUTH/                     # TruthfulQA validation output files
  GSM8K/                     # GSM8K validation output files
```

## Revision System

Files are tagged with user-set revision (default: `v1`). Change via `[V]` menu option.
- Each revision is a separate run
- `load_json` and `get_last_completed_phase` use current revision
- Allows multiple evaluation runs side-by-side

## 5-Phase Pipeline

1. **Phase 1**: Each model generates questions across active categories
2. **Phase 2**: All models answer all questions (web search configurable, default ON)
3. **Phase 3**: Each model evaluates all responses in 3 bias modes:
   - `shuffle_only`: Randomized order, real model names shown
   - `blind_only`: Fixed order, model names hidden (Response A, B, C...)
   - `shuffle_blind`: Both randomized order + hidden names
4. **Phase 4**: Generate markdown report with rankings and bias analysis
5. **Phase 5**: Judge LLM analyzes the report and provides comprehensive insights

## Report Sections (Phase 4)

- **Model Order**: Fixed presentation order (1-10) used in blind evaluation
- **Phase Timing**: Duration of each phase with Phase 3 mode breakdown
- **Question Analysis**: By category, by source model, category coverage matrix
- **Answer/Evaluation Response Time**: Average response time per model
- **Answering API Cost Analysis**: Total costs, token usage, and per-answer costs for Phase 2
- **Performance vs. Cost**: Efficiency rankings combining quality scores with cost (Points²/¢)
- **Final Peer Rankings**: Scores from shuffle+blind mode (excluding self-ratings)
- **Elo Ratings**: Pairwise comparison rankings with W-L-T records and rank comparison
- **Bias Analysis**: Three bias types with Position Bias table and Model Bias table
- **Judge Generosity**: How lenient/strict each model judges
- **Judge Agreement Matrix**: Pairwise correlation between judges' scoring patterns
- **Question Autopsy**: Hardest, easiest, most controversial, and consensus questions
- **Performance Overview**: ASCII chart of scores vs response time

## Analysis Report (Phase 5)

Judge LLM (configurable, default: gpt-5.2) analyzes the Phase 4 report and provides:
- **Overall Quality Assessment**: Holistic evaluation of the peer ranking results
- **Top Performers & Outliers**: Identification of standout models and anomalies
- **Bias Patterns**: Analysis of self-bias, name bias, and position bias trends
- **Judge Generosity Comparison**: Which models are harsh vs. lenient evaluators
- **Performance vs. Cost Insights**: Efficiency analysis and value recommendations
- **Media Headlines**: 5 attention-grabbing news-style headlines with specific numbers

Configuration:
- `get_phase5_judge()` / `set_phase5_judge(provider, model_id, display_name)`
- Default: `("openai", "gpt-5.2", "gpt-5.2")`
- CLI: `python peerrank.py --judge gpt-5.2`
- Menu: `[J] Judge - Select Phase 5 analysis judge`

## Models

```python
ALL_MODELS = [
    ("openai", "gpt-5.2", "gpt-5.2"),
    ("openai", "gpt-4.1-mini", "gpt-4.1-mini"),
    ("anthropic", "claude-opus-4-5", "claude-opus-4-5"),
    ("anthropic", "claude-sonnet-4-5", "claude-sonnet-4-5"),
    ("google", "gemini-2.5-pro", "gemini-2.5-pro"),
    ("google", "gemini-2.5-flash", "gemini-2.5-flash"),
    ("grok", "grok-4", "grok-4"),
    ("deepseek", "deepseek-chat", "deepseek-chat"),
    ("together", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "llama-4-maverick"),
    ("perplexity", "sonar-pro", "sonar-pro"),
    ("kimi", "kimi-k2-0905-preview", "kimi-k2-0905"),
    ("mistral", "mistral-large-latest", "mistral-large"),
]
```

Total: **12 models** across 8 providers (OpenAI, Anthropic, Google, xAI, DeepSeek, Together AI, Perplexity, Moonshot AI, Mistral)

## Categories

```python
ALL_CATEGORIES = [
    "current events (needs recent info)",
    "factual knowledge",
    "reasoning/logic",
    "creative/open-ended",
    "practical how-to",
]
```

Filter by keyword: `--categories factual,logic` matches categories containing those words.

## Provider Implementations

All calls route through `call_llm()` in providers.py:
- **OpenAI**: Responses API for web search, Chat Completions otherwise
- **Anthropic**: web-search-2025-03-05 beta header
- **Google**: Vertex AI (service account) or API key with google_search tool
- **Perplexity**: Native web search in Sonar models
- **Grok**: Native xAI SDK with web_search + x_search tools
- **Mistral**: Native Agents API with web_search connector
- **DeepSeek/Together/Kimi**: Tavily search augmentation

## API Keys

Create a `.env` file in the project root with your API keys:

```bash
# Required: At least one LLM provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROK_API_KEY=xai-...
DEEPSEEK_API_KEY=sk-...
TOGETHER_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
KIMI_API_KEY=sk-...
MISTRAL_API_KEY=...

# Required for web search with DeepSeek/Together/Kimi
TAVILY_API_KEY=tvly-...
```

You only need keys for the providers you want to test. Use `--models` to select specific models.

## Bias Analysis (Phase 3)

Phase 3 automatically runs **3 evaluation passes** to detect **3 types of bias**:

**UNIFIED CONVENTION: Positive = factor HELPED the model**

| Bias Type | Cause | Formula | Interpretation |
|-----------|-------|---------|----------------|
| **Self Bias** | Evaluator rates own answers | Self − Peer | + overrates self, − underrates self |
| **Name Bias** | Brand/model recognition | Shuffle − Peer | + name helped, − name hurt |
| **Position Bias** | Fixed order in answer list | Blind − Peer | + position helped, − position hurt |

**Evaluation Modes:**

| Mode | Order | Names | Purpose |
|------|-------|-------|---------|
| `shuffle_only` | Random | Visible | Measure name effect (vs baseline) |
| `blind_only` | Fixed | Hidden | Measure position effect (vs baseline) |
| `shuffle_blind` | Random | Hidden | **Baseline** (Peer score) |

**Bias Formulas Explained:**
- `shuffle_blind` = baseline Peer score (both biases removed)
- `shuffle_only` = name visible → Name Bias = shuffle_only − shuffle_blind
- `blind_only` = fixed order → Position Bias = blind_only − shuffle_blind

**Usage:**
```bash
python peerrank.py --phase 3           # Run all 3 modes
python peerrank.py --phase 3 --seed 42 # Reproducible ordering
python generate_figures_PeerRank.py --revision v1  # Generate bias figures
```

**Report output:**

Position Bias table (by position, not model):
| Pos | Blind Score | Pos Bias |
|-----|-------------|----------|
| 1   | 8.56        | +0.29    |

Model Bias table:
| Model | Peer | Self | Self Bias | Shuffle | Name Bias |
|-------|------|------|-----------|---------|-----------|
| gpt-5.2 | 8.27 | 8.71 | +0.44 | 8.56 | +0.29 |

**Data structure in phase3_rankings.json:**
```json
{
  "mode_durations": {"shuffle_only": 255.0, "blind_only": 258.0, "shuffle_blind": 260.0},
  "evaluations_by_mode": {
    "shuffle_only": {...},
    "blind_only": {...},
    "shuffle_blind": {...}
  },
  "evaluations": {...}  // backward compat (uses shuffle_blind)
}
```

## Cost Tracking (Jan 2026)

**Token Costs**: Defined in `config.py` - `TOKEN_COSTS` dictionary with input/output pricing per million tokens

Updated Jan 2026 pricing for all models:
```python
TOKEN_COSTS = {
    # OpenAI
    "gpt-5.2": (1.75, 14.00),
    "gpt-4.1-mini": (0.40, 1.60),

    # Anthropic (with prompt caching, cache reads are 90% off)
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-5": (3.00, 15.00),

    # Google Gemini
    "gemini-2.0-flash": (0.10, 0.40),    # Very fast, huge quotas
    "gemini-2.0-pro": (2.00, 12.00),     # High intelligence
    "gemini-2.5-pro": (1.25, 10.00),     # Smart "Thinking" model
    "gemini-2.5-flash": (0.15, 0.60),    # Fast "Workhorse" model

    # xAI
    "grok-4": (3.00, 15.00),

    # DeepSeek (with auto caching)
    "deepseek-chat": (0.28, 0.42),

    # Together AI
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.27),

    # Perplexity (includes native web search)
    "sonar-pro": (3.00, 15.00),

    # Moonshot AI (Kimi) - with auto caching
    "kimi-k2-0905-preview": (0.60, 2.50),
    "kimi-k2-0711-preview": (0.60, 2.50),
    "kimi-k2-turbo-preview": (1.15, 8.00),
    "kimi-k2-thinking": (0.60, 2.50),

    # Mistral AI
    "mistral-large-latest": (2.00, 6.00),
}
```

**Phase 2 Tracking**:
- Each answer stores: `{text, input_tokens, output_tokens, cost}`
- Real-time cost display: `(avg 2.34s/q, $0.0023/q)`
- Output JSON includes: `cost_stats`, `total_cost`, `web_search` flag
- Per-model tracking: total cost, input/output tokens, call count

**Phase 4 Report**:
- **Answering API Cost Analysis**: Total costs, tokens, avg cost per question
- **Performance vs. Cost**: Efficiency metric `(Score ^ EXPONENT) / Cost_cents`
  - `EFFICIENCY_QUALITY_EXPONENT` in config.py (default 2.0)
  - Higher exponent = stronger quality weighting
  - Formula: Points²/¢ rewards high-quality models
  - Dynamic superscript display in report headers (², ¹·⁵, etc.)

**Cost Calculation**:
```python
calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float
```
Returns total cost in USD using TOKEN_COSTS pricing table

**Provider Behavior**: All `call_llm()` calls return `(content, duration, input_tokens, output_tokens)`

## Web Search Control (Phase 2)

**Configuration**:
- `PHASE2_WEB_SEARCH` global setting in config.py (default: True)
- Functions: `get_phase2_web_search()` / `set_phase2_web_search(enabled: bool)`
- CLI: `python peerrank.py --web-search on/off`
- Menu: `[W] Web Search - Toggle Phase 2 grounding`

**Use Cases**:
- `--web-search on` (default): Test models with external knowledge augmentation
- `--web-search off`: Test pure model knowledge without search assistance
- Compare performance: Run two revisions (web vs no-web) for A/B testing

**Report Indicator**: Phase 4 header shows "Web search: ON/OFF" status

## Elo Ratings (Phase 4)

Alternative ranking methodology using pairwise comparisons from evaluation scores.

**Configuration**:
- `PHASE4_ELO` global setting in config.py (default: True)
- Functions: `get_phase4_elo()` / `set_phase4_elo(enabled: bool)`
- CLI: `python peerrank.py --elo on/off`
- Menu: `[E] Elo - Toggle Phase 4 Elo ratings`

**Algorithm**:
- Initial rating: 1500 (configurable via `ELO_INITIAL_RATING`)
- K-factor: 32 (configurable via `ELO_K_FACTOR`)
- Expected score: `E_a = 1 / (1 + 10^((R_b - R_a) / 400))`
- Rating update: `R_a' = R_a + K * (actual - E_a)`

**Pairwise Conversion**:
For each (evaluator, question), scores are converted to C(N,2) pairwise matches:
- `score_a > score_b` → A wins (1.0, 0.0)
- `score_a < score_b` → B wins (0.0, 1.0)
- `score_a == score_b` → Tie (0.5, 0.5)
- Self-evaluations excluded by default

**Report Output**:
| # | Model | Elo | Win% | W-L-T | Peer | P# | Diff |
|---|-------|-----|------|-------|------|----|------|
| 1 | gpt-5.2 | 1687 | 68.2% | 9012-4188-786 | 8.27 | 1 | 0 |

- **Elo**: Final Elo rating
- **Win%**: Win rate including ties (wins + 0.5*ties / total)
- **W-L-T**: Wins-Losses-Ties record
- **Peer**: Peer score from averaging method
- **P#**: Peer ranking position
- **Diff**: Peer rank − Elo rank (positive = Elo ranks model higher)

**Data Volume** (typical):
- 12 evaluators × 60 questions × C(10,2) = ~32,400 pairwise matches
- Sufficient for Elo convergence

**Functions** (in config.py):
```python
calculate_elo_ratings(evaluations, model_names=None, initial_rating=1500,
                      k_factor=32, exclude_self=True, seed=None)
# Returns: {ratings, matches, win_rates, total_matches}
```

## Key Patterns

- Async batch processing: `asyncio.gather()`, batch size 5
- Models: 3-tuples `(provider, model_id, display_name)`
- Iterate: `for provider, model_id, name in MODELS`
- 130s timeout, 3 retries with exponential backoff
- Revision: `get_revision()` / `set_revision(rev)` in config.py
- Phase 3 seed: `set_bias_test_config(seed=N)` / `get_bias_test_config()` in config.py
- Phase 5 judge: `set_phase5_judge(provider, model_id, name)` / `get_phase5_judge()` in config.py
- Answer length: `MAX_ANSWER_WORDS` in config.py (default 200) limits Phase 2 response length
- Phase 3 progress: Shows batch completion with avg time per question
- Temperature overrides: `MODEL_TEMPERATURE_OVERRIDES` for model-specific adjustments

## Shared Constants & Functions (config.py)

```python
# Bias modes (tuples for backend)
BIAS_MODES = [
    ("shuffle_only", True, False),   # (name, shuffle, blind)
    ("blind_only", False, True),
    ("shuffle_blind", True, True),
]

# Bias configs (dicts for UI with icons/descriptions)
BIAS_CONFIGS = [
    {"name": "shuffle_only", "shuffle": True, "blind": False, "icon": "🔀", "desc": "..."},
    ...
]

# UI display modes (subset for presentation)
UI_DISPLAY_MODES = [
    {"name": "shuffle_blind", "display_name": "Peer Score", "icon": "🏆", ...},
    {"name": "shuffle_only", "display_name": "Shuffle (names visible)", "icon": "🔀", ...},
]

# Shared score calculation (used by phase4.py and peerrank_ui.py)
calculate_scores_from_evaluations(evaluations, model_names) -> {
    "peer_scores": {model: [scores]},
    "self_scores": {model: [scores]},
    "raw_scores": {model: [scores]},
    "judge_given": {model: [scores]},
}
```

Handles both data formats:
- Phase3: `{evaluator: {question: {model: {score, reason}}}}`
- UI: `{evaluator: {evaluator, scores: {model: {score, reason}}}}`

## Streamlit UI (peerrank_ui.py)

Live comparison interface with bias analysis. Structure mirrors Phase 4 report.

**Two Result Tables:**
| Table | Mode | Columns | Purpose |
|-------|------|---------|---------|
| Peer Score | shuffle_blind | Rank, Model, Peer, Self, Self Bias | Final ranking with self-favoritism |
| Shuffle | shuffle_only | Rank, Model, Score, Name Bias | Shows effect of hiding names |

**Bias Effect Analysis (2 columns):**
- **Position Bias**: By position number (1-10), not model name. Shows `Blind − Peer` (positive = position helped)
- **Self-Bias by Mode**: Average self-favoritism across all 3 modes (positive = overrates self)

## TruthfulQA Validation (`truthful.py`)
Correlates peer rankings with TruthfulQA ground truth to validate the peer evaluation methodology:
- 5-phase pipeline mirroring main PeerRank system
- Uses multiple choice questions with known correct answers
- Computes Pearson/Spearman correlation between peer scores and accuracy

**Usage**:
```bash
python truthful.py                    # Interactive menu
python truthful.py --all              # Run all phases
python truthful.py --phase 1-5        # Run specific phase
python truthful.py --num-questions 50 # Set question count
```

**Output files** (in `data/TRUTH/`):
- `phase1_questions_TFQ.json` - MC questions from TruthfulQA
- `phase1_ground_truth_TFQ.json` - Correct answers
- `phase2_answers_TFQ.json` - Model responses
- `phase3_rankings_TFQ.json` - Peer evaluations
- `phase4_TFQ_scores_TFQ.json` - Ground truth accuracy scores
- `TFQ_analysis_TFQ.json` - Correlation analysis
- `TFQ_validation_report_TFQ.md` - Final report

## GSM8K Validation (`gsm8k.py`)
Correlates peer rankings with GSM8K (Grade School Math 8K) ground truth to validate peer evaluation on mathematical reasoning:
- 5-phase pipeline mirroring main PeerRank system
- Uses open-ended math problems with numerical answers (not multiple choice)
- Extracts answers via `#### <number>` pattern with fallback regex patterns
- Computes Pearson/Spearman correlation between peer scores and math accuracy
- **Strong correlation observed**: r=0.986 (p<0.0001) in validation testing

**Key difference from TruthfulQA**: GSM8K uses open-ended problems requiring chain-of-thought reasoning with numerical answers, rather than multiple choice questions.

**Usage**:
```bash
python gsm8k.py                           # Interactive menu
python gsm8k.py --all                     # Run all phases
python gsm8k.py --phase 1-5               # Run specific phase
python gsm8k.py --num-questions 50        # Set question count
python gsm8k.py --difficulty easy,medium  # Filter by difficulty
python gsm8k.py --difficulty hard         # Only hard questions
```

**Difficulty levels** (based on solution step count):
- `easy`: 1-3 reasoning steps (708 questions available)
- `medium`: 4-5 reasoning steps (477 questions available)
- `hard`: 6+ reasoning steps (134 questions available)

**Output files** (in `data/GSM8K/`):
- `phase1_questions_GSM8K.json` - Math problems from GSM8K
- `phase1_ground_truth_GSM8K.json` - Gold answers with solutions
- `phase2_answers_GSM8K.json` - Model responses with extracted answers
- `phase3_rankings_GSM8K.json` - Peer evaluations
- `phase4_GSM8K_scores_GSM8K.json` - Ground truth accuracy scores
- `GSM8K_analysis_GSM8K.json` - Correlation analysis
- `GSM8K_validation_report_GSM8K.md` - Final report

**Answer extraction** (in order of preference):
1. `#### number` - Explicit format requested in prompt
2. `final answer is/= number` - Common model phrasing
3. `therefore/so/thus... number` - Reasoning conclusion
4. `\boxed{number}` - LaTeX format
5. Last standalone number - Fallback

### Figure Generation (`generate_figures_PeerRank.py`)
Publication-quality figure generation for research papers:
- PDF + 600 DPI PNG output
- Matplotlib with Times New Roman serif font
- Colorblind-safe model color palette
- Supports per-revision data extraction

**Usage**:
```bash
python generate_figures_PeerRank.py --revision v1 --output figures/
```

**Generates** (Figures 4-6, 10-18):
- Fig 4: Peer score rankings with error bars
- Fig 5: Cross-evaluation heatmap
- Fig 6: Peer score vs response time
- Fig 10: Self bias analysis
- Fig 11: Name bias analysis
- Fig 12: Position bias analysis
- Fig 13: Judge generosity
- Fig 14: Judge generosity vs peer ranking
- Fig 15: Judge agreement matrix (pairwise correlation heatmap)
- Fig 16: Question autopsy (difficulty vs controversy scatter)
- Fig 17: Radar chart (multi-dimensional comparison)
- Fig 18: Elo vs Peer ranking (slope graph with correlation stats)

### TFQ Figure Generation (`generate_figures_TFQ.py`)
Publication-quality figures for TruthfulQA validation analysis:
- PDF + 600 DPI PNG output
- Correlation scatter plots with regression lines
- Statistical analysis reports (text + JSON)

**Usage**:
```bash
python generate_figures_TFQ.py                    # Generate all figures
python generate_figures_TFQ.py --output figures/  # Custom output directory
python generate_figures_TFQ.py --stats-only       # Print stats without figures
```

**Generates** (Figures 10-14 for TFQ validation):
- `fig10_peerrank_correlation` - Scatter plot of peer vs truth scores with Pearson/Spearman correlation
- `fig11_score_comparison` - Side-by-side bar chart comparing peer and truth scores
- `fig12_rank_agreement` - Slope graph showing rank changes between methods
- `fig13_accuracy_ranking` - Horizontal bar chart of TruthfulQA accuracy by model
- `fig14_residual_analysis` - Over/under-rated models by peer evaluation

**Reports**:
- `TFQ_stats_report.txt` - Full statistical analysis with correlation, rank agreement, accuracy summary
- `TFQ_stats_summary.json` - Machine-readable summary for further analysis
- `TFQ_figures_latex.tex` - LaTeX templates for Overleaf integration

## Advanced Configuration (config.py)

### Token Limits
Model-specific maximum token limits for API calls:
```python
MAX_TOKENS_SHORT = 2048         # Short responses
MAX_TOKENS_ANSWER = 8192        # Phase 2 answers
MAX_TOKENS_EVAL = 12000         # Phase 3 evaluations
MAX_TOKENS_DEEPSEEK = 8192      # DeepSeek-specific limit
MAX_TOKENS_GOOGLE = 12000       # Google models limit
MAX_ANSWER_WORDS = 200          # Phase 2 answer word limit
```

### Temperature Settings
```python
TEMPERATURE_DEFAULT = 0.7       # Generation (Phase 1, 2)
TEMPERATURE_EVAL = 0            # Evaluation (Phase 3)

# Model-specific overrides for models that don't support certain values
MODEL_TEMPERATURE_OVERRIDES = {
    "gpt-5-mini": 1.0,          # GPT-5-mini doesn't support 0.7
}
```

### Retry & Timeout
```python
DEFAULT_TIMEOUT = 130           # API call timeout (seconds)
MAX_RETRIES = 3                 # Number of retry attempts
RETRY_DELAY = 2                 # Base delay between retries (exponential backoff)
```

### Debug Flags
```python
TAVILY_DEBUG = True             # Verbose logging for Tavily search augmentation
```

### Provider Concurrency
Maximum concurrent requests per provider for parallel processing:
```python
PROVIDER_CONCURRENCY = {
    "openai": 8, "anthropic": 8, "google": 8, "grok": 8,
    "deepseek": 8, "together": 8, "perplexity": 8, "kimi": 8,
}
```

### Utility Functions
Core helper functions used across multiple files:

**Scoring & Analysis**:
- `calculate_scores_from_evaluations(evaluations, model_names)` - Central scoring function
  - Returns: `{peer_scores, self_scores, raw_scores, judge_given}`
  - Handles both Phase3 and UI data formats
  - Used by: phase4.py, peerrank_ui.py, generate_figures_PeerRank.py
- `calculate_judge_agreement(evaluations)` - Pairwise correlation between judges
  - Returns: `{matrix, pairs, judges}`
  - Used by: phase4.py, generate_figures_PeerRank.py
- `calculate_question_stats(evaluations, questions)` - Question difficulty/controversy analysis
  - Returns: `{questions, hardest, easiest, controversial, consensus}`
  - Used by: phase4.py, generate_figures_PeerRank.py
- `_record_score(score, model_name, evaluator, ...)` - Categorizes scores as peer/self/raw

**Model Matching**:
- `match_model_name(name)` - Fuzzy matching for shortened model names
- `list_available_models()` - Returns list of all model display names
- `set_active_models(include=None, exclude=None)` - Filter active models

**Category Management**:
- `list_available_categories()` - Returns all available categories
- `set_active_categories(include=None, exclude=None)` - Filter active categories

**File I/O**:
- `save_json(filename, data)` - Save with revision tag to data directory
- `load_json(filename)` - Load with current revision tag
- `get_last_completed_phase()` - Detect highest completed phase for resume

**Formatting**:
- `format_duration(seconds)` - Human-readable duration (e.g., "2m 34.5s")
- `format_table(headers, rows, alignments)` - Markdown table with alignment control
- `extract_json(text)` - Robust JSON extraction from LLM responses
- `calculate_timing_stats(timing)` - Aggregate timing data with avg/total/count

**API Keys**:
- `get_api_key(provider)` - Fetch API key from environment variables

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the health check (`python peerrank.py --health`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Adding a New Provider

1. Add provider config to `ALL_MODELS` in `config.py`
2. Implement `call_{provider}()` in `providers.py`
3. Add token costs to `TOKEN_COSTS` in `config.py`
4. Update the health check in `peerrank.py`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use PeerRank in your research, please cite:

```bibtex
@software{peerrank2026,
  title = {PeerRank: LLM Peer Evaluation System},
  year = {2026},
  url = {https://github.com/yourusername/peerrank}
}
```
