# PeerRank

**Do LLMs agree on which LLM is best?** PeerRank lets AI models evaluate each other through a structured peer review process—generating questions, answering them with web search, and cross-evaluating responses to produce ranked results with bias analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Key Features

- **5-Phase Pipeline** — Question generation → Answering → Cross-evaluation → Report → Analysis
- **12 Models Supported** — OpenAI, Anthropic, Google, xAI, DeepSeek, Together AI, Perplexity, Moonshot AI, Mistral
- **Bias Detection** — Measures self-bias, name bias, and position bias through controlled evaluation modes
- **Web Search Integration** — Native search for most providers, Tavily fallback for others
- **Cost Tracking** — Real-time token usage and cost analysis per model
- **Publication Figures** — Generate publication-quality charts and statistical analysis
- **Ground Truth Validation** — TruthfulQA (r=0.858) and GSM8K (r=0.986) correlation with accuracy

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
python peerrank.py --models gpt-5.2,claude-opus-4-5     # Include only these
python peerrank.py --exclude deepseek-chat              # Exclude these

# Configuration
python peerrank.py --web-search off     # Disable web search (test pure knowledge)
python peerrank.py --seed 42            # Reproducible shuffle ordering
python peerrank.py --judge claude-opus-4-5   # Set Phase 5 judge model
python peerrank.py --rev v2             # Set revision tag for output files

# UI and figures
streamlit run peerrank_ui.py                              # Launch web UI
python generate_figures_PeerRank.py --revision v1           # Publication figures
```

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-5.2, gpt-5-mini |
| Anthropic | claude-opus-4-5, claude-sonnet-4-5 |
| Google | gemini-3-pro-preview, gemini-3-flash-preview |
| xAI | grok-4-1-fast |
| DeepSeek | deepseek-chat |
| Together AI | llama-4-maverick |
| Perplexity | sonar-pro |
| Moonshot AI | kimi-k2-0905 |
| Mistral | mistral-large |

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

# Required for web search with DeepSeek/Together/Kimi
TAVILY_API_KEY=tvly-...
```

You only need keys for providers you want to test. Use `--models` to select specific models.

## Output

Results are saved to the `data/` directory with revision tags:

```
data/
├── phase1_questions_v1.json    # Generated questions
├── phase2_answers_v1.json      # Model responses (with cost tracking)
├── phase3_rankings_v1.json     # Cross-evaluations (3 bias modes)
├── phase4_report_v1.md         # Markdown report with rankings
└── phase5_analysis_v1.md       # Judge analysis and insights
```

### Sample Report Sections

- **Final Peer Rankings** — Scores from blind+shuffled evaluation
- **Elo Ratings** — Pairwise comparison rankings with W-L-T records
- **Bias Analysis** — Self-bias, name bias, and position bias metrics
- **Ablation Study** — Effect of bias correction on ground truth correlation
- **Judge Generosity** — Which models rate harshly vs. leniently
- **Performance vs. Cost** — Efficiency rankings (Points²/¢)
- **Question Autopsy** — Hardest, easiest, and most controversial questions

## Project Structure

```
peerrank/                # Core package (pip installable)
  __init__.py            # Package exports
  config.py              # Settings, model configs, utilities
  providers.py           # LLM API implementations
peerrank.py              # CLI entry point
peerrank_ui.py           # Streamlit web interface
peerrank_phase1-5.py     # Pipeline phases
generate_figures_*.py    # Publication figure generation
validate_truthfulqa.py   # TruthfulQA validation
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
```

**Ablation Study**: Bias correction improves correlation with ground truth by +0.285 (Pearson).

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run the health check (`python peerrank.py --health`)
4. Commit and push your changes
5. Open a Pull Request

### Adding a New Provider

1. Add model config to `ALL_MODELS` in `peerrank/config.py`
2. Implement `call_{provider}()` in `peerrank/providers.py`
3. Add token costs to `TOKEN_COSTS` in `peerrank/config.py`

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
