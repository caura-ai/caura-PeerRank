# PeerRank

**LLM Peer Evaluation System** - Models generate questions, answer them with web search, cross-evaluate each other's responses, and produce a ranked report with bias analysis.

## Features

- **5-Phase Pipeline**: Question generation -> Answering -> Cross-evaluation -> Report -> Analysis
- **12 Models**: OpenAI, Anthropic, Google, xAI, DeepSeek, Together AI, Perplexity, Moonshot AI, Mistral
- **Bias Detection**: Measures self-bias, name bias, and position bias through controlled evaluation modes
- **Web Search Integration**: Native search support for most providers, Tavily fallback for others
- **Cost Tracking**: Real-time token usage and cost analysis per model
- **Publication Figures**: Generate publication-quality charts and statistical analysis
- **TruthfulQA Validation**: Correlate peer rankings with ground truth accuracy

## Prerequisites

- Python 3.10+
- API keys for LLM providers you want to test (see [API Keys](#api-keys))

## Installation

```bash
git clone https://github.com/caura-ai/caura-PeerRank.git
cd caura-PeerRank
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
python peerrank.py --judge gpt-5.2   # Select judge model for Phase 5
python peerrank.py --rev v2     # Set revision tag for output files
python peerrank.py --health     # API health check
streamlit run peerrank_ui.py    # Launch Streamlit UI
python generate_figures_phase4.py --revision v1 --output figures/  # Generate publication figures
python generate_figures_TFQ.py --output figures/              # Generate TFQ validation figures
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
generate_figures_phase4.py   # Publication-quality figure generation
generate_figures_TFQ.py      # TruthfulQA validation figures
truthful.py                  # TruthfulQA validation
data/
  phase1_questions_{rev}.json
  phase2_answers_{rev}.json
  phase3_rankings_{rev}.json
  phase4_report_{rev}.md
  phase5_analysis_{rev}.md
  TRUTH/                     # TruthfulQA validation output files
```

## 5-Phase Pipeline

1. **Phase 1**: Each model generates questions across active categories
2. **Phase 2**: All models answer all questions (web search configurable, default ON)
3. **Phase 3**: Each model evaluates all responses in 3 bias modes:
   - `shuffle_only`: Randomized order, real model names shown
   - `blind_only`: Fixed order, model names hidden (Response A, B, C...)
   - `shuffle_blind`: Both randomized order + hidden names
4. **Phase 4**: Generate markdown report with rankings and bias analysis
5. **Phase 5**: Judge LLM analyzes the report and provides comprehensive insights

## Models

12 models across 8 providers:

| Provider | Models |
|----------|--------|
| OpenAI | gpt-5.2, gpt-4.1-mini |
| Anthropic | claude-opus-4-5, claude-sonnet-4-5 |
| Google | gemini-2.5-pro, gemini-2.5-flash |
| xAI | grok-4 |
| DeepSeek | deepseek-chat |
| Together AI | llama-4-maverick |
| Perplexity | sonar-pro |
| Moonshot AI | kimi-k2-0905 |
| Mistral | mistral-large |

## Categories

- Current events (needs recent info)
- Factual knowledge
- Reasoning/logic
- Creative/open-ended
- Practical how-to

Filter by keyword: `--categories factual,logic` matches categories containing those words.

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

## Bias Analysis

Phase 3 automatically runs **3 evaluation passes** to detect **3 types of bias**:

| Bias Type | Cause | Formula | Interpretation |
|-----------|-------|---------|----------------|
| **Self Bias** | Evaluator rates own answers | Self - Peer | + overrates self, - underrates self |
| **Name Bias** | Brand/model recognition | Shuffle - Peer | + name helped, - name hurt |
| **Position Bias** | Fixed order in answer list | Blind - Peer | + position helped, - position hurt |

**Evaluation Modes:**

| Mode | Order | Names | Purpose |
|------|-------|-------|---------|
| `shuffle_only` | Random | Visible | Measure name effect (vs baseline) |
| `blind_only` | Fixed | Hidden | Measure position effect (vs baseline) |
| `shuffle_blind` | Random | Hidden | **Baseline** (Peer score) |

## Report Sections (Phase 4)

- **Model Order**: Fixed presentation order used in blind evaluation
- **Phase Timing**: Duration of each phase with Phase 3 mode breakdown
- **Question Analysis**: By category, by source model, category coverage matrix
- **Answer/Evaluation Response Time**: Average response time per model
- **Answering API Cost Analysis**: Total costs, token usage, and per-answer costs
- **Performance vs. Cost**: Efficiency rankings (Points^2/cent)
- **Final Peer Rankings**: Scores from shuffle+blind mode (excluding self-ratings)
- **Bias Analysis**: Three bias types with Position Bias and Model Bias tables
- **Judge Generosity**: How lenient/strict each model judges
- **Judge Agreement Matrix**: Pairwise correlation between judges
- **Question Autopsy**: Hardest, easiest, most controversial questions

## TruthfulQA Validation

Correlates peer rankings with TruthfulQA ground truth:

```bash
python truthful.py                    # Interactive menu
python truthful.py --all              # Run all phases
python truthful.py --num-questions 50 # Set question count
```

## Figure Generation

Publication-quality figures for research papers:

```bash
python generate_figures_phase4.py --revision v1 --output figures/
python generate_figures_TFQ.py --output figures/
```

Generates PDF + 600 DPI PNG outputs with colorblind-safe palettes.

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
  author = {Caura AI},
  year = {2026},
  url = {https://github.com/caura-ai/caura-PeerRank}
}
```
