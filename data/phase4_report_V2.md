# PeerRank.ai LLM Evaluation Report

Revision: **V2** | Generated: 2026-01-16 14:26:17

Models evaluated: 11 | Questions: 275 | Web search: **ON**

## Model Order and Web Mode

1. gpt-5.2 (Tool)
2. gpt-4.1-mini (Tool)
3. claude-opus-4-5 (Tool)
4. claude-sonnet-4-5 (Tool)
5. gemini-3-pro-preview (Tool)
6. gemini-3-flash-thinking (Tool)
7. grok-4-1-fast (Tool)
8. deepseek-chat (Tavily)
9. llama-4-maverick (Tavily)
10. sonar-pro (Native)
11. kimi-k2-0905 (Tavily)

## Phase Total Runtime
| Phase                |  Duration |
|----------------------|----------:|
| Phase 1 (Questions)  |   1m 0.8s |
| Phase 2 (Answers)    | 25m 47.3s |
| Phase 3 (Evaluation) | 130m 2.1s |
|   └─ Shuffle Only    |  41m 0.5s |
|   └─ Blind Only      |  42m 8.8s |
|   └─ Shuffle + Blind | 46m 52.8s |
| Phase 4 (Report)     |         — |

## Question Analysis
| Model                   | creative | current | factual | practica | reasonin | Total |
|-------------------------|:--------:|:-------:|:-------:|:--------:|:--------:|------:|
| gpt-5.2                 |    5     |    5    |    5    |    5     |    5     |    25 |
| gpt-4.1-mini            |    5     |    5    |    5    |    5     |    5     |    25 |
| claude-opus-4-5         |    5     |    5    |    5    |    5     |    5     |    25 |
| claude-sonnet-4-5       |    5     |    5    |    5    |    5     |    5     |    25 |
| gemini-3-pro-preview    |    5     |    5    |    5    |    5     |    5     |    25 |
| gemini-3-flash-thinking |    5     |    5    |    5    |    5     |    5     |    25 |
| grok-4-1-fast           |    5     |    5    |    5    |    5     |    5     |    25 |
| deepseek-chat           |    5     |    5    |    5    |    5     |    5     |    25 |
| llama-4-maverick        |    5     |    5    |    5    |    5     |    5     |    25 |
| sonar-pro               |    5     |    5    |    5    |    5     |    5     |    25 |
| kimi-k2-0905            |    5     |    5    |    5    |    5     |    5     |    25 |
| **Total**               |    55    |    55   |    55   |    55    |    55    |   275 |

## Answers
| Model                   | Avg Time | OK/Total |
|-------------------------|---------:|---------:|
| llama-4-maverick        |    3.05s |  275/275 |
| sonar-pro               |    3.40s |  275/275 |
| gpt-4.1-mini            |    3.41s |  275/275 |
| gemini-3-flash-thinking |    4.20s |  275/275 |
| gpt-5.2                 |    4.65s |  275/275 |
| deepseek-chat           |    7.04s |  275/275 |
| claude-opus-4-5         |    8.24s |  275/275 |
| grok-4-1-fast           |    8.56s |  275/275 |
| claude-sonnet-4-5       |    9.54s |  275/275 |
| kimi-k2-0905            |   12.32s |  275/275 |
| gemini-3-pro-preview    |   16.05s |  275/275 |

## Evaluations
| Model                   | Avg Time | OK/Total |
|-------------------------|---------:|---------:|
| sonar-pro               |    3.63s |  275/275 |
| gpt-4.1-mini            |    4.12s |  275/275 |
| gpt-5.2                 |    4.38s |  275/275 |
| llama-4-maverick        |    4.66s |  275/275 |
| claude-opus-4-5         |    5.07s |  275/275 |
| claude-sonnet-4-5       |    5.24s |  275/275 |
| deepseek-chat           |    8.61s |  275/275 |
| grok-4-1-fast           |   11.01s |  275/275 |
| gemini-3-flash-thinking |   14.29s |  275/275 |
| gemini-3-pro-preview    |   29.81s |  274/275 |
| kimi-k2-0905            |   30.38s |  273/275 |

## Final Peer Rankings (Shuffle + Blind mode)

Scores from peer evaluations (excluding self-ratings):
| #  | Model                   | Peer Score |  Std |  Raw |
|:--:|-------------------------|-----------:|-----:|-----:|
| 1  | claude-opus-4-5         |       7.82 | 2.06 | 7.83 |
| 2  | gemini-3-pro-preview    |       7.82 | 1.60 | 7.85 |
| 3  | claude-sonnet-4-5       |       7.69 | 1.83 | 7.68 |
| 4  | gpt-5.2                 |       7.60 | 2.12 | 7.65 |
| 5  | deepseek-chat           |       7.32 | 1.93 | 7.31 |
| 6  | grok-4-1-fast           |       6.89 | 1.84 | 6.94 |
| 7  | gemini-3-flash-thinking |       6.86 | 1.76 | 6.91 |
| 8  | gpt-4.1-mini            |       6.74 | 1.96 | 6.76 |
| 9  | kimi-k2-0905            |       6.70 | 2.44 | 6.67 |
| 10 | sonar-pro               |       6.09 | 2.28 | 6.15 |
| 11 | llama-4-maverick        |       5.66 | 2.10 | 5.76 |

## Bias Analysis

Three types of bias detected in the evaluation process:

| Bias Type | Cause | Interpretation |
|-----------|-------|----------------|
| **Self Bias** | Evaluator rates own answers | + overrates self, − underrates self |
| **Name Bias** | Brand/model recognition | + name helped, − name hurt |
| **Position Bias** | Fixed order in answer list | + position helped, − position hurt |

### Position Bias

Effect of answer position in fixed-order (blind) evaluation:

| Pos | Blind Score | Pos Bias |
|:---:|------------:|---------:|
|  1  |        8.23 |    +0.63 |
|  2  |        6.98 |    +0.24 |
|  3  |        7.78 |    -0.04 |
|  4  |        7.51 |    -0.18 |
|  5  |        7.63 |    -0.19 |
|  6  |        6.61 |    -0.25 |
|  7  |        6.69 |    -0.20 |
|  8  |        7.07 |    -0.25 |
|  9  |        5.32 |    -0.34 |
|  10 |        6.00 |    -0.09 |
|  11 |        6.70 |    +0.00 |

*Pos Bias = Blind − Peer (positive = position helped)*

### Model Bias

Self-favoritism and brand recognition effects:

| #  | Model                   | Peer | Self | Self Bias | Shuffle | Name Bias |
|:--:|-------------------------|-----:|-----:|----------:|--------:|----------:|
| 1  | claude-opus-4-5         | 7.82 | 7.87 |     +0.05 |    8.25 |     +0.43 |
| 2  | gemini-3-pro-preview    | 7.82 | 8.20 |     +0.38 |    8.00 |     +0.19 |
| 3  | claude-sonnet-4-5       | 7.69 | 7.52 |     -0.17 |    7.90 |     +0.21 |
| 4  | gpt-5.2                 | 7.60 | 8.16 |     +0.56 |    7.87 |     +0.27 |
| 5  | deepseek-chat           | 7.32 | 7.23 |     -0.09 |    7.32 |     -0.00 |
| 6  | grok-4-1-fast           | 6.89 | 7.43 |     +0.53 |    7.07 |     +0.17 |
| 7  | gemini-3-flash-thinking | 6.86 | 7.43 |     +0.56 |    7.05 |     +0.19 |
| 8  | gpt-4.1-mini            | 6.74 | 7.02 |     +0.28 |    6.89 |     +0.16 |
| 9  | kimi-k2-0905            | 6.70 | 6.39 |     -0.30 |    6.80 |     +0.10 |
| 10 | sonar-pro               | 6.09 | 6.73 |     +0.64 |    6.64 |     +0.54 |
| 11 | llama-4-maverick        | 5.66 | 6.79 |     +1.13 |    5.81 |     +0.15 |

*Self Bias = Self − Peer (+ overrates self) | Name Bias = Shuffle − Peer (+ name helped)*

## Judge Generosity
| #  | Model                   | Avg Given |  Std | Count |
|:--:|-------------------------|----------:|-----:|------:|
| 1  | llama-4-maverick        |      7.82 | 1.77 |  2589 |
| 2  | gemini-3-flash-thinking |      7.59 | 2.55 |  2586 |
| 3  | grok-4-1-fast           |      7.21 | 2.28 |  2590 |
| 4  | gpt-4.1-mini            |      7.04 | 1.64 |  2590 |
| 5  | sonar-pro               |      7.01 | 2.18 |  2579 |
| 6  | deepseek-chat           |      7.00 | 1.97 |  2590 |
| 7  | gpt-5.2                 |      6.82 | 2.12 |  2588 |
| 8  | claude-sonnet-4-5       |      6.81 | 1.79 |  2590 |
| 9  | claude-opus-4-5         |      6.76 | 1.60 |  2590 |
| 10 | kimi-k2-0905            |      6.71 | 2.20 |  2570 |
| 11 | gemini-3-pro-preview    |      6.42 | 2.54 |  2580 |

## Performance Overview
```
  PEER SCORE                              RESPONSE TIME
                    0    2    4    6    8   10    │    0s   10s   20s   30s
                    ├────┼────┼────┼────┼────┤    │    ├─────┼─────┼─────┤
  #1 claude-opus-4-5    ███████████████████░░░░░░  7.82  │  ▓▓▓▓▓▓                     8.24s
  #2 gemini-3-pro-previ ███████████████████░░░░░░  7.82  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓             16.05s
  #3 claude-sonnet-4-5  ███████████████████░░░░░░  7.69  │  ▓▓▓▓▓▓▓                    9.54s
  #4 gpt-5.2            ███████████████████░░░░░░  7.60  │  ▓▓▓                        4.65s
  #5 deepseek-chat      ██████████████████░░░░░░░  7.32  │  ▓▓▓▓▓                      7.04s
  #6 grok-4-1-fast      █████████████████░░░░░░░░  6.89  │  ▓▓▓▓▓▓▓                    8.56s
  #7 gemini-3-flash-thi █████████████████░░░░░░░░  6.86  │  ▓▓▓                        4.20s
  #8 gpt-4.1-mini       ████████████████░░░░░░░░░  6.74  │  ▓▓                         3.41s
  #9 kimi-k2-0905       ████████████████░░░░░░░░░  6.70  │  ▓▓▓▓▓▓▓▓▓▓                12.32s
  #10 sonar-pro          ███████████████░░░░░░░░░░  6.09  │  ▓▓                         3.40s
  #11 llama-4-maverick   ██████████████░░░░░░░░░░░  5.66  │  ▓▓                         3.05s
                    └─────────────────────────┘    │    └─────────────────────────┘
```

---
## Methodology
- Phase 1: Each model generates questions across categories
- Phase 2: All models answer all questions with web search enabled
- Phase 3: Each model evaluates all responses in 3 modes:
  - Shuffle Only: Randomized order, real model names shown
  - Blind Only: Fixed order, model names hidden (Response A, B, C...)
  - Shuffle + Blind: Randomized order + hidden names (baseline Peer score)
- Phase 4: Aggregate scores and generate this report

**Bias Detection (positive = factor helped):**
- **Self Bias** = Self − Peer: How much the model overrated itself
- **Name Bias** = Shuffle − Peer: How much name recognition helped
- **Position Bias** = Blind − Peer: How much the fixed position helped