# PeerRank.ai LLM Evaluation Report

Revision: **V6** | Generated: 2026-01-28 15:59:25

Models evaluated: 12 | Questions: 420 | P2 grounding: **ON (TAVILY)** | P3 grounding: **OFF**

## Model Order

Fixed position order used in blind evaluation mode:

1. gpt-5.2
2. gpt-5-mini
3. claude-opus-4-5
4. claude-sonnet-4-5
5. gemini-3-pro-preview
6. gemini-3-flash-preview
7. grok-4-1-fast
8. deepseek-chat
9. llama-4-maverick
10. sonar-pro
11. kimi-k2.5
12. mistral-large

## Phase Total Runtime
| Phase                |   Duration |
|----------------------|-----------:|
| Phase 1 (Questions)  |   1m 28.2s |
| Phase 2 (Answers)    |  40m 41.2s |
| Phase 3 (Evaluation) | 126m 13.7s |
|   └─ Shuffle Only    |  153m 6.1s |
|   └─ Blind Only      |  194m 9.7s |
|   └─ Shuffle + Blind | 126m 13.1s |
| Phase 4 (Report)     |          — |

## Question Analysis
| Model                  | creative | current | factual | practica | reasonin | Total |
|------------------------|:--------:|:-------:|:-------:|:--------:|:--------:|------:|
| gpt-5.2                |    8     |    5    |    10   |    2     |    10    |    35 |
| gpt-5-mini             |    7     |    7    |    7    |    7     |    7     |    35 |
| claude-opus-4-5        |    7     |    5    |    7    |    9     |    7     |    35 |
| claude-sonnet-4-5      |    7     |    6    |    8    |    7     |    7     |    35 |
| gemini-3-pro-preview   |    7     |    7    |    7    |    7     |    7     |    35 |
| gemini-3-flash-preview |    7     |    7    |    7    |    7     |    7     |    35 |
| grok-4-1-fast          |    7     |    7    |    7    |    7     |    7     |    35 |
| deepseek-chat          |    7     |    7    |    7    |    7     |    7     |    35 |
| llama-4-maverick       |    7     |    7    |    7    |    7     |    7     |    35 |
| sonar-pro              |    7     |    6    |    7    |    8     |    7     |    35 |
| kimi-k2-0905           |    7     |    7    |    7    |    7     |    7     |    35 |
| mistral-large          |    7     |    5    |    10   |    5     |    8     |    35 |
| **Total**              |    85    |    76   |    91   |    80    |    88    |   420 |

## Answers
| Model                  | Avg Time | OK/Total |
|------------------------|---------:|---------:|
| kimi-k2.5              |    0.00s |      0/0 |
| sonar-pro              |    3.38s |  420/420 |
| gemini-3-flash-preview |    4.88s |  420/420 |
| llama-4-maverick       |    4.90s |  420/420 |
| gpt-5.2                |    4.91s |  420/420 |
| mistral-large          |    5.85s |  420/420 |
| deepseek-chat          |    7.12s |  420/420 |
| grok-4-1-fast          |    7.74s |  420/420 |
| claude-opus-4-5        |    7.94s |  420/420 |
| claude-sonnet-4-5      |    8.38s |  420/420 |
| gpt-5-mini             |   13.41s |  420/420 |
| gemini-3-pro-preview   |   16.27s |  420/420 |

## Evaluations
| Model                  | Avg Time | OK/Total |
|------------------------|---------:|---------:|
| kimi-k2.5              |    0.00s |      0/0 |
| sonar-pro              |    5.18s |  420/420 |
| llama-4-maverick       |    7.32s |  419/420 |
| mistral-large          |    7.52s |  420/420 |
| claude-opus-4-5        |    9.51s |  420/420 |
| gpt-5.2                |    9.83s |  420/420 |
| claude-sonnet-4-5      |   11.33s |  420/420 |
| deepseek-chat          |   16.01s |  419/420 |
| grok-4-1-fast          |   16.78s |  420/420 |
| gemini-3-flash-preview |   18.49s |  420/420 |
| gpt-5-mini             |   35.61s |  420/420 |
| gemini-3-pro-preview   |   52.42s |  417/420 |

## Final Peer Rankings (Shuffle + Blind mode)

Scores from peer evaluations (excluding self-ratings):
| #  | Model                  | Peer Score |  Std |  Raw |
|:--:|------------------------|-----------:|-----:|-----:|
| 1  | gpt-5-mini             |       8.72 | 1.79 | 8.74 |
| 2  | gpt-5.2                |       8.71 | 1.77 | 8.71 |
| 3  | gemini-3-pro-preview   |       8.48 | 1.85 | 8.48 |
| 4  | claude-opus-4-5        |       8.30 | 2.16 | 8.32 |
| 5  | claude-sonnet-4-5      |       8.14 | 2.16 | 8.16 |
| 6  | mistral-large          |       8.13 | 2.05 | 8.16 |
| 7  | gemini-3-flash-preview |       7.90 | 1.90 | 7.92 |
| 8  | deepseek-chat          |       7.87 | 2.22 | 7.84 |
| 9  | grok-4-1-fast          |       7.83 | 2.06 | 7.87 |
| 10 | sonar-pro              |       7.24 | 2.45 | 7.24 |
| 11 | llama-4-maverick       |       6.33 | 2.52 | 6.38 |

## Rankings by Category

Model performance breakdown by question category (rank and score):

| Model       |  Overall  |  creative |  current  |  factual  |  practica |  reasonin |
|-------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| gpt-5-mini  |  1 (8.72) |  2 (8.82) |  2 (7.13) |  6 (8.99) |  2 (9.25) |  4 (9.29) |
| gpt-5.2     |  2 (8.71) |  1 (9.00) |  1 (7.31) |  9 (8.55) |  1 (9.41) |  6 (9.13) |
| gem-3-pro   |  3 (8.48) |  3 (8.28) |  4 (6.77) |  3 (9.24) |  3 (8.68) |  3 (9.30) |
| opus-4.5    |  4 (8.30) |  6 (7.89) |  8 (6.01) |  1 (9.38) |  4 (8.62) |  1 (9.44) |
| sonnet-4.5  |  5 (8.14) |  4 (8.23) |  7 (6.07) |  2 (9.25) |  9 (7.65) |  2 (9.35) |
| mistral     |  6 (8.13) |  5 (7.92) |  6 (6.12) |  5 (9.15) |  5 (8.36) |  7 (8.99) |
| gem-3-flash |  7 (7.90) |  8 (7.32) |  3 (6.79) | 10 (8.53) |  6 (8.01) |  9 (8.79) |
| deepseek    |  8 (7.87) |  7 (7.77) | 10 (5.38) |  4 (9.16) |  8 (7.67) |  5 (9.18) |
| grok-4      |  9 (7.83) |  9 (7.11) |  5 (6.47) |  7 (8.69) |  7 (7.99) |  8 (8.83) |
| sonar-pro   | 10 (7.24) | 10 (5.92) |  9 (5.78) |  8 (8.65) | 10 (7.54) | 11 (8.23) |
| llama-4     | 11 (6.33) | 11 (5.42) | 11 (4.55) | 11 (7.66) | 11 (5.39) | 10 (8.51) |
| kimi-k2.5   | 12 (0.00) | 12 (0.00) | 12 (0.00) | 12 (0.00) | 12 (0.00) | 12 (0.00) |


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
|  1  |        9.05 |    +0.35 |
|  2  |        8.77 |    +0.05 |
|  3  |        8.05 |    -0.25 |
|  4  |        7.90 |    -0.25 |
|  5  |        8.28 |    -0.20 |
|  6  |        7.74 |    -0.16 |
|  7  |        7.66 |    -0.17 |
|  8  |        7.72 |    -0.15 |
|  9  |        6.11 |    -0.22 |
|  10 |        7.15 |    -0.09 |
|  11 |        0.00 |    +0.00 |
|  12 |        8.06 |    -0.07 |

*Pos Bias = Blind − Peer (positive = position helped)*

### Model Bias

Self-favoritism and brand recognition effects:

| #  | Model                  | Peer | Self | Self Bias | Shuffle | Name Bias |
|:--:|------------------------|-----:|-----:|----------:|--------:|----------:|
| 1  | gpt-5-mini             | 8.72 | 8.95 |     +0.23 |    8.71 |     -0.01 |
| 2  | gpt-5.2                | 8.71 | 8.74 |     +0.04 |    8.84 |     +0.14 |
| 3  | gemini-3-pro-preview   | 8.48 | 8.51 |     +0.03 |    8.58 |     +0.10 |
| 4  | claude-opus-4-5        | 8.30 | 8.48 |     +0.18 |    8.54 |     +0.24 |
| 5  | claude-sonnet-4-5      | 8.14 | 8.31 |     +0.17 |    8.32 |     +0.18 |
| 6  | mistral-large          | 8.13 | 8.50 |     +0.36 |    8.25 |     +0.11 |
| 7  | gemini-3-flash-preview | 7.90 | 8.14 |     +0.24 |    8.03 |     +0.13 |
| 8  | deepseek-chat          | 7.87 | 7.50 |     -0.38 |    7.92 |     +0.05 |
| 9  | grok-4-1-fast          | 7.83 | 8.36 |     +0.53 |    7.89 |     +0.05 |
| 10 | sonar-pro              | 7.24 | 7.29 |     +0.06 |    7.45 |     +0.21 |
| 11 | llama-4-maverick       | 6.33 | 6.96 |     +0.63 |    6.52 |     +0.19 |

*Self Bias = Self − Peer (+ overrates self) | Name Bias = Shuffle − Peer (+ name helped)*
