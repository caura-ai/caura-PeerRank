# PeerRank.ai LLM Evaluation Report

Revision: **V3** | Generated: 2026-01-18 10:19:16

Models evaluated: 12 | Questions: 300 | Web search: **ON**

## Model Order and Web Mode

1. gpt-5.2 (Tool)
2. gpt-5-mini (Tool)
3. claude-opus-4-5 (Tool)
4. claude-sonnet-4-5 (Tool)
5. gemini-3-pro-preview (Tool)
6. gemini-3-flash-thinking (Tool)
7. grok-4-1-fast (Tool)
8. deepseek-chat (Tavily)
9. llama-4-maverick (Tavily)
10. sonar-pro (Native)
11. kimi-k2-0905 (Tavily)
12. mistral-large (Agents)

## Phase Total Runtime
| Phase                |   Duration |
|----------------------|-----------:|
| Phase 1 (Questions)  |    1m 8.5s |
| Phase 2 (Answers)    |  32m 38.0s |
| Phase 3 (Evaluation) | 372m 22.2s |
|   └─ Shuffle Only    | 120m 42.4s |
|   └─ Blind Only      |  99m 26.3s |
|   └─ Shuffle + Blind | 152m 13.5s |
| Phase 4 (Report)     |          — |

## Question Analysis
| Model                   | creative | current | factual | practica | reasonin | Total |
|-------------------------|:--------:|:-------:|:-------:|:--------:|:--------:|------:|
| gpt-5.2                 |    5     |    5    |    5    |    5     |    5     |    25 |
| gpt-5-mini              |    5     |    5    |    5    |    5     |    5     |    25 |
| claude-opus-4-5         |    5     |    5    |    5    |    5     |    5     |    25 |
| claude-sonnet-4-5       |    5     |    5    |    5    |    5     |    5     |    25 |
| gemini-3-pro-preview    |    5     |    5    |    5    |    5     |    5     |    25 |
| gemini-3-flash-thinking |    5     |    5    |    5    |    5     |    5     |    25 |
| grok-4-1-fast           |    5     |    5    |    5    |    5     |    5     |    25 |
| deepseek-chat           |    5     |    5    |    5    |    5     |    5     |    25 |
| llama-4-maverick        |    5     |    5    |    5    |    5     |    5     |    25 |
| sonar-pro               |    5     |    5    |    5    |    5     |    5     |    25 |
| kimi-k2-0905            |    5     |    5    |    5    |    5     |    5     |    25 |
| mistral-large           |    5     |    5    |    5    |    5     |    5     |    25 |
| **Total**               |    60    |    60   |    60   |    60    |    60    |   300 |

## Answers
| Model                   | Avg Time | OK/Total |
|-------------------------|---------:|---------:|
| sonar-pro               |    3.39s |  300/300 |
| gemini-3-flash-thinking |    3.67s |  300/300 |
| llama-4-maverick        |    4.18s |  300/300 |
| gpt-5.2                 |    5.67s |  300/300 |
| deepseek-chat           |    6.77s |  300/300 |
| claude-opus-4-5         |    8.23s |  300/300 |
| claude-sonnet-4-5       |    8.67s |  300/300 |
| grok-4-1-fast           |    8.68s |  300/300 |
| mistral-large           |    8.73s |  300/300 |
| kimi-k2-0905            |   13.04s |  300/300 |
| gpt-5-mini              |   13.70s |  300/300 |
| gemini-3-pro-preview    |   15.77s |  300/300 |

## Evaluations
| Model                   | Avg Time | OK/Total |
|-------------------------|---------:|---------:|
| sonar-pro               |    5.90s |  300/300 |
| llama-4-maverick        |    7.05s |  300/300 |
| mistral-large           |    7.46s |  300/300 |
| gpt-5.2                 |    9.36s |  300/300 |
| claude-opus-4-5         |    9.49s |  300/300 |
| claude-sonnet-4-5       |   10.89s |  300/300 |
| gpt-5-mini              |   12.71s |  300/300 |
| deepseek-chat           |   17.53s |  300/300 |
| grok-4-1-fast           |   18.05s |  300/300 |
| gemini-3-flash-thinking |   20.17s |  300/300 |
| kimi-k2-0905            |   29.06s |  298/300 |
| gemini-3-pro-preview    |   54.43s |  293/300 |

## Final Peer Rankings (Shuffle + Blind mode)

Scores from peer evaluations (excluding self-ratings):
| #  | Model                   | Peer Score |  Std |  Raw |
|:--:|-------------------------|-----------:|-----:|-----:|
| 1  | gpt-5.2                 |       8.69 | 1.85 | 8.70 |
| 2  | gpt-5-mini              |       8.64 | 1.98 | 8.67 |
| 3  | gemini-3-pro-preview    |       8.40 | 1.89 | 8.39 |
| 4  | claude-opus-4-5         |       8.22 | 2.24 | 8.25 |
| 5  | claude-sonnet-4-5       |       8.14 | 2.14 | 8.16 |
| 6  | mistral-large           |       8.14 | 2.09 | 8.16 |
| 7  | deepseek-chat           |       8.12 | 2.10 | 8.10 |
| 8  | gemini-3-flash-thinking |       7.89 | 1.94 | 7.91 |
| 9  | kimi-k2-0905            |       7.77 | 2.30 | 7.76 |
| 10 | grok-4-1-fast           |       7.70 | 2.12 | 7.74 |
| 11 | sonar-pro               |       7.27 | 2.42 | 7.22 |
| 12 | llama-4-maverick        |       6.96 | 2.44 | 7.00 |

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
|  1  |        9.07 |    +0.37 |
|  2  |        8.74 |    +0.10 |
|  3  |        7.98 |    -0.24 |
|  4  |        7.95 |    -0.19 |
|  5  |        8.26 |    -0.14 |
|  6  |        7.76 |    -0.13 |
|  7  |        7.61 |    -0.09 |
|  8  |        8.00 |    -0.12 |
|  9  |        6.71 |    -0.25 |
|  10 |        7.22 |    -0.05 |
|  11 |        7.67 |    -0.10 |
|  12 |        8.13 |    -0.01 |

*Pos Bias = Blind − Peer (positive = position helped)*

### Model Bias

Self-favoritism and brand recognition effects:

| #  | Model                   | Peer | Self | Self Bias | Shuffle | Name Bias |
|:--:|-------------------------|-----:|-----:|----------:|--------:|----------:|
| 1  | gpt-5.2                 | 8.69 | 8.82 |     +0.13 |    8.85 |     +0.16 |
| 2  | gpt-5-mini              | 8.64 | 8.89 |     +0.24 |    8.64 |     -0.00 |
| 3  | gemini-3-pro-preview    | 8.40 | 8.27 |     -0.13 |    8.50 |     +0.10 |
| 4  | claude-opus-4-5         | 8.22 | 8.52 |     +0.30 |    8.52 |     +0.30 |
| 5  | claude-sonnet-4-5       | 8.14 | 8.32 |     +0.18 |    8.34 |     +0.20 |
| 6  | mistral-large           | 8.14 | 8.46 |     +0.32 |    8.29 |     +0.15 |
| 7  | deepseek-chat           | 8.12 | 7.84 |     -0.28 |    8.23 |     +0.10 |
| 8  | gemini-3-flash-thinking | 7.89 | 8.07 |     +0.18 |    7.99 |     +0.10 |
| 9  | kimi-k2-0905            | 7.77 | 7.59 |     -0.18 |    7.92 |     +0.14 |
| 10 | grok-4-1-fast           | 7.70 | 8.26 |     +0.56 |    7.83 |     +0.13 |
| 11 | sonar-pro               | 7.27 | 6.64 |     -0.63 |    7.49 |     +0.23 |
| 12 | llama-4-maverick        | 6.96 | 7.45 |     +0.49 |    7.12 |     +0.15 |

*Self Bias = Self − Peer (+ overrates self) | Name Bias = Shuffle − Peer (+ name helped)*

## Judge Generosity
| #  | Model                   | Avg Given |  Std | Count |
|:--:|-------------------------|----------:|-----:|------:|
| 1  | llama-4-maverick        |      8.43 | 1.50 |  3102 |
| 2  | grok-4-1-fast           |      8.27 | 2.22 |  2827 |
| 3  | gemini-3-flash-thinking |      8.26 | 2.68 |  3134 |
| 4  | mistral-large           |      8.23 | 1.91 |  2640 |
| 5  | sonar-pro               |      8.17 | 2.06 |  3111 |
| 6  | claude-sonnet-4-5       |      8.07 | 2.02 |  3135 |
| 7  | kimi-k2-0905            |      7.90 | 2.27 |  3113 |
| 8  | claude-opus-4-5         |      7.84 | 1.73 |  3135 |
| 9  | gpt-5.2                 |      7.81 | 2.47 |  3135 |
| 10 | gpt-5-mini              |      7.80 | 2.15 |  3135 |
| 11 | deepseek-chat           |      7.73 | 2.13 |  3135 |
| 12 | gemini-3-pro-preview    |      7.49 | 2.63 |  3058 |

## Judge Agreement Matrix

Pearson correlation between judges' scores (1.0 = perfect agreement):

| Judge    | gpt-5.2 | gpt-5-mi | claude-o | claude-s | gemini-3 | gemini-3 | grok-4-1 | deepseek | llama-4- | sonar-pr | kimi-k2- | mistral- |
|----------|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| gpt-5.2  |    —    |   0.73   |   0.60   |   0.73   |   0.68   |   0.71   |   0.65   |   0.57   |   0.43   |   0.58   |   0.67   |   0.57   |
| gpt-5-mi |   0.73  |    —     |   0.70   |   0.65   |   0.56   |   0.51   |   0.62   |   0.60   |   0.40   |   0.66   |   0.62   |   0.52   |
| claude-o |   0.60  |   0.70   |    —     |   0.72   |   0.56   |   0.41   |   0.64   |   0.69   |   0.53   |   0.78   |   0.71   |   0.52   |
| claude-s |   0.73  |   0.65   |   0.72   |    —     |   0.64   |   0.52   |   0.63   |   0.59   |   0.46   |   0.69   |   0.71   |   0.55   |
| gemini-3 |   0.68  |   0.56   |   0.56   |   0.64   |    —     |   0.66   |   0.70   |   0.56   |   0.43   |   0.51   |   0.65   |   0.51   |
| gemini-3 |   0.71  |   0.51   |   0.41   |   0.52   |   0.66   |    —     |   0.62   |   0.49   |   0.41   |   0.35   |   0.55   |   0.59   |
| grok-4-1 |   0.65  |   0.62   |   0.64   |   0.63   |   0.70   |   0.62   |    —     |   0.52   |   0.47   |   0.59   |   0.68   |   0.54   |
| deepseek |   0.57  |   0.60   |   0.69   |   0.59   |   0.56   |   0.49   |   0.52   |    —     |   0.49   |   0.60   |   0.62   |   0.58   |
| llama-4- |   0.43  |   0.40   |   0.53   |   0.46   |   0.43   |   0.41   |   0.47   |   0.49   |    —     |   0.42   |   0.52   |   0.54   |
| sonar-pr |   0.58  |   0.66   |   0.78   |   0.69   |   0.51   |   0.35   |   0.59   |   0.60   |   0.42   |    —     |   0.63   |   0.46   |
| kimi-k2- |   0.67  |   0.62   |   0.71   |   0.71   |   0.65   |   0.55   |   0.68   |   0.62   |   0.52   |   0.63   |    —     |   0.59   |
| mistral- |   0.57  |   0.52   |   0.52   |   0.55   |   0.51   |   0.59   |   0.54   |   0.58   |   0.54   |   0.46   |   0.59   |    —     |

**Most similar judges:**
- claude-opus-4-5 ↔ sonar-pro: r=0.775 (n=3394)
- gpt-5.2 ↔ claude-sonnet-4-5: r=0.730 (n=3420)
- gpt-5.2 ↔ gpt-5-mini: r=0.726 (n=3420)

**Least similar judges:**
- gemini-3-flash-thinking ↔ llama-4-maverick: r=0.408 (n=3383)
- gpt-5-mini ↔ llama-4-maverick: r=0.398 (n=3384)
- gemini-3-flash-thinking ↔ sonar-pro: r=0.353 (n=3393)

## Question Autopsy

Analysis of question difficulty and controversy based on evaluation scores.

### 🔥 Hardest Questions (lowest avg score)

|  Avg |   Std | Category     | Creator    | Question                                              |
|-----:|------:|--------------|------------|-------------------------------------------------------|
| 3.78 | ±3.06 | current even | gpt-5-mini | Summarize the latest significant geopolitical conf... |
| 4.23 | ±2.60 | current even | sonar-pro  | What major AI advancements in healthcare diagnosti... |
| 4.36 | ±2.94 | current even | gemini-3-f | Name a major scientific breakthrough or discovery ... |
| 4.40 | ±2.86 | current even | gpt-5-mini | Which recently approved drugs or medical treatment... |
| 4.51 | ±2.74 | current even | gemini-3-f | What are the most recent significant developments ... |

### ⚔️ Most Controversial (judges disagree)

|   Std |  Avg | Category     | Creator    | Question                                              |
|------:|-----:|--------------|------------|-------------------------------------------------------|
| ±3.99 | 4.67 | current even | kimi-k2-09 | Who won the most recent Nobel Prize in Literature ... |
| ±3.96 | 4.62 | current even | llama-4-ma | Who won the most recent Nobel Prize in Literature?    |
| ±3.71 | 5.84 | current even | claude-opu | Who won the most recent Academy Award for Best Pic... |
| ±3.68 | 5.96 | current even | grok-4-1-f | Who is the president-elect of the United States fo... |
| ±3.66 | 5.44 | current even | gpt-5.2    | What were the biggest winners and announcements fr... |

### ✅ Easiest Questions (highest avg score)

|  Avg |   Std | Category     | Creator    | Question                                              |
|-----:|------:|--------------|------------|-------------------------------------------------------|
| 9.90 | ±0.35 | reasoning/lo | kimi-k2-09 | A clock loses 3 minutes every hour. If set correct... |
| 9.89 | ±0.38 | reasoning/lo | claude-son | If you have a 3-gallon jug and a 5-gallon jug, how... |
| 9.81 | ±0.43 | reasoning/lo | mistral-la | You have a 3-gallon jug and a 5-gallon jug. How ca... |
| 9.78 | ±0.49 | reasoning/lo | gemini-3-f | You have a 3-gallon jug and a 5-gallon jug. How ca... |
| 9.77 | ±0.48 | reasoning/lo | sonar-pro  | A bat and a ball cost $1.10 total. The bat costs $... |

### 🤝 Consensus Questions (judges agree)

|   Std |  Avg | Category     | Creator    | Question                                              |
|------:|-----:|--------------|------------|-------------------------------------------------------|
| ±0.35 | 9.90 | reasoning/lo | kimi-k2-09 | A clock loses 3 minutes every hour. If set correct... |
| ±0.38 | 9.89 | reasoning/lo | claude-son | If you have a 3-gallon jug and a 5-gallon jug, how... |
| ±0.43 | 9.81 | reasoning/lo | mistral-la | You have a 3-gallon jug and a 5-gallon jug. How ca... |
| ±0.46 | 9.76 | reasoning/lo | gpt-5-mini | A train travels from A to B at 60 km/h and returns... |
| ±0.48 | 9.77 | reasoning/lo | sonar-pro  | A bat and a ball cost $1.10 total. The bat costs $... |

## Performance Overview
```
  PEER SCORE                              RESPONSE TIME
                    0    2    4    6    8   10    │    0s   10s   20s   30s
                    ├────┼────┼────┼────┼────┤    │    ├─────┼─────┼─────┤
  #1 gpt-5.2            █████████████████████░░░░  8.69  │  ▓▓▓▓                       5.67s
  #2 gpt-5-mini         █████████████████████░░░░  8.64  │  ▓▓▓▓▓▓▓▓▓▓▓               13.70s
  #3 gemini-3-pro-previ ████████████████████░░░░░  8.40  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓             15.77s
  #4 claude-opus-4-5    ████████████████████░░░░░  8.22  │  ▓▓▓▓▓▓                     8.23s
  #5 claude-sonnet-4-5  ████████████████████░░░░░  8.14  │  ▓▓▓▓▓▓▓                    8.67s
  #6 mistral-large      ████████████████████░░░░░  8.14  │  ▓▓▓▓▓▓▓                    8.73s
  #7 deepseek-chat      ████████████████████░░░░░  8.12  │  ▓▓▓▓▓                      6.77s
  #8 gemini-3-flash-thi ███████████████████░░░░░░  7.89  │  ▓▓▓                        3.67s
  #9 kimi-k2-0905       ███████████████████░░░░░░  7.77  │  ▓▓▓▓▓▓▓▓▓▓                13.04s
  #10 grok-4-1-fast      ███████████████████░░░░░░  7.70  │  ▓▓▓▓▓▓▓                    8.68s
  #11 sonar-pro          ██████████████████░░░░░░░  7.27  │  ▓▓                         3.39s
  #12 llama-4-maverick   █████████████████░░░░░░░░  6.96  │  ▓▓▓                        4.18s
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