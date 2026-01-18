# PeerRank.ai LLM Evaluation Report

Revision: **z** | Generated: 2026-01-18 03:27:37

Models evaluated: 12 | Questions: 24 | Web search: **ON**

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
| Phase                |  Duration |
|----------------------|----------:|
| Phase 1 (Questions)  |     18.0s |
| Phase 2 (Answers)    |  2m 37.1s |
| Phase 3 (Evaluation) | 22m 46.6s |
|   └─ Shuffle Only    |  5m 43.9s |
|   └─ Blind Only      |  9m 47.3s |
|   └─ Shuffle + Blind |  7m 15.4s |
| Phase 4 (Report)     |         — |

## Question Analysis
| Model                   | creative | current | factual | practica | reasonin | Total |
|-------------------------|:--------:|:-------:|:-------:|:--------:|:--------:|------:|
| gpt-5.2                 |    -     |    -    |    -    |    1     |    1     |     2 |
| gpt-5-mini              |    1     |    1    |    -    |    -     |    -     |     2 |
| claude-opus-4-5         |    1     |    -    |    -    |    -     |    1     |     2 |
| claude-sonnet-4-5       |    1     |    -    |    -    |    -     |    1     |     2 |
| gemini-3-pro-preview    |    1     |    -    |    -    |    -     |    1     |     2 |
| gemini-3-flash-thinking |    1     |    -    |    1    |    -     |    -     |     2 |
| grok-4-1-fast           |    -     |    1    |    -    |    -     |    1     |     2 |
| deepseek-chat           |    -     |    1    |    -    |    -     |    1     |     2 |
| llama-4-maverick        |    1     |    -    |    -    |    -     |    1     |     2 |
| sonar-pro               |    -     |    -    |    -    |    1     |    1     |     2 |
| kimi-k2-0905            |    -     |    1    |    -    |    -     |    1     |     2 |
| mistral-large           |    -     |    1    |    -    |    -     |    1     |     2 |
| **Total**               |    6     |    5    |    1    |    2     |    10    |    24 |

## Answers
| Model                   | Avg Time | OK/Total |
|-------------------------|---------:|---------:|
| sonar-pro               |    3.27s |    24/24 |
| gemini-3-flash-thinking |    3.76s |    24/24 |
| llama-4-maverick        |    3.81s |    24/24 |
| mistral-large           |    5.17s |    24/24 |
| gpt-5.2                 |    5.58s |    24/24 |
| deepseek-chat           |    7.35s |    24/24 |
| claude-opus-4-5         |    8.37s |    24/24 |
| claude-sonnet-4-5       |    8.70s |    24/24 |
| grok-4-1-fast           |   10.24s |    24/24 |
| gpt-5-mini              |   14.59s |    24/24 |
| gemini-3-pro-preview    |   14.73s |    24/24 |
| kimi-k2-0905            |   15.00s |    24/24 |

## Evaluations
| Model                   | Avg Time | OK/Total |
|-------------------------|---------:|---------:|
| sonar-pro               |    4.54s |    24/24 |
| mistral-large           |    7.49s |    24/24 |
| gpt-5.2                 |    8.63s |    24/24 |
| claude-opus-4-5         |    9.35s |    24/24 |
| claude-sonnet-4-5       |   10.69s |    24/24 |
| gpt-5-mini              |   13.58s |    24/24 |
| llama-4-maverick        |   13.65s |    24/24 |
| grok-4-1-fast           |   14.97s |    24/24 |
| deepseek-chat           |   17.31s |    24/24 |
| gemini-3-flash-thinking |   20.53s |    24/24 |
| kimi-k2-0905            |   37.62s |    23/24 |
| gemini-3-pro-preview    |   55.18s |    24/24 |

## Final Peer Rankings (Shuffle + Blind mode)

Scores from peer evaluations (excluding self-ratings):
| #  | Model                   | Peer Score |  Std |  Raw |
|:--:|-------------------------|-----------:|-----:|-----:|
| 1  | gpt-5.2                 |       8.86 | 2.08 | 8.82 |
| 2  | gpt-5-mini              |       8.66 | 2.21 | 8.69 |
| 3  | gemini-3-pro-preview    |       8.54 | 2.14 | 8.54 |
| 4  | deepseek-chat           |       8.35 | 2.22 | 8.32 |
| 5  | claude-sonnet-4-5       |       8.31 | 2.44 | 8.32 |
| 6  | claude-opus-4-5         |       8.10 | 2.56 | 8.15 |
| 7  | mistral-large           |       8.05 | 2.59 | 8.04 |
| 8  | gemini-3-flash-thinking |       8.02 | 2.21 | 8.07 |
| 9  | kimi-k2-0905            |       7.93 | 2.52 | 7.91 |
| 10 | grok-4-1-fast           |       7.75 | 2.62 | 7.82 |
| 11 | llama-4-maverick        |       7.27 | 2.65 | 7.33 |
| 12 | sonar-pro               |       6.73 | 2.93 | 6.77 |

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
|  1  |        9.14 |    +0.28 |
|  2  |        8.85 |    +0.18 |
|  3  |        7.92 |    -0.18 |
|  4  |        8.25 |    -0.06 |
|  5  |        8.40 |    -0.14 |
|  6  |        7.76 |    -0.26 |
|  7  |        7.67 |    -0.09 |
|  8  |        7.98 |    -0.38 |
|  9  |        6.97 |    -0.29 |
|  10 |        6.42 |    -0.30 |
|  11 |        7.64 |    -0.30 |
|  12 |        8.11 |    +0.07 |

*Pos Bias = Blind − Peer (positive = position helped)*

### Model Bias

Self-favoritism and brand recognition effects:

| #  | Model                   | Peer | Self | Self Bias | Shuffle | Name Bias |
|:--:|-------------------------|-----:|-----:|----------:|--------:|----------:|
| 1  | gpt-5.2                 | 8.86 | 8.38 |     -0.48 |    9.04 |     +0.18 |
| 2  | gpt-5-mini              | 8.66 | 9.00 |     +0.34 |    8.81 |     +0.15 |
| 3  | gemini-3-pro-preview    | 8.54 | 8.54 |     +0.00 |    8.71 |     +0.17 |
| 4  | deepseek-chat           | 8.35 | 7.96 |     -0.39 |    8.33 |     -0.03 |
| 5  | claude-sonnet-4-5       | 8.31 | 8.38 |     +0.07 |    8.63 |     +0.32 |
| 6  | claude-opus-4-5         | 8.10 | 8.71 |     +0.61 |    8.53 |     +0.44 |
| 7  | mistral-large           | 8.05 | 8.00 |     -0.05 |    8.34 |     +0.29 |
| 8  | gemini-3-flash-thinking | 8.02 | 8.62 |     +0.61 |    8.26 |     +0.24 |
| 9  | kimi-k2-0905            | 7.93 | 7.65 |     -0.28 |    8.00 |     +0.06 |
| 10 | grok-4-1-fast           | 7.75 | 8.52 |     +0.77 |    8.06 |     +0.30 |
| 11 | llama-4-maverick        | 7.27 | 8.14 |     +0.88 |    7.34 |     +0.07 |
| 12 | sonar-pro               | 6.73 | 7.21 |     +0.48 |    7.02 |     +0.29 |

*Self Bias = Self − Peer (+ overrates self) | Name Bias = Shuffle − Peer (+ name helped)*

## Judge Generosity
| #  | Model                   | Avg Given |  Std | Count |
|:--:|-------------------------|----------:|-----:|------:|
| 1  | sonar-pro               |      8.35 | 2.52 |   264 |
| 2  | grok-4-1-fast           |      8.28 | 2.50 |   253 |
| 3  | llama-4-maverick        |      8.23 | 2.22 |   231 |
| 4  | gemini-3-flash-thinking |      8.20 | 3.11 |   264 |
| 5  | claude-sonnet-4-5       |      8.16 | 2.22 |   264 |
| 6  | mistral-large           |      8.10 | 2.40 |   220 |
| 7  | claude-opus-4-5         |      8.08 | 2.05 |   264 |
| 8  | deepseek-chat           |      8.08 | 2.14 |   264 |
| 9  | kimi-k2-0905            |      8.00 | 2.63 |   253 |
| 10 | gpt-5-mini              |      7.80 | 2.30 |   264 |
| 11 | gpt-5.2                 |      7.69 | 2.86 |   264 |
| 12 | gemini-3-pro-preview    |      7.62 | 2.79 |   264 |

## Judge Agreement Matrix

Pearson correlation between judges' scores (1.0 = perfect agreement):

| Judge    | gpt-5.2 | gpt-5-mi | claude-o | claude-s | gemini-3 | gemini-3 | grok-4-1 | deepseek | llama-4- | sonar-pr | kimi-k2- | mistral- |
|----------|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| gpt-5.2  |    —    |   0.65   |   0.52   |   0.64   |   0.68   |   0.89   |   0.55   |   0.49   |   0.66   |   0.61   |   0.82   |   0.77   |
| gpt-5-mi |   0.65  |    —     |   0.82   |   0.72   |   0.60   |   0.45   |   0.64   |   0.73   |   0.38   |   0.69   |   0.63   |   0.45   |
| claude-o |   0.52  |   0.82   |    —     |   0.78   |   0.61   |   0.31   |   0.75   |   0.80   |   0.22   |   0.72   |   0.61   |   0.33   |
| claude-s |   0.64  |   0.72   |   0.78   |    —     |   0.77   |   0.51   |   0.81   |   0.67   |   0.37   |   0.70   |   0.70   |   0.49   |
| gemini-3 |   0.68  |   0.60   |   0.61   |   0.77   |    —     |   0.62   |   0.72   |   0.63   |   0.36   |   0.72   |   0.64   |   0.52   |
| gemini-3 |   0.89  |   0.45   |   0.31   |   0.51   |   0.62   |    —     |   0.44   |   0.36   |   0.67   |   0.50   |   0.70   |   0.82   |
| grok-4-1 |   0.55  |   0.64   |   0.75   |   0.81   |   0.72   |   0.44   |    —     |   0.58   |   0.33   |   0.66   |   0.66   |   0.44   |
| deepseek |   0.49  |   0.73   |   0.80   |   0.67   |   0.63   |   0.36   |   0.58   |    —     |   0.27   |   0.69   |   0.48   |   0.41   |
| llama-4- |   0.66  |   0.38   |   0.22   |   0.37   |   0.36   |   0.67   |   0.33   |   0.27   |    —     |   0.23   |   0.75   |   0.77   |
| sonar-pr |   0.61  |   0.69   |   0.72   |   0.70   |   0.72   |   0.50   |   0.66   |   0.69   |   0.23   |    —     |   0.54   |   0.40   |
| kimi-k2- |   0.82  |   0.63   |   0.61   |   0.70   |   0.64   |   0.70   |   0.66   |   0.48   |   0.75   |   0.54   |    —     |   0.68   |
| mistral- |   0.77  |   0.45   |   0.33   |   0.49   |   0.52   |   0.82   |   0.44   |   0.41   |   0.77   |   0.40   |   0.68   |    —     |

**Most similar judges:**
- gpt-5.2 ↔ gemini-3-flash-thinking: r=0.891 (n=288)
- gpt-5-mini ↔ claude-opus-4-5: r=0.824 (n=288)
- gemini-3-flash-thinking ↔ mistral-large: r=0.823 (n=240)

**Least similar judges:**
- deepseek-chat ↔ llama-4-maverick: r=0.274 (n=252)
- llama-4-maverick ↔ sonar-pro: r=0.229 (n=252)
- claude-opus-4-5 ↔ llama-4-maverick: r=0.224 (n=252)

## Question Autopsy

Analysis of question difficulty and controversy based on evaluation scores.

### 🔥 Hardest Questions (lowest avg score)

|  Avg |   Std | Category     | Creator    | Question                                              |
|-----:|------:|--------------|------------|-------------------------------------------------------|
| 4.22 | ±2.99 | current even | deepseek-c | What were the main outcomes of the most recent G7 ... |
| 5.24 | ±3.13 | current even | mistral-la | Summarize the key outcomes of the most recent G20 ... |
| 5.42 | ±3.83 | current even | grok-4-1-f | What were the results of the 2024 US presidential ... |
| 5.87 | ±3.10 | current even | gpt-5-mini | Summarize the major outcomes and commitments from ... |
| 6.89 | ±2.60 | reasoning/lo | gpt-5.2    | You have 12 identical-looking coins, and one is co... |

### ⚔️ Most Controversial (judges disagree)

|   Std |  Avg | Category     | Creator    | Question                                              |
|------:|-----:|--------------|------------|-------------------------------------------------------|
| ±3.83 | 5.42 | current even | grok-4-1-f | What were the results of the 2024 US presidential ... |
| ±3.13 | 5.24 | current even | mistral-la | Summarize the key outcomes of the most recent G20 ... |
| ±3.10 | 5.87 | current even | gpt-5-mini | Summarize the major outcomes and commitments from ... |
| ±2.99 | 4.22 | current even | deepseek-c | What were the main outcomes of the most recent G7 ... |
| ±2.60 | 6.89 | reasoning/lo | gpt-5.2    | You have 12 identical-looking coins, and one is co... |

### ✅ Easiest Questions (highest avg score)

|  Avg |   Std | Category     | Creator    | Question                                              |
|-----:|------:|--------------|------------|-------------------------------------------------------|
| 9.75 | ±0.63 | reasoning/lo | kimi-k2-09 | If a train leaves Station A at 60 km/h and another... |
| 9.75 | ±0.53 | reasoning/lo | grok-4-1-f | If it takes 5 machines 5 minutes to make 5 widgets... |
| 9.58 | ±1.14 | reasoning/lo | claude-son | If a train leaves Station A at 2:00 PM traveling a... |
| 9.47 | ±1.31 | reasoning/lo | llama-4-ma | A snail is at the bottom of a 20-foot well. Each d... |
| 9.34 | ±1.35 | reasoning/lo | sonar-pro  | Alice has 4 brothers and 3 sisters. How many siste... |

### 🤝 Consensus Questions (judges agree)

|   Std |  Avg | Category     | Creator    | Question                                              |
|------:|-----:|--------------|------------|-------------------------------------------------------|
| ±0.53 | 9.75 | reasoning/lo | grok-4-1-f | If it takes 5 machines 5 minutes to make 5 widgets... |
| ±0.63 | 9.75 | reasoning/lo | kimi-k2-09 | If a train leaves Station A at 60 km/h and another... |
| ±0.81 | 9.34 | reasoning/lo | deepseek-c | If all Bloops are Razzies, and some Razzies are Ta... |
| ±1.00 | 9.22 | current even | kimi-k2-09 | Who won the 2024 Nobel Prize in Chemistry and what... |
| ±1.14 | 9.58 | reasoning/lo | claude-son | If a train leaves Station A at 2:00 PM traveling a... |

## Performance Overview
```
  PEER SCORE                              RESPONSE TIME
                    0    2    4    6    8   10    │    0s   10s   20s   30s
                    ├────┼────┼────┼────┼────┤    │    ├─────┼─────┼─────┤
  #1 gpt-5.2            ██████████████████████░░░  8.86  │  ▓▓▓▓                       5.58s
  #2 gpt-5-mini         █████████████████████░░░░  8.66  │  ▓▓▓▓▓▓▓▓▓▓▓▓              14.59s
  #3 gemini-3-pro-previ █████████████████████░░░░  8.54  │  ▓▓▓▓▓▓▓▓▓▓▓▓              14.73s
  #4 deepseek-chat      ████████████████████░░░░░  8.35  │  ▓▓▓▓▓▓                     7.35s
  #5 claude-sonnet-4-5  ████████████████████░░░░░  8.31  │  ▓▓▓▓▓▓▓                    8.70s
  #6 claude-opus-4-5    ████████████████████░░░░░  8.10  │  ▓▓▓▓▓▓                     8.37s
  #7 mistral-large      ████████████████████░░░░░  8.05  │  ▓▓▓▓                       5.17s
  #8 gemini-3-flash-thi ████████████████████░░░░░  8.02  │  ▓▓▓                        3.76s
  #9 kimi-k2-0905       ███████████████████░░░░░░  7.93  │  ▓▓▓▓▓▓▓▓▓▓▓▓              15.00s
  #10 grok-4-1-fast      ███████████████████░░░░░░  7.75  │  ▓▓▓▓▓▓▓▓                  10.24s
  #11 llama-4-maverick   ██████████████████░░░░░░░  7.27  │  ▓▓▓                        3.81s
  #12 sonar-pro          ████████████████░░░░░░░░░  6.73  │  ▓▓                         3.27s
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