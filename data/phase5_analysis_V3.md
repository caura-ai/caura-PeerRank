# PeerRank.ai Final Analysis

**V3** | Generated: 2026-01-18 07:25:31

Judge: **gpt-5.2**

---

## Answering API Cost Analysis

Total Phase 2 Cost: **$32.0564**

| Model                   | Total Cost | Total Input Tokens | Total Output Tokens | Avg Cost/Q |
|-------------------------|-----------:|-------------------:|--------------------:|-----------:|
| llama-4-maverick        |    $0.0174 |             17,246 |              47,053 |    $0.0001 |
| deepseek-chat           |    $0.0247 |             15,462 |              48,402 |    $0.0001 |
| kimi-k2-0905            |    $0.1071 |             16,088 |              38,982 |    $0.0004 |
| gemini-3-flash-thinking |    $0.4984 |             14,683 |             163,675 |    $0.0017 |
| mistral-large           |    $0.5453 |             70,597 |              67,357 |    $0.0018 |
| sonar-pro               |    $0.8960 |             14,298 |              56,874 |    $0.0030 |
| grok-4-1-fast           |    $0.9764 |          1,317,304 |              61,995 |    $0.0033 |
| gpt-5-mini              |    $0.9858 |          2,105,001 |             229,782 |    $0.0033 |
| gpt-5.2                 |    $3.7498 |          1,642,310 |              62,556 |    $0.0125 |
| gemini-3-pro-preview    |    $4.2546 |             14,683 |             352,103 |    $0.0142 |
| claude-sonnet-4-5       |    $8.9263 |          2,498,330 |              95,422 |    $0.0298 |
| claude-opus-4-5         |   $11.0746 |          1,746,947 |              93,596 |    $0.0369 |

## Performance vs. Cost

Efficiency analysis (score-weighted value, ranked by Points²/¢):
| Rank | Model                   | Score | Reported In | Est. In | Output | Cost/Ans |  Pts²/¢ |
|:----:|-------------------------|------:|------------:|--------:|-------:|---------:|--------:|
|  1   | llama-4-maverick        |  6.96 |          57 |   2,071 |    156 |  $0.0001 | 8377.23 |
|  2   | deepseek-chat           |  8.12 |          51 |   2,071 |    161 |  $0.0001 | 8025.85 |
|  3   | kimi-k2-0905            |  7.77 |          53 |   2,071 |    129 |  $0.0004 | 1691.92 |
|  4   | gemini-3-flash-thinking |  7.89 |          48 |   2,071 |    545 |  $0.0017 |  374.99 |
|  5   | mistral-large           |  8.14 |         235 |   2,071 |    224 |  $0.0018 |  364.36 |
|  6   | gpt-5-mini              |  8.64 |       7,016 |   2,071 |    765 |  $0.0033 |  227.39 |
|  7   | grok-4-1-fast           |  7.70 |       4,391 |   2,071 |    206 |  $0.0033 |  182.16 |
|  8   | sonar-pro               |  7.27 |          47 |   2,071 |    189 |  $0.0030 |  176.87 |
|  9   | gpt-5.2                 |  8.69 |       5,474 |   2,071 |    208 |  $0.0125 |   60.46 |
|  10  | gemini-3-pro-preview    |  8.40 |          48 |   2,071 |  1,173 |  $0.0142 |   49.73 |
|  11  | claude-sonnet-4-5       |  8.14 |       8,327 |   2,071 |    318 |  $0.0298 |   22.29 |
|  12  | claude-opus-4-5         |  8.22 |       5,823 |   2,071 |    311 |  $0.0369 |   18.30 |

---

## 1) Overall Peer Ranking Assessment

**Leaderboard (Peer Score, Shuffle+Blind baseline):**
1) **gpt-5.2 — 8.69** (Std 1.85)  
2) **gpt-5-mini — 8.64** (1.98)  
3) **gemini-3-pro-preview — 8.40** (1.89)  
4) **claude-opus-4-5 — 8.22** (2.24)  
5) **claude-sonnet-4-5 — 8.14** (2.14)  
6) **mistral-large — 8.14** (2.09)  
7) **deepseek-chat — 8.12** (2.10)  
8) **gemini-3-flash-thinking — 7.89** (1.94)  
9) **kimi-k2-0905 — 7.77** (2.30)  
10) **grok-4-1-fast — 7.70** (2.12)  
11) **sonar-pro — 7.27** (2.42)  
12) **llama-4-maverick — 6.96** (2.44)

**What this says overall**
- The distribution is **tight at the top**: #1 to #7 are all between **8.69 and 8.12** (spread **0.57**), suggesting many models are “good enough” on this set when web search is enabled.
- The **bottom two** separate more clearly: **sonar-pro (7.27)** and **llama-4-maverick (6.96)** trail the pack.
- **Consistency (Std)**: Most models sit around ~2.0 Std; the most consistent among top scorers is **gpt-5.2 (1.85)**. Highest variability is **llama-4-maverick (2.44)** and **sonar-pro (2.42)**—more “hit or miss.”

**Runtime/throughput context**
- Phase 3 evaluation dominates runtime (**372m** total), so judge behavior and evaluation latency matter operationally.
- All models successfully answered **300/300** questions (no answer failures). Evaluation completion issues appear for **gemini-3-pro-preview (293/300)** and **kimi-k2-0905 (298/300)** in the *evaluation* phase (not answering), which can affect reliability of the judging pipeline.

---

## 2) Top Performers (score + efficiency)

### Best overall quality
- **gpt-5.2**: **#1 score (8.69)** with strong consistency (**Std 1.85**) and solid speed (**5.67s avg answer**). This is the cleanest “quality + stability” winner.

### Best “near-top” quality
- **gpt-5-mini**: **8.64 (#2)**, but notably slower (**13.70s**). It nearly matches gpt-5.2 on score, but not on latency.

### Best of the rest (strong score, moderate latency)
- **claude-opus-4-5**: **8.22** at **8.23s**.
- **claude-sonnet-4-5 / mistral-large / deepseek-chat**: clustered at **8.14 / 8.14 / 8.12**, with **deepseek-chat** relatively fast (**6.77s**) for that tier.

### Best speed among mid-pack
- **gemini-3-flash-thinking**: **3.67s** (very fast) with **7.89** peer score. Good for latency-sensitive use where “pretty good” is sufficient.

---

## 3) Outliers & Anomalies

### A) Self-rating vs peer-rating (Self Bias)
Largest **positive** self-bias (overrating self):
- **grok-4-1-fast: +0.56** (Self 8.26 vs Peer 7.70) — biggest overconfidence signal.
- **llama-4-maverick: +0.49** (7.45 vs 6.96)
- **mistral-large: +0.32** (8.46 vs 8.14)
- **claude-opus-4-5: +0.30** (8.52 vs 8.22)

Largest **negative** self-bias (underrating self):
- **sonar-pro: −0.63** (Self 6.64 vs Peer 7.27) — strongest self-undervaluation.
- **deepseek-chat: −0.28** (7.84 vs 8.12)

**Why it matters:** large self-bias can distort peer-evaluation ecosystems (models that inflate themselves can skew averages if not excluded; here self-ratings are excluded from Peer Score, which is good).

### B) Name bias (brand/model recognition)
Biggest **name helped** effects (Shuffle − Peer):
- **claude-opus-4-5: +0.30**
- **sonar-pro: +0.23**
- **claude-sonnet-4-5: +0.20**
- **gpt-5.2: +0.16**
- **mistral-large / llama-4-maverick: +0.15**

Near-zero:
- **gpt-5-mini: −0.00** (essentially no brand lift in this setup)

**Interpretation:** showing model names tends to raise scores for several brands—most notably **Claude Opus (+0.30)**—meaning blind evaluation is essential if you want “content-only” ranking.

### C) Position bias (fixed-order blind mode)
Strongest positional advantage:
- **Position 1: +0.37** (Blind Score 9.07 vs Peer baseline)
- **Position 2: +0.10**

Strongest disadvantage:
- **Position 9: −0.25**
- **Position 3: −0.24**
- **Position 4: −0.19**

**Interpretation:** even with names hidden, being early in the list materially helps. This is a real red flag for any “blind-only” evaluation that doesn’t randomize order.

### D) Response-time outliers
Fastest answerers:
- **sonar-pro: 3.39s**
- **gemini-3-flash-thinking: 3.67s**
- **llama-4-maverick: 4.18s**

Slowest answerers:
- **gemini-3-pro-preview: 15.77s**
- **gpt-5-mini: 13.70s**
- **kimi-k2-0905: 13.04s**

Evaluation-time outlier:
- **gemini-3-pro-preview: 54.43s avg evaluation** (and only **293/300 OK**). That’s extreme relative to the next slowest (**kimi 29.06s**). This suggests tool/judge pipeline friction, timeouts, or heavier reasoning templates.

---

## 4) Interesting Findings

### Speed vs score is not monotonic
- The **fastest** model (**sonar-pro 3.39s**) is **#11 by score (7.27)**.
- A **fast** model can still be mid/high: **deepseek-chat** is relatively quick (**6.77s**) yet **#7 (8.12)**.
- The winner **gpt-5.2** is neither fastest nor slowest (**5.67s**)—suggesting a “sweet spot” between deliberation and throughput.

### Judge generosity varies a lot (and can shape outcomes)
Most generous judges (Avg Given):
- **llama-4-maverick: 8.43**
- **grok-4-1-fast: 8.27**
- **gemini-3-flash-thinking: 8.26**

Harshest judge:
- **gemini-3-pro-preview: 7.49** (lowest Avg Given)

**Notable inversion:** **llama-4-maverick** is the *lowest-ranked* model by Peer Score (**6.96**) yet the **most generous evaluator (8.43)**. That combination can systematically inflate others while not improving its own standing (since self-ratings excluded).

### Category performance differences: not observable here
The “Question Analysis” table shows **every model contributed exactly 5 questions per category** (creative/current/factual/practical/reasoning), totaling **300**. That means:
- The dataset is **balanced by construction**, but
- The report provides **no per-category scoring breakdown**, so you can’t tell *which* models excel in reasoning vs factual, etc., from this summary alone.

---

## 5) Potential Concerns / Red Flags

1) **Evaluation reliability issues**
- **gemini-3-pro-preview**: only **293/300** evaluations completed and **54.43s** average evaluation time. That’s a stability/latency risk if you rely on it as a judge or in large-scale eval loops.
- **kimi-k2-0905**: **298/300** evaluations OK and **29.06s** evaluation time—also concerning.

2) **Bias sensitivity**
- **Position bias** is substantial (Pos1 **+0.37**, Pos9 **−0.25**). Any results from “Blind Only” (fixed order) are likely skewed unless corrected.
- **Name bias** is non-trivial for some models (e.g., **Claude Opus +0.30**). Publishing non-blind leaderboards can reward branding over content.

3) **Possible evaluation gaming / rubric mismatch**
- Large self-bias (e.g., **grok +0.56**, **llama +0.49**) can indicate either overconfidence or rubric interpretation differences. Not “proof” of gaming, but worth auditing: do these models systematically give themselves higher scores with weaker justifications?

4) **Web search ON confound**
- With web enabled, models that are better at tool use/citation/aggregation may outperform on “current” and “factual,” potentially compressing differences in pure reasoning. The tight clustering at the top (#1–#7 within 0.57) is consistent with tool-assisted convergence.

---

## 6) Recommendations (by use case)

### General use (best overall)
- **gpt-5.2**: best Peer Score (**8.69**), best consistency among top models (**Std 1.85**), and good latency (**5.67s**). Most balanced choice.

### Speed-critical applications (lowest latency)
- **gemini-3-flash-thinking**: **3.67s** with acceptable quality (**7.89**, #8).
- **sonar-pro**: fastest (**3.39s**) but quality is lower (**7.27**, #11). Recommend when latency dominates and you can tolerate more misses.
- **llama-4-maverick**: **4.18s** but lowest quality (**6.96**). Only for cost/latency-first scenarios with heavy downstream validation.

### High-stakes / accuracy-critical tasks
Based purely on peer score + consistency (and ignoring brand):
- **gpt-5.2** (8.69, Std 1.85)
- **gpt-5-mini** (8.64, Std 1.98) if latency is acceptable
- **gemini-3-pro-preview** (8.40, Std 1.89) *but* its evaluation instability (293/300, 54.43s eval time) is a deployment concern; I’d use it as a responder only, not as a judge, unless you fix the reliability issue.
- **claude-opus-4-5** (8.22) as a strong alternative, though slightly higher variability (Std 2.24).

---

## 7) Media Headlines (5)

1) **“GPT-5.2 Tops PeerRank at 8.69, Beating GPT-5-mini by Just 0.05 Points”**  
2) **“Position Bias Exposed: Being Answer #1 Boosts Scores by +0.37, While Slot #9 Drops −0.25”**  
3) **“Gemini 3 Pro’s Paradox: #3 in Quality (8.40) but a 54.43s Judge—And 7 Evaluations Failed”**  
4) **“Overconfidence Alert: Grok-4-1-fast Rates Itself +0.56 Higher Than Peers (8.26 vs 7.70)”**  
5) **“Fastest Isn’t Best: Sonar-pro Answers in 3.39s Yet Ranks #11 at 7.27—While GPT-5.2 Wins at 5.67s”**

If you want, I can also propose **bias-corrected rankings** (adjusting for measured name/position effects) and a **recommended evaluation protocol** to reduce the 0.3–0.4 point systematic distortions shown here.

---

*Analysis generated by gpt-5.2 in 51.2s*
