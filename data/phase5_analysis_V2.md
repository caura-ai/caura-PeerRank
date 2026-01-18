# PeerRank.ai Final Analysis

**V2** | Generated: 2026-01-16 14:27:02

Judge: **gpt-5.2**

---

## Answering API Cost Analysis

Total Phase 2 Cost: **$28.6335**

| Model                   | Total Cost | Total Input Tokens | Total Output Tokens | Avg Cost/Q |
|-------------------------|-----------:|-------------------:|--------------------:|-----------:|
| llama-4-maverick        |    $0.0154 |             15,687 |              41,356 |    $0.0001 |
| deepseek-chat           |    $0.0218 |             14,043 |              42,479 |    $0.0001 |
| kimi-k2-0905            |    $0.0933 |             14,611 |              33,803 |    $0.0003 |
| gpt-4.1-mini            |    $0.3044 |            526,082 |              58,743 |    $0.0011 |
| gemini-3-flash-thinking |    $0.4684 |             13,361 |             153,893 |    $0.0017 |
| sonar-pro               |    $0.7935 |             12,978 |              50,302 |    $0.0029 |
| grok-4-1-fast           |    $0.8123 |          1,095,574 |              51,667 |    $0.0030 |
| gpt-5.2                 |    $3.4076 |          1,526,025 |              52,644 |    $0.0124 |
| gemini-3-pro-preview    |    $4.0742 |             13,361 |             337,293 |    $0.0148 |
| claude-sonnet-4-5       |    $9.2135 |          2,630,609 |              88,114 |    $0.0335 |
| claude-opus-4-5         |    $9.4291 |          1,477,189 |              81,727 |    $0.0343 |

## Performance vs. Cost

Efficiency analysis (score-weighted value, ranked by Points²/¢):
| Rank | Model                   | Score | Reported In | Est. In | Output | Cost/Ans |  Pts²/¢ |
|:----:|-------------------------|------:|------------:|--------:|-------:|---------:|--------:|
|  1   | deepseek-chat           |  7.32 |          51 |   2,069 |    154 |  $0.0001 | 6768.26 |
|  2   | llama-4-maverick        |  5.66 |          57 |   2,069 |    150 |  $0.0001 | 5716.20 |
|  3   | kimi-k2-0905            |  6.70 |          53 |   2,069 |    122 |  $0.0003 | 1322.12 |
|  4   | gpt-4.1-mini            |  6.74 |       1,913 |   2,069 |    213 |  $0.0011 |  409.88 |
|  5   | gemini-3-flash-thinking |  6.86 |          48 |   2,069 |    559 |  $0.0017 |  276.50 |
|  6   | grok-4-1-fast           |  6.89 |       3,983 |   2,069 |    187 |  $0.0030 |  160.94 |
|  7   | sonar-pro               |  6.09 |          47 |   2,069 |    182 |  $0.0029 |  128.69 |
|  8   | gpt-5.2                 |  7.60 |       5,549 |   2,069 |    191 |  $0.0124 |   46.62 |
|  9   | gemini-3-pro-preview    |  7.82 |          48 |   2,069 |  1,226 |  $0.0148 |   41.26 |
|  10  | claude-opus-4-5         |  7.82 |       5,371 |   2,069 |    297 |  $0.0343 |   17.84 |
|  11  | claude-sonnet-4-5       |  7.69 |       9,565 |   2,069 |    320 |  $0.0335 |   17.65 |

---

## 1. Overall Peer Ranking Assessment

**Leaderboard (Shuffle + Blind baseline peer scores):**
1) **claude-opus-4-5: 7.82** (Std 2.06)  
2) **gemini-3-pro-preview: 7.82** (Std 1.60)  
3) **claude-sonnet-4-5: 7.69** (Std 1.83)  
4) **gpt-5.2: 7.60** (Std 2.12)  
5) **deepseek-chat: 7.32** (Std 1.93)  
… bottom: **sonar-pro: 6.09**, **llama-4-maverick: 5.66**

**Key takeaways:**
- The top tier is **tightly clustered**: #1 and #2 are effectively tied (7.82 vs 7.82), and #1–#4 span only **0.22 points** (7.82 → 7.60). This suggests **no runaway winner**—multiple models are peer-competitive.
- **Consistency (Std)**:  
  - Most consistent among top models: **gemini-3-pro-preview (Std 1.60)**, then **claude-sonnet-4-5 (1.83)**.  
  - Highest variability overall: **kimi-k2-0905 (2.44)**, then **sonar-pro (2.28)**—their quality appears less predictable across prompts.
- **Efficiency reality check**: “Best” by score is not “best” by latency. The fastest responders (≈3–4s) are not the highest-ranked on quality.

**Runtime composition:** Evaluation dominates: **130m** total in Phase 3 vs **25m** answering—peer judging is the bottleneck, not generation.

---

## 2. Top Performers (score + efficiency)

### Quality leaders
- **claude-opus-4-5 (7.82)**: Top (tied) peer score, solid evaluation speed (**5.07s avg eval**) and moderate answer time (**8.24s**). Strong overall quality perception.
- **gemini-3-pro-preview (7.82, Std 1.60)**: Tied for #1 and **most consistent** among leaders, but **very slow**: **16.05s avg answer** and **29.81s avg eval** (and 1 eval failure: **274/275**).
- **claude-sonnet-4-5 (7.69)**: High quality with moderate slowness (**9.54s answer**, **5.24s eval**).
- **gpt-5.2 (7.60)**: Best “quality-per-second” among top 4: **4.65s answer**, **4.38s eval**—close to the leaders while being much faster than Claude/Gemini Pro.

### Best “balanced” pick
- **gpt-5.2** stands out as the strongest blend of **top-tier score (7.60)** and **fast response** (4.65s), especially compared with **gemini-3-pro-preview** which matches quality but at ~**3.5×** the answer latency.

---

## 3. Outliers & Anomalies

### A) Score outliers
- **llama-4-maverick**: Fastest answers (**3.05s**) but **lowest peer score (5.66)**—a large quality gap despite speed.
- **sonar-pro**: Also fast (**3.40s**) but low score (**6.09**) and **large name bias** (+0.54; more below).

### B) Self-bias (Self − Peer) outliers
Largest self-overrating:
- **llama-4-maverick: +1.13** (Self 6.79 vs Peer 5.66) — biggest gap in the field.
- **sonar-pro: +0.64** (6.73 vs 6.09)
- **gpt-5.2: +0.56**, **gemini-3-flash-thinking: +0.56**, **grok-4-1-fast: +0.53**

Self-underrating (negative self-bias):
- **kimi-k2-0905: −0.30**
- **claude-sonnet-4-5: −0.17**
- **deepseek-chat: −0.09**

Interpretation: several models rate themselves materially higher than peers do; a few are comparatively self-critical.

### C) Name bias (Shuffle − Peer) anomalies
Biggest “brand helped” effects:
- **sonar-pro: +0.54** (largest)
- **claude-opus-4-5: +0.43**
- **gpt-5.2: +0.27**

Near-zero name bias:
- **deepseek-chat: −0.00** (essentially none)

This implies identifiable model names can meaningfully shift outcomes for certain models—especially **sonar-pro** and **claude-opus**.

### D) Position bias (Blind fixed-order) anomalies
- **Position 1** gets a major boost: **+0.63** (Blind 8.23 vs Peer baseline). That’s the single strongest positional effect and suggests “first answer advantage.”
- Worst position effect: **Position 9: −0.34**.
- Net: fixed-order blind evaluation is **not neutral**; it systematically benefits early placement.

### E) Response-time outliers
- **gemini-3-pro-preview** is the clear latency outlier: **16.05s answers**, **29.81s evals** (plus 1 eval miss).
- **kimi-k2-0905** also slow: **12.32s answers**, **30.38s evals** (and **273/275** eval OK—2 misses).
- Fastest answers: **llama-4-maverick (3.05s)**, **sonar-pro (3.40s)**, **gpt-4.1-mini (3.41s)**.

---

## 4. Interesting Findings

### A) Speed vs peer score: weak/negative relationship at extremes
- Fastest models are mostly mid-to-low scoring:  
  - **llama-4-maverick: 3.05s, score 5.66**  
  - **sonar-pro: 3.40s, score 6.09**  
  - **gpt-4.1-mini: 3.41s, score 6.74**
- Yet **gpt-5.2** breaks the “slow = smart” pattern: **4.65s** with **7.60** score, suggesting strong optimization or better instruction-following under this setup.

### B) Judge generosity is inversely related to being top-ranked (not always, but notable)
“Most generous judges” (Avg Given):
- **llama-4-maverick: 7.82** (most generous) yet ranked last (5.66).
- **gemini-3-flash-thinking: 7.59** (2nd most generous) yet mid-pack (#7, 6.86).

“Harshest judges”:
- **gemini-3-pro-preview: 6.42** (harshest) yet tied for #1 quality (7.82).
- **claude-opus-4-5: 6.76** also relatively harsh while being #1/#2.

This pattern is consistent with weaker models inflating others (or being less discriminating), while stronger models grade more strictly.

### C) Category performance differences: not measurable here
The report shows **balanced question counts** (each model contributed 25; totals 55 per category), but **no per-category scoring outcomes** are included—so you can’t conclude “Model X is best at reasoning/current events” from this artifact alone.

---

## 5. Potential Concerns / Red Flags

1) **Position bias is large** (Pos 1: **+0.63**). If any evaluation mode relied heavily on fixed ordering, results could be skewed. The baseline uses Shuffle+Blind (good), but the presence of such a strong effect suggests care is needed when interpreting non-baseline modes.

2) **Name bias is non-trivial** (up to **+0.54**). This indicates brand recognition influences scoring when names are visible. Any “public leaderboard” using named responses risks reputational bias.

3) **High self-bias in some models**:  
   - **llama-4-maverick (+1.13)** and **sonar-pro (+0.64)** are large. This could reflect miscalibrated judging criteria, overconfidence, or rubric mismatch. It’s not “gaming” proof, but it’s a calibration warning.

4) **Reliability issues in evaluation**:  
   - **kimi-k2-0905: 273/275 eval OK**, **gemini-3-pro-preview: 274/275**. Even small failure rates matter at scale; they can bias aggregate scores if failures correlate with hard questions.

5) **Web search ON across all models**: results may partly measure *tool-use strategy* (querying, citation habits, browsing speed) rather than pure model reasoning. That’s fine if intended, but it changes what “best” means.

---

## 6. Recommendations (by use case)

### A) General use (best overall)
- **claude-opus-4-5** and **gemini-3-pro-preview** (both **7.82**) are top on peer quality.  
  - If you need **consistency**, gemini-3-pro-preview has the lowest Std (**1.60**).  
  - If you need **less latency**, claude-opus-4-5 is far faster than gemini-pro (**8.24s vs 16.05s** answers).

Practical pick: **claude-opus-4-5** for “best overall” given the tie on score but much better speed.

### B) Speed-critical applications
- **gpt-5.2**: best combination of **high score (7.60)** and **fast answers (4.65s)**.  
- If you must go ultra-fast and can accept quality drop: **gpt-4.1-mini (3.41s, 6.74)**.  
- Avoid for speed: **gemini-3-pro-preview (16.05s)**, **kimi-k2-0905 (12.32s)**.

### C) High-stakes / accuracy-critical tasks
Using peer score + consistency + bias signals:
- **gemini-3-pro-preview**: top score (**7.82**) and best consistency (**Std 1.60**), but watch operational latency and the **1/275 eval failure**.
- **claude-opus-4-5**: top score (**7.82**) with acceptable latency and strong standing across modes; note **name bias +0.43** suggests it benefits when identified, but baseline still places it at the top.
- **claude-sonnet-4-5 (7.69)** is a strong alternative with slightly lower score and moderate speed.

For high-stakes, I’d favor **claude-opus-4-5** (operationally smoother) or **gemini-3-pro-preview** (most consistent) depending on latency tolerance.

---

## 7. Media Headlines (5)

1) **“Claude Opus 4.5 and Gemini 3 Pro Tie for #1 at 7.82—But Gemini Takes 16.05s per Answer vs Claude’s 8.24s”**  
2) **“First-Answer Advantage: Position #1 Gains +0.63 Points in Blind Scoring, PeerRank Finds”**  
3) **“Llama-4 Maverick Is Fastest (3.05s) Yet Last (5.66)—And Overrates Itself by +1.13”**  
4) **“Name Recognition Matters: Sonar-Pro Gets the Biggest Brand Boost (+0.54) Despite a 6.09 Peer Score”**  
5) **“GPT-5.2 Breaks the ‘Slow = Smart’ Rule: 7.60 Score at 4.65s, Nearly Matching the 7.82 Leaders”**

---

*Analysis generated by gpt-5.2 in 44.4s*
