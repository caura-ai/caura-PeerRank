# Results for the eight-model generalization test (grok-4.7 and kimi-k3)

The pre-registration is `2026-10-25-panel-8.md`, registered and pushed before any call of this test. This file only
reports the results.

**Scoring:**
- G1–G4 read `analysis.json` of both revisions and `descriptions-s1.json`.
- G5 reads `voice.json` (`E5_new_readers`).
- `python -m spectra.generalize` collects all five and the exploratory items into
  `data/spectra/oct-23-26-p8/panel8.json`.

Intervals are 95% and resample prompts.

**Data:**
- **Answers:** 1,600 per prompt set, the new models' 400 each included, with no failed call.
- **Labels:** deepseek-v4-pro labeled every answer. mistral-large labeled 1,582 of 1,600 on v1 and 1,576 on v2 (deviation 1).

**Cost recorded: $18.28.**
- New models' answers: $4.98 on v1 and $4.72 on v2.
- Describing: $1.78.
- Contagion with the new readers: $6.79.

The coders' calls are not costed by the pipeline. With the pre-registration's estimate for them ($9.79), the total is
about $28, against $25.48 estimated (upper bound $42.34).

## Pre-registered tests

| | Test | Result | Verdict |
|---|---|---|---|
| **G1 (primary)** | The accent order holds on new prompts with eight models | Spearman between v1 and v2: em dashes 0.98, length 1.00, vocabulary range 0.95, hedging 0.90 (4 of 4 at 0.8 or more) | **Confirmed** |
| G2 | The accent is small per answer | Median model share 4.8% on v1 (4.3% on v2), prompt 58% | Confirmed |
| G3 | Each new model has a stable accent of its own | grok-4.7 keeps all 4 card habits on v2; kimi-k3 keeps 3 of 4 (below) | Confirmed |
| G4 | Describing generalizes | H1: rank agreement 0.18, p = 0.001; H5: caveats overestimated 19.5-fold, warm openers 11-fold, trap-catching underestimated (0.72–0.82 of the measured rate) | Confirmed |
| G5 | Contagion generalizes to new readers | grok-4.7 and kimi-k3 as readers move **48%** of the way toward the source (28.6% to 67.0%); the classifier's probability for the source rises 4.4 points (3.3 to 5.6) | Confirmed |

**The new accent cards** (habits held on the same side in at least 80% of categories):
- **grok-4.7,** the same on both prompt sets: keeps answers short, leads with the answer, does just what's asked,
  cooler in tone.
- **kimi-k3:**
  - kept on both sets: heavy em-dash habit, wide vocabulary range, ends by offering more;
  - v1 only: opens with warmth or praise;
  - v2 only: warm in tone.

**The other description tests, 8 × 8:**
- **H2:** still no self-advantage. A model's own placement differs from the others' by +0.09 places (Wilcoxon
  p = 0.91).
- **H3:** behavior is known better than style (0.39 against 0.10, p = 0.03).
- **H4:** the widest sibling gap, em dashes between claude-fable-5.1 and claude-opus-5.5, is 2.07 SD measured. The
  describers place it at −0.19, the wrong way round.

## Exploratory

**Contagion by reader** (post hoc, deviation 3): with eight models, E5's shift is 28.5% over all readers (21.7% to
35.8%). The fraction of the way inflates when a reader and a source sound alike, because the axis between them is
short. So the shift is also given in SD units along the axis, which does not inflate.

| Reader | Shift toward the three sources, SD | Fraction of the way |
|---|---|---|
| kimi-k3 | 0.74 to 1.13 | 0.36 to 1.25 |
| gemini-4-argon | 0.50 to 0.65 | 0.34 to 0.39 |
| claude-fable-5.1 | 0.34 to 0.50 | 0.33 |
| gemini-3.8-flash | 0.02 to 0.35 | 0.03 to 0.26 |
| muse-wonder-1.0 | 0.07 to 0.32 | 0.05 to 0.61 |
| grok-4.7 | 0.08 to 0.25 | 0.05 to 0.25 |
| claude-opus-5.5 | 0.04 to 0.23 | 0.03 to 0.20 |
| gpt-6-astra | −0.10 to 0.00 | −0.06 to 0.00 |

Readers differ sharply. kimi-k3 takes on the voice of whatever it reads, while gpt-6-astra does not move at all. A
source does not count as its own reader, so each model has two or three sources.

**Family tree:**
- **Nearest neighbors:** grok-4.7's is gpt-6-astra (distance 0.86); kimi-k3's is claude-fable-5.1 (0.78).
- **The two new models are far apart** (1.86, against a median pair distance of 1.41).
- **The closest pair is still claude-fable-5.1 and muse-wonder-1.0,** on both prompt sets (0.60 and 0.65).

**The new coder pair** (deepseek-v4-pro and mistral-large) agrees less than the old pair (deepseek-v4-pro and grok-4.7).

| Label | New pair | Old pair |
|---|---|---|
| False premise, tagged prompts | 1.00 (86 answers) | 1.00 (66) |
| Pasted error, tagged | 0.82 (48) | 0.91 (36) |
| Ambiguity (clarification), tagged | 0.82 (86) | 0.84 (66) |
| Stated opinion, tagged | 0.62 (79) | 0.70 (60) |
| Emotional acknowledgment, tagged | 0.00 (40; 95% raw agreement) | undefined (30; all "yes") |
| Answers first, all answers | 0.35 | 0.62 |
| Unasked advice, all answers | 0.46 | 0.69 |
| Moral commentary, all answers | 0.35 | 0.49 |

Emotional acknowledgment shows the paradox of high agreement with low κ: one coder said "no" twice in 40.

**The six models' coded traits, new pair against old:**
- **Model order:** Spearman 0.81 to 1.00 on every behavior trait with variation.
- **Levels:** mostly similar. The new pair sees more initiative (0.75 against 0.62) and more unasked advice (0.42
  against 0.35).

## What this means

- **The accent findings generalize.** Two new models from two more vendors, one of them a non-US lab, have the same
  kind of accent:
  - small per answer (4.8%);
  - its order kept on new prompts (4 of 4 traits);
  - each model with its own stable habits.

  grok-4.7 sounds like gpt-6-astra: short, direct, cool. kimi-k3 sounds like claude-fable-5.1: em dashes, wide
  vocabulary, warm.
- **The models still mishear accents.** With eight models, descriptions again rank the panel only weakly (0.18) and
  overstate the AI stereotype (caveats 19.5-fold). There is again no self-advantage, and the largest sibling gap is
  seen the wrong way round.
- **Contagion generalizes, and readers differ in how much they take on.** The two new readers pick up the notes'
  accent, kimi-k3 more than any model in the panel. gpt-6-astra is immune.

## Deviations

1. **Missing labels.** mistral-large failed on 18 v1 and 24 v2 answers (1.1% and 1.5%) after two passes, on rate
   limits. Those answers carry deepseek-v4-pro's labels only. G1 rests on style traits alone, which no coder touches.
2. **A wording error in the exploratory list.** It asks whether "the two non-US-lab models" are close. grok-4.7 is
   from xAI, a US lab; the item is reported as the distance between the two new models.
3. **Post hoc additions:**
   - contagion for every reader, in SD units as well as fractions, added because kimi-k3's fraction exceeded 1 on a
     short axis;
   - trap-label κ on tagged prompts only, matching the corrected paper.
