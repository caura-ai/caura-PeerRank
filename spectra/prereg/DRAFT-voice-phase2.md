# DRAFT pre-registration: describe, recognize, imitate, and accent contagion

**Status: draft.** It is not yet registered. It becomes binding once three things have happened, in this order:
1. The code it names exists and is tested.
2. Both are committed and pushed together, under a dated file name.
3. All of that comes before any phase 2 call.

Nothing below has been run, and no phase 2 money is spent until the user approves the budget.

## Thesis

Frontier models write with a small, stable accent. Per answer, the model explains a median 3% of each everyday trait
and the prompt 59%, yet the model averages agree between runs at 0.89–1.00 on most traits (analysis.json →
`accent`). The models cannot describe these accents: the 2026-10-25 pre-registration found rank agreement of 0.21
and the AI stereotype overestimated.

Phase 2 asks two questions:
- **Do models know accents in ways they cannot put into words?** Can they recognize and imitate them?
- **Do accents spread** from one model to another through shared context?

The literature review (`spectra/article/literature-review.md`) gives the evidence that each experiment is new.

## Shared rules

- **Prompts:** v1 only, except experiment E6, which is the planned v2 test. Prompts are chosen by fixed hash so
  every model gets the same ones.
  - **Everyday prompts:** no trap tag and no stated limit.
  - **Stratified:** taken evenly from the 10 categories.
- **Answers:** calls use the panel's answering settings (`run_config.json` of oct-23-26), with no system prompt.
- **Measurement:** everyday traits come from `spectra.analyze.features`. Distances are in units of each trait's
  pooled within-model SD in oct-23-26.
- **Classifier:** the character classifier (`spectra.attribute`, character 2–5-grams, settings fixed) is trained on
  the real answers of oct-14-26 and oct-23-26 to the prompts not used in that experiment.
- **Intervals:** 95%, from 2,000 resamples of prompts, seed 7.
  - A directional hypothesis is confirmed when its interval lies entirely on the predicted side of zero.
  - Each experiment has one primary hypothesis. The rest are secondary and reported whatever the outcome; no
    correction across experiments.
- **Accent and stereotype trait sets,** fixed now from oct-23-26 among the style traits, which need no coding:
  - **ACCENT:** model share of 5% or more: em dashes (18.5%), length (9.5%), vocabulary range (5.6%), hedging (5.2%).
  - **STEREOTYPE:** model share of 2% or less: caveats (0.4%), warm openers (0.9%), stock phrases (1.4%),
    self-reference (1.5%), exclamations (1.7%).
  - Traits in between, such as plain wording (4.9%) and follow-up hooks (3.3%), are in neither set.
- **Costs:** each estimate uses the measured cost per answer in oct-23-26 (astra $0.013, fable $0.031, opus $0.014,
  argon $0.017, flash $0.005, muse $0.023; mean $0.017), argon's measured cost per guessing call ($0.079), and $1.04
  per 36 descriptions. Input text added to a call is priced at each model's input rate.

## E1. Describe: more samples (explicit knowledge)

- **Design:** `spectra.describe --sample 2`, `3` and `4`, with the same request and shuffled orders.
- **Purpose:** steadier per-trait descriptions for E3's trait-level test. The pooled side for each (describer,
  target, trait) is the majority over samples 1–4.
- **Hypotheses:** none new. H1–H5 of the 2026-10-25 pre-registration are reported for each sample.
- **Cost:** about $3.

## E2. Recognize from written portraits (does a portrait carry the voice?)

- **Design:** gemini-4-argon plays the guessing game, each model named once, all 200 prompts. Its brief is replaced
  by written portraits taken from the descriptions of sample 1:
  - **SELF:** each model's portrait of itself.
  - **PEER:** each model's portrait as written by the next model in roster order, skipping argon.
- **Leakage:** none. The portraits never saw an answer.
- **Comparisons:** argon's existing games, with the measured plain profile (held-out, 91.0%) and blind (44.9%).
- **R1 (primary):** SELF portraits recognize less than the measured profile. The paired difference (SELF minus
  measured, on shared prompts) has an interval below zero.
- **R2:** SELF portraits beat blind; the interval lies above zero.
- **R3 (exploratory):** SELF against PEER.
- **Code needed:** a profile source in `guess.py` built from `descriptions.jsonl`, scored as its own game key.
- **Cost:** 2 games × 200 calls × about $0.08, about $32.

## E3. Imitate: the impression contest (implicit knowledge, production)

- **Prompts:** 30 everyday v1 prompts, 3 per category.
- **NAME condition:** each model writes the answer "the way <target> would answer it", for each of the five other
  models.
- **PROFILE condition:** the same, with the target's measured portrait from the plain profile appended.
- **SELF condition:** "the way you yourself would answer it", as a control for the instruction itself.
- **Movement metric:** for each imitation, how far its trait vector moves from the imitator's own oct-23-26 answer
  to the same prompt toward the target's.
  - Measured as the projection on the imitator-to-target axis, over the everyday style traits.
  - 0 means it did not move; 1 means it reached the target.
- **I1 (primary):** NAME imitations move toward the target. The mean fraction has an interval above zero.
- **I2:** PROFILE imitations move further than NAME. The paired difference has an interval above zero.
- **I3 (the link between describing and imitating):**
  - **Cells:** every (imitator, target, trait) where the measured gap is at least 0.5 SD, and the imitator's pooled
    description (E1) put the target on the wrong side of itself.
  - **Measure:** the share of those cells where NAME imitations move the trait toward the target.
  - **Hypothesis:** the share is above 0.5, with the interval resampling prompts. That would mean imitation uses
    knowledge the description lacks.
- **I4 (exploratory):** how often the classifier attributes imitations to the target rather than to the imitator.
- **Code needed:** the experiment runner and scorer in a new `spectra/voice.py`.
- **Cost:** 6 × 5 × 30 × 2 calls plus 6 × 30, about 1,980 calls at $0.017, plus portrait input: about $36.

## E4. Hide your own voice (steerability)

- **Prompts:** 50 everyday v1 prompts. Each model rewrites its own oct-23-26 answer to each.
- **UNCOACHED:** "Rewrite this answer so that a reader could not tell which AI model wrote it. Keep the content."
- **COACHED:** the same, plus the model's accent card and its three largest measured departures, as habits to drop.
- **Scores:** the classifier's accuracy at naming the true author, and the change in each trait in SD units.
- **H1 (primary):** UNCOACHED rewrites still give their author away. Classifier accuracy has an interval above
  chance (1/6).
- **H2:** UNCOACHED rewrites change STEREOTYPE traits more than ACCENT traits. Paired by model and prompt, mean |Δ|
  of STEREOTYPE minus ACCENT has an interval above zero.
- **H3:** COACHED rewrites hide better. The paired drop in classifier accuracy has an interval above zero.
- **Cost:** 600 calls at $0.017, plus the answers as input: about $15.

## E5. Accent contagion (shared context)

- **Sources:** claude-fable-5.1 (em dashes), gemini-4-argon (long), gpt-6-astra (short, answers first).
- **Notes:** five of the source's oct-23-26 answers to other v1 prompts, given as notes from earlier work before the
  new prompt. The framing does not ask for imitation.
- **Conditions for each target:**
  - SELF notes: its own five answers.
  - Each source other than itself.
  - NONE: no notes.
- **Prompts:** 40 everyday v1 prompts, none of them used as notes.
- **Shift metric:** the projection of the change from SELF notes to SOURCE notes onto the target-to-source axis.
  SELF notes control for the framing and the amount of text, so only the author changes.
- **C1 (primary):** answers move toward the source's accent. The mean projection, pooled over target and source
  pairs, has an interval above zero.
- **C2:** the classifier's probability for the source rises from SELF notes to SOURCE notes; the paired interval
  lies above zero.
- **C3 (exploratory):**
  - which traits move, ACCENT against STEREOTYPE;
  - whether SELF notes amplify a model's accent against NONE (amplification);
  - whether larger models converge less (Blevins et al.).
- **Cost:** 21 conditions with notes plus 6 with none, × 40 prompts: 1,080 calls at about $0.022 including the notes
  as input. About $24.

## E6. The accent on the locked v2 set (replication)

- **Design:** collect v2 answers from the same six models, code them with the same two coders, and run
  `spectra.analyze --previous oct-23-26`. This is the planned v2 test; the plain profile carried by Part B of the
  2026-10-25 pre-registration is used in any v2 guessing game.
- **V1 (primary):** on the ACCENT traits, the v2 model order agrees with oct-23-26 (Spearman of at least 0.8) on at
  least 4 of 5 traits.
- **V2:** the median model share on v2 lies between 1% and 10%, a small accent again.
- **V3 (exploratory):**
  - whether the closest pair in the family tree (fable and muse in oct-23-26) stays closest;
  - the accent cards.
- **Cost:** 1,200 answers (about $21) plus coding (about $0.012 per answer, $15): about $36.

## Optional E7. Two more answers per v1 prompt

- **Design:** 2 × 1,200 answers, style traits only.
- **Purpose:** separates a model's consistent reaction to a prompt from randomness, and gives each model's
  self-consistency. Descriptive only, no hypothesis.
- **Cost:** about $41.

## Budget

| Experiment | Cost |
|---|---|
| E1 Describe, three more samples | $3 |
| E2 Recognize from portraits | $32 |
| E3 Impression contest | $36 |
| E4 Hide your own voice | $15 |
| E5 Accent contagion | $24 |
| E6 v2 replication | $36 |
| **Subtotal** | **$146** |
| Optional E7, two more samples | $41 |
| **Total with E7** | **$187** |

Estimates assume no retries. The 2026-10-25 run overshot its estimate by about 6%.

## Before this becomes binding

1. Write `spectra/voice.py` (E3–E5 runners and scorers), the portrait source for `guess.py` (E2), and a classifier
   entry point in `spectra.attribute` that trains on real answers and scores new texts. Test each.
2. Run every experiment as a dry run, to fix prompt lists and exact message texts and confirm the costs.
3. Rename this file to a dated pre-registration, record its commit, push it, and only then make the first call.
