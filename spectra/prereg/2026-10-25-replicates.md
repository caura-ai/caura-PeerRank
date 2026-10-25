# Pre-registration: three answers per prompt (E7)

**Registered 2026-10-25.**
- **The registration:** the commit that adds this file, together with its code (`spectra/replicates.py`,
  `spectra/config-s3.yaml`), pushed before any E7 call.
- **Approval:** the user approved E7, the optional experiment of `2026-10-25-voice-phase2.md` ($41.13 there), and
  asked for it to run at a lower concurrency.
- **Changes from here on:** no hypothesis, metric or prompt list changes once data exist. Any deviation is appended
  at the end with its reason.

## Why

Every Spectra run so far has one answer per model per prompt. With one answer, two things cannot be told apart, and
both count against the accent:
- **the model's steady reaction** to that particular prompt (model × prompt), which is part of how the model writes;
- **sampling noise,** which is not.

The paper's headline split is 3.3% for the model and 59% for the prompt, and leaves the remaining 38% unexplained.
Three answers per prompt split that remainder.

E7 also tests the explanation given for a confound in the impression contest (E3, deviation 2 of phase 2). There, a
model's fresh SELF answer drifted 25.0% of the way toward the other model (21.4% to 28.6%). The axis ran from the
model's single earlier answer to the other model's, and the explanation was regression to the mean. If that is
right, the drift shrinks when the axis runs between the means of three answers.

## Data

- **The revision:** `oct-23-26-s3`, made by `python -m spectra.extend --from oct-23-26 --to oct-23-26-s3`. It copies
  the oct-23-26 answers as sample 0, unchanged; no coder is kept, because E7 uses no coded traits.
- **New answers:** samples 1 and 2, collected by
  `python -m spectra.run --config spectra/config-s3.yaml --revision oct-23-26-s3`:
  - the six panel models, all 200 v1 prompts;
  - the answering settings of oct-23-26, with no system prompt;
  - 6 calls in flight per provider instead of 10, so the run can share providers with the 8-model test.
- **Retries:** failed calls are retried by re-running the same command, up to two times. Any (model, prompt) cell
  still missing a sample is left out of every trait for that prompt, and the count is reported.
- **Cost:** 2,400 answers at oct-23-26's measured cost per answer, **$41.13**, plus retries. There is no coding.

**A known gap in time.** Sample 0 was collected on 2026-10-23, samples 1 and 2 on 2026-10-25 or later. A provider
change in between would add to the noise between samples, not to the model or the reaction. That is why the
same-day check below scores samples 1 and 2 on their own.

## Measurement

- **Traits:** the 12 style traits that need no coding: everyday traits minus limit adherence
  (`voice.trait_columns`), measured by `spectra.analyze.features`.
- **Prompts:** for each trait, the prompts where every model has all three answers with a finite value. Traps
  included, matching `analyze.accent`.
- **Four-way split** (`replicates.decompose`), per trait: a two-way ANOVA with replication over models × prompts ×
  3 samples. Each part is reported as its share of the total sum of squares over individual answers: model, prompt,
  model × prompt, noise.
  - The one-answer split on the same cells uses sample 0 alone. On oct-23-26 it reproduces `analyze.accent` exactly:
    median model share **3.6%** over these 12 traits.
- **Interaction test:** F = MS(model × prompt) / MS(noise), with (M−1)(P−1) and MP(S−1) degrees of freedom.
- **Self-consistency** (`replicates.consistency`), per model and trait: 1 minus (mean variance across a prompt's three
  answers) / (variance over all of the model's answers). It is averaged over traits per model.
- **Imitation re-check** (`replicates.imitation_recheck`):
  - The rows are the E3 records in `data/spectra/oct-23-26/voice.jsonl`.
  - Units are those of E3: the oct-23-26 baseline's pooled within-model SDs.
  - For each NAME imitation, the axis runs from the imitator's answer to the target's, with two versions:
    - *single*: the first answers, as E3 scored it. On oct-23-26 this reproduces the published 25.0% exactly.
    - *averaged*: the means of three answers.
- **Intervals:** 95%, from 2,000 resamples of prompts, seed 7 (`voice.prompt_ci`).

## Hypotheses

| | Hypothesis | Confirmed when |
|---|---|---|
| **R1 (primary)** | A model's reaction to a prompt is real, not noise | model × prompt is significant (p < 0.05) on at least 7 of the 12 traits |
| R2 | The accent's size replicates with fresh answers | the median model share with three answers lies within 1.5 points of the one-answer median on the same cells |
| R3 | The E3 SELF drift was regression to the mean | the drift shrinks: the interval of (single − averaged) lies above zero |
| R4 | Naming a target still barely moves a model's voice | NAME − SELF on averaged axes has an upper bound below 10% of the way (published, single: 3.2%, −0.5% to 6.8%) |

R1 is primary. R2–R4 are secondary and reported whatever the outcome.

**Same-day check.** R1 is also scored on samples 1 and 2 alone (`later_answers`). Both versions are reported. If R1
holds on all three samples but fails on the same-day pair, the reaction cannot be separated from a change between
days, and the paper says so.

## Exploratory, reported without a test

- The full four-way split per trait, and its medians.
- The model plus its reactions (model + model × prompt): an upper bound on how much of an answer is the model's own.
- The reactions' share of what is neither model nor prompt, against the 29% expected if model × prompt were pure
  noise (its degrees of freedom over the two parts' total).
- Self-consistency per model, ranked.
- The averaged SELF drift against 12.5%: the value pure regression to the mean predicts from the single-axis 25.0% if
  the two models' noise is equal.

## Scoring

`python -m spectra.replicates --config spectra/config-s3.yaml --revision oct-23-26-s3` writes
`data/spectra/oct-23-26-s3/replicates.json`. The results go into `2026-10-25-replicates-results.md`.

## Deviations

Appended after the data come in. Nothing above changes.
