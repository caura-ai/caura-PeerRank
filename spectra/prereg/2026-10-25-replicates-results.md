# Results for E7: three answers per prompt

The pre-registration is `2026-10-25-replicates.md`, registered as commit e36f802 and pushed before any E7 call.
This file only reports the results.

**Scoring:** `python -m spectra.replicates --config spectra/config-s3.yaml --revision oct-23-26-s3`, written to
`data/spectra/oct-23-26-s3/replicates.json`. Intervals are 95% and resample prompts.

**Data and cost:**
- **Answers:** 2,400 of 2,400 new answers (samples 1 and 2), at 6 calls per provider. One call failed and succeeded
  on the first retry.
- **Coverage:** every (model, prompt) cell has all three answers. Per trait, 160–200 prompts have finite values for
  every model.
- **Total:** $40.71, against the $41.13 estimate.

## Pre-registered tests

| | Test | Result | Verdict |
|---|---|---|---|
| **R1 (primary)** | A model's reaction to a prompt is real, not noise | model × prompt significant on **12 of 12** traits (largest p = 3 × 10⁻⁴⁹); same-day answers alone: 12 of 12 | **Confirmed** |
| R2 | The accent's size replicates with fresh answers | median model share **3.6%** with three answers, against 3.6% from first answers on the same cells | Confirmed |
| R3 | The E3 SELF drift was regression to the mean | drift falls from 25.0% (21.4% to 28.6%) to **16.2%** (11.8% to 20.5%); shrink 8.8% (6.0% to 11.6%) | Confirmed |
| R4 | Naming a target still barely moves a model's voice | NAME − SELF on averaged axes **3.9%** (−0.4% to 8.6%), upper bound below 10% | Confirmed |

## The four-way split

The registered metric is each part's share of the total sum of squares over individual answers. Medians over the 12
style traits:

| Model | Prompt | Model × prompt | Noise |
|---|---|---|---|
| 3.6% | 59.8% | 25.6% | 11.3% |

- **The model plus its reactions:** 28.4%.
- **Reactions as a share of what is neither model nor prompt:** 64.8%, against 29% if model × prompt were pure noise.
  Every trait is above that null value, the lowest being stock phrases (47%) and caveats (47%).

**Post hoc: variance components.** Sum-of-squares shares hand 1/S of the noise to model × prompt (deviation 1). The
standard estimate from expected mean squares gives each part's share of one answer's variance:

| Trait | Model | Prompt | Model × prompt | Noise |
|---|---|---|---|---|
| Em dashes | 22.0% | 7.3% | 36.2% | 34.5% |
| Length | 11.1% | 80.8% | 7.1% | 0.9% |
| Vocabulary range | 8.3% | 62.3% | 15.4% | 14.0% |
| Plain wording | 7.5% | 71.5% | 12.7% | 8.4% |
| Hedging | 5.8% | 51.7% | 23.1% | 19.3% |
| Structure | 4.4% | 73.0% | 15.7% | 6.9% |
| Follow-up hooks | 3.8% | 16.3% | 37.0% | 43.0% |
| Stock phrases | 2.1% | 36.3% | 16.7% | 44.9% |
| Self-reference | 1.7% | 69.3% | 15.0% | 14.0% |
| Exclamations | 1.5% | 57.0% | 28.0% | 13.5% |
| Warm openers | 0.8% | 33.0% | 42.3% | 23.9% |
| Caveats | 0.1% | 7.6% | 25.9% | 66.4% |
| **Median** | **4.1%** | **54.4%** | **19.9%** | **16.7%** |

The median of model plus reactions is 24.9%. Reactions take a median 53% of what is neither model nor prompt.

## Self-consistency

Self-consistency is the share of a model's own variation that lies between prompts rather than between repeat
answers, averaged over traits.

| gpt-6-astra | gemini-4-argon | claude-opus-5.5 | gemini-3.8-flash | muse-wonder-1.0 | claude-fable-5.1 |
|---|---|---|---|---|---|
| 0.87 | 0.83 | 0.83 | 0.81 | 0.81 | 0.79 |

## Exploratory

- **The averaged SELF drift against regression to the mean.** Pure regression to the mean with equal noise predicts
  12.5%. The measured 16.2% (11.8% to 20.5%) contains it, so the result is consistent with pure regression to the
  mean. The point estimate leaves room for a little real drift.
- **No shift between days** (post hoc, deviation 2). Sample 0 was collected on 2026-10-23 and samples 1 and 2 on
  2026-10-25, with the same model ids.
  - Each model's first answer sits within 0.16 within-model SD of the mean of its later two, on every trait.
  - No shift reaches |t| > 4.
  - The same-day control, sample 1 against sample 2, looks the same.

## What this means

- **The accent's steady part is small and real: 4%.** Fresh answers reproduce it exactly (R2).
- **Models write differently mostly through their reactions.** Each model reacts to particular prompts in its own
  repeatable way. Those reactions are about five times the steady accent: 20% of an answer's variance against 4%.
  Together, which model wrote an answer accounts for about a quarter of each style trait, against a sixth for noise.
- **The stereotype traits are reactions, not habits.** Warm openers, exclamations and caveats barely differ between
  models overall (0.1–1.5%), yet each model uses them on its own set of prompts (26–42%). Every model has these
  habits; what differs is where it uses them. This fits the finding that the models overestimate how much their
  stereotype separates them.
- **Some traits are mostly noise.** Caveats (66%), stock phrases (45%) and follow-up hooks (43%) vary from one answer
  to the next more than anything else. Length is almost fully determined (0.9% noise).
- **The E3 confound is largely explained.** A third of the SELF drift disappears against three-answer baselines,
  as much as regression to the mean predicts within the interval. Naming a target still moves the voice by only 3.9%, so the paper's
  imitation conclusion stands.
- **One answer per prompt was enough for the headline split.** The paper's 3.3% is unchanged; E7 adds what was
  hidden in the remaining 38%.

## Deviations

1. **Post-hoc variance components** (`replicates.components`, added after the data).
   - **Why:** the same-day check showed less noise (8.1%) than all three answers (11.3%). That is what sum-of-squares
     shares predict: they put 1/S of the noise into model × prompt, and 11.3% × (1/2)/(2/3) = 8.5%.
   - **What changed:** the registered shares and tests are unchanged. The components are reported alongside them,
     labelled post hoc, because they are the unbiased answer to "how much of one answer is reaction, how much is
     noise".
2. **Post-hoc day-shift check.** Run to rule out a provider change between 10-23 and 10-25 as the cause of
   deviation 1's difference. None was found.
