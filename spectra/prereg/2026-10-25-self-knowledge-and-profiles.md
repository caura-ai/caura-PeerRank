# Pre-registration: self-knowledge replication and profile selection

Written 2026-10-25 and committed and pushed before any of the data it covers were collected. The code that
computes every test below is in the same commit: `spectra/describe.py` (`confirm`, `stability`) and
`spectra/guess.py` (`profile_contrasts`). Neither will change once data exist; any deviation is appended at the end
of this file with its reason.

Both parts use the v1 prompt set (`prompts_v1.json`) and the oct-23-26 run. The locked test set (`prompts_v2.json`,
hash in `prompts_v2.lock`) is not touched here.

## Part A. Do models know how they write? A replication

### What came before

Sample 0 (36 descriptions: six describers, each describing all six panel models including itself; oct-23-26;
$1.05) was exploratory. The hypotheses below were formed after seeing it, and its scoring was changed once after
seeing it (from a ratio-based error to the rank- and sign-based measures in `score`). Sample 0 therefore supports
none of them; it only motivates them. Its values are listed for reference.

### Design

```
python -m spectra.describe --revision oct-23-26 --sample 1
python -m spectra.describe --revision oct-23-26 --sample 1 --measured oct-14-26
```

- The same six describers and six targets, the same request word for word, temperature 0.
- The panel and the fourteen estimates are listed in an order shuffled for (sample, describer, target), so each call
  is fresh and no reply can depend on list order.
- Failed or unparseable replies are retried by rerunning the command, at most three times. The tests run on what is
  collected, and the number of usable descriptions is reported.
- Primary scoring is against oct-23-26's `analysis.json`. Scoring against oct-14-26 is secondary and reported next to
  it.

### Hypotheses and tests

`confirm()` with seed 7 and 10,000 resamples:

| | Hypothesis | Test | Confirmed if | Sample 0 |
|---|---|---|---|---|
| H1 | Describers rank the panel better than chance | Mean Spearman (per describer over traits, then over describers) against a null that permutes each trait's measured values across the six models | one-sided p < 0.05 | 0.31, p = 0.0002 |
| H2 | No self-advantage: a model places itself no better than the others place it | Self minus mean of others' places off, per model and trait; 95% interval resampling traits | interval inside ±0.25 places | −0.01 (−0.17 to +0.14) |
| H3 | Behavior is known better than style | Mean rank agreement on premise, errors, ambiguity, agree_user minus length, structure, emdash, hedging, exuberance; exact permutation of the nine traits' group labels (126 splits) | one-sided p < 0.05 | 0.56 vs 0.22, p = 0.008 |
| H4 | The widest same-vendor gap is missed | Describers' mean gap between claude-fable-5.1 and claude-opus-5.5 on em dashes, in each describer's spread, against the measured gap in the panel's spread | seen gap under half the measured gap | measured +2.54, seen −0.52 |
| H5 | Stereotyped habits are overestimated and trap-catching underestimated | Median estimate / measured value per trait | above 1 on hedging, caveats, warm_open and hooks; below 1 on premise, errors and ambiguity | 2.9, 15.0, 12.0, 3.8; 0.8, 0.7, 0.8 |

The ±0.25 bound in H2 is about 60% of the margin by which sample 0's describers beat chance (1.94 places off by
chance, 1.52 observed).

H1 to H3 are inferential tests at α = 0.05 each, with no correction for three tests. H4 and H5 are descriptive
criteria with no p-value. H5's ratios depend on how each trait is defined and coded, so it says which way the
estimates lean, not how calibrated they are.

### Reported whatever the outcome

- All five outcomes against oct-23-26, and against oct-14-26.
- Agreement between sample 1 and sample 0 (mean Spearman of each describer's estimates across the six models), with
  no prediction.
- The sibling-gap summary (same vendor against different vendors), with no prediction.
- The cost.

A hypothesis that fails in sample 1 is reported as not replicated. It will not be retested with another metric on
these data.

## Part B. Choosing the profile for the v2 test

### Design

gemini-4-argon is the only guesser, in the held-out game with each model named once, on oct-23-26's answers:

```
python -m spectra.guess --revision oct-23-26 --mode heldout --profile <variant> --guessers gemini-4-argon
python -m spectra.guess --revision oct-23-26 --mode heldout --compare-profiles --guessers gemini-4-argon
```

- Variants: `portrait`, `pairs`, `examples`, `category`, and `all` (the four together). There are 200 calls each,
  with an estimated $80 for the five.
- The comparison is argon's existing plain-profile game: 91.0% of 990 judgments of other models' answers in 198
  calls. Its two failed calls will not be retried, so the published game stays as it is.

### Metric and selection rule

`profile_contrasts` gives argon's accuracy on other models' answers with a variant minus with the plain profile.
Both are scored on the prompts the two games share, and the 95% interval resamples those prompts (1,000 resamples,
seed 7).

- The variant with the largest gain is carried to the v2 test if its interval lies above zero.
- Otherwise the plain profile is carried.
- If two variants gain equally, the shorter profile wins.

### What a null result means

Argon is the strongest guesser, so the ceiling is about 9 points. With about 1,000 paired judgments, a gain smaller
than about 2 points will not be distinguishable from zero. A null result means a part adds nothing for the strongest
guesser. It does not mean the part adds nothing for weaker guessers.

### What follows

The chosen profile and the plain profile are both played on v2 by the full panel. That test gets its own
pre-registration before any v2 answer is collected.

## Deviations

None so far.
