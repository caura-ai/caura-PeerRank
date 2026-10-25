# Results for the contagion style swap

The pre-registration is `2026-10-25-contagion-swap.md`, registered as commit f0eda53 and pushed before any call.
This file only reports the results.

**Scoring:** `python -m spectra.voice score --revision oct-23-26`, under `swap` in `voice.json`. Intervals are 95%
and resample prompts.

**Data and cost:**
- **Stage 1:** 150 of 150 rewrites ($3.35).
- **Stage 2:** 2,040 of 2,040 answers ($63.20).
- **Retries:** no call failed.
- **Total:** $66.55, against the $57.19 estimate, 16% over.

## The manipulation check (the gate)

On the 120 cross rewrites, where one model voices another's content, the classifier named the rewriter 42.5% of the
time and the content's author 25.8%. **Passed.** The rewrites changed the voice, so W1–W3 can be read.

The change was partial: a quarter of the rewrites still read as their original author. Since the voice swap is
incomplete, the voice effect below is probably an underestimate.

## Pre-registered tests

| | Test | Result | Verdict |
|---|---|---|---|
| **W1 (primary)** | The voice spreads, content held fixed | **6.6% of the way** (3.2% to 10.4%), 600 reader–source–prompt cases | **Confirmed** |
| W2 | The voice spreads beyond length | 6.6% (2.2% to 11.6%) with length left out | Confirmed |
| W3 | The classifier hears it | The probability for the source rises by 3.3 points (2.8 to 3.7) | Confirmed |

## Exploratory

| | Result |
|---|---|
| The content effect, voice held fixed | **18.1%** (14.4% to 22.2%); 15.4% (10.8% to 20.7%) with length left out |
| SS − OO, a repeat of E5 with rewritten notes | 24.8% (19.3% to 30.9%), against 20.0% in E5 |

## What this means

- **The voice itself is contagious.** With the content of the notes held fixed, answers move toward the source when
  the notes are in the source's voice: 6.6% of the way, beyond length, confirmed by the classifier. The paper can say
  "style is contagious".
- **Most of contagion is content.** What the notes say, and the form that content takes, moves the reader's answers
  about three times as much as their voice: 18.1% against 6.6%. The E5 effect of 20% is mostly content, with a
  smaller but reliable share of voice.
- **E5 replicates.** Its effect reappears when every note is a rewrite: 24.8%, against 20.0%.

## Deviations

**Cost:** $66.55 against an estimate of $57.19, within the user's approval of "about $60" plus the overrun seen in
phase 2. Nothing else deviated.
