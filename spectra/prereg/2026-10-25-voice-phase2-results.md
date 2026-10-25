# Results for the 2026-10-25 voice pre-registration (phase 2)

The pre-registration is `2026-10-25-voice-phase2.md`, registered as commit 5514322 and pushed before any phase 2
call. This file reports the results. The pre-registration itself is unchanged, apart from the deviations appended
to it.

**Scoring:** `python -m spectra.voice score --revision oct-23-26` writes `data/spectra/oct-23-26/voice.json`.
Intervals are 95% and resample prompts.

## Summary

| Experiment | Primary test | Outcome |
|---|---|---|
| E1 Describe, samples 2–4 | H1–H5 of the first pre-registration, reported per sample | H1, H4, H5 hold in all three; H3 in two of three; H2 in one of three |
| E2 Recognize from portraits | R1: self-portraits recognize less than the measured profile | Confirmed: 52.8% against 91.3% (−38.5 points, −42.3 to −34.4); still 8.0 points above blind |
| E3 Impression contest | I1: imitations move toward the target | Confirmed as registered (0.28), but this is regression to the mean. Against the self-imitation control, imitating from the name alone gives +0.03 (−0.005 to +0.07), and from a measured profile +0.39 |
| E4 Hide your own voice | H1: uncoached rewrites still give the author away | Confirmed: named 68% (chance 17%; originals 84%) |
| E5 Accent contagion | C1: answers move toward the accent of the notes' author | Confirmed: 0.20 of the way (0.15 to 0.26) |
| E6 v2 replication | V1: the accent order holds on the locked v2 prompts | Confirmed: 4 of 4 ACCENT traits at a Spearman of at least 0.94 |

## E1. Describe: samples 2–4

Every sample has 36 of 36 descriptions and cost $1.03–1.04.

| Sample | H1 rank agreement | H2 self minus others (places) | H3 | H4 | H5 |
|---|---|---|---|---|---|
| 1 (first pre-registration) | 0.21 | +0.19 (−0.04 to +0.40), not shown | yes | yes | yes |
| 2 | 0.28, p = 0.001 | −0.14 (−0.35 to +0.08), not shown | yes | yes | yes |
| 3 | 0.25, p = 0.001 | −0.06 (−0.24 to +0.09), shown | no | yes | yes |
| 4 | 0.27, p = 0.001 | +0.14 (−0.05 to +0.32), not shown | yes | yes | yes |

What holds across all four samples:
- **Better than chance:** describers rank the panel above chance (0.21–0.28).
- **The em-dash gap is missed:** describers miss the em-dash gap between fable and opus.
- **The stereotype:** describers overestimate stereotyped habits and underestimate catching traps.
- **No consistent self-advantage:** the sign of self minus others flips between samples. The equivalence bound of
  ±0.25 places is met in one sample of four.
- **Behavior is known better than style** in three samples of four.

## E3. Impression contest

All 1,980 calls succeeded; cost $39.44.

### As registered

| Test | Result | Registered verdict |
|---|---|---|
| I1 NAME imitations move toward the target | 0.28 of the way (0.25 to 0.32) | Confirmed |
| I2 PROFILE moves further than NAME | +0.36 (0.27 to 0.45) | Confirmed |
| I3 the right direction on traits the imitator described wrongly | 0.59 (0.53 to 0.64); chance 0.5, a perfect impression 0.93 | Confirmed |
| I4 the classifier names the target (exploratory) | NAME 9%, PROFILE 38%; it names the imitator 70% and 47% | |

### The confound, and the results against the control

**Why the registered metric misleads.** Each imitation is measured from the imitator's single earlier answer to the
same prompt. A fresh answer to the same prompt drifts away from that one sample, toward the prompt's typical
answer, and so toward the other models' answers. The pre-registration missed this.

**What the control shows.** The pre-registered SELF condition is the same message with the imitator's own name. It
moves 0.25 of the way (0.21 to 0.29) toward the other models' answers, so most of I1 is this drift, not imitation.

**The fix.** Comparing each imitation with the same model's SELF answer, on the same axis, removes the drift. This
comparison is post hoc and is logged as a deviation:

| Contrast against SELF | Result |
|---|---|
| NAME toward the target | **+0.03 (−0.005 to +0.07)**: no reliable imitation from the name alone |
| PROFILE toward the target | **+0.39 (0.31 to 0.47)**: a measured profile moves the answer well toward the target |
| I3 cells: right direction | NAME 0.59, SELF 0.62: no evidence of knowledge beyond the description |
| Classifier names the target | NAME 9% against a baseline of 4% (SELF answers named as that model); PROFILE 38% |

**What this means.**
- **From a name alone,** models do not reproduce another model's accent, beyond a doubling of the classifier's rare
  attributions to it.
- **The triad hypothesis is not supported.** Imitation from a name does not carry knowledge that description lacks.
- **A measured Spectra profile works as a voice spec:** it moves the answer about 40% of the way to the target.

## E4. Hide your own voice

All 600 calls succeeded; cost $13.86. The reference is the classifier's accuracy on the original answers to these
50 prompts, which it never trained on: 84%.

| Test | Result | Verdict |
|---|---|---|
| H1 uncoached rewrites still named | 68% (62 to 74), chance 17% | Confirmed |
| H2 STEREOTYPE traits change more than ACCENT traits | −0.17 (−0.26 to −0.07) | Not confirmed; the opposite direction |
| H3 coaching hides better | the classifier's accuracy drops 30 points, 68% to 38% (24 to 35) | Confirmed |

**A caveat for H2.** The STEREOTYPE traits are rare yes/no habits, such as a caveat or a warm opener. Most answers
have none to change, so their mean |Δ| is small by construction. The registered contrast cannot separate "models
target the stereotype" from base rates. What it shows is that uncoached rewrites move length, em dashes, hedging and
vocabulary more than the rare habits.

## E5. Accent contagion

All 1,080 calls succeeded, four of them on the third attempt after Google rate-limit refusals; cost $29.43.

| Test | Result | Verdict |
|---|---|---|
| C1 answers move toward the notes' author | 0.20 of the way (0.15 to 0.26) | Confirmed |
| C2 the classifier's probability for the source rises | +4.6 points (4.0 to 5.2) | Confirmed |

C1 compares two fresh answers given the same framing, notes by the source against notes by the model itself, so
the regression problem of E3 does not arise.

### Exploratory (C3)

**Per trait,** in SD toward the source:

| Moves toward the source | Shift | Does not move | Shift |
|---|---|---|---|
| length | +0.19 | caveats | −0.05 |
| exclamations | +0.14 | warm openers | −0.03 |
| em dashes | +0.10 | stock phrases | +0.01 |
| follow-up hooks | +0.09 | | |
| formatting | +0.07 | | |
| vocabulary range | +0.07 | | |

Averaged, the ACCENT traits move +0.10 and the STEREOTYPE traits +0.02.

**Own notes do not amplify an accent.** After reading its own notes, a model moves −0.22 (−0.38 to −0.08) along the
axis from the panel toward its own accent, so slightly away from its accent, not into it.

**Framing control.** Notes of its own against no notes moves 0.06 (−0.00 to 0.13) along the target-to-source axis.

## E6. The accent on the locked v2 set (`v2-oct-25-26`)

- **Data:** 1,200 of 1,200 answers ($19.81) and 2,400 of 2,400 codes. The coders do not record their cost; the
  estimate from token prices was $14.40.
- **V1 (primary), confirmed.** The four ACCENT traits keep the oct-23-26 model order: em dashes 1.00, length 1.00,
  vocabulary range 0.94, hedging 0.94.
- **V2, confirmed.** The median model share on new prompts is 3.4%, against 3.3% on v1. The median prompt share is
  62%.
- **V3 (exploratory).** fable and muse remain the closest pair (distance 0.63), and argon and flash come next
  (0.98), as on v1.
- **Less stable:** on v2 the least stable traits are caveats (0.44 run to run), follow-up hooks (0.58), stock
  phrases (0.66) and moral commentary (0.70), the same habits that were least consistent on v1.

## E2. Recognize from portraits

gemini-4-argon played both games, with portraits from sample 1. Usable calls: 197 of 200 with self-portraits and 195
of 200 with peer portraits. The rest ran out of output tokens (`MAX_TOKENS` at 32,768) on each of four attempts.
Cost: $53.50.

The comparisons are paired on the prompts both games share:

| Test | Result | Verdict |
|---|---|---|
| R1 self-portraits recognize less than the measured profile | 52.8% against 91.3%: −38.5 points (−42.3 to −34.4), 195 prompts | Confirmed |
| R2 self-portraits beat blind | 52.8% against 44.8%: +8.0 points (+3.7 to +12.5) | Confirmed |
| R3 self against peer portraits (exploratory) | 52.8% against 45.7%: +7.1 points (+2.9 to +11.1) | |

**What this means.**
- **Self-portraits carry little of the voice.** Measured profiles carry most of it.
- **Peer portraits help even less.** Recognition with them is close to blind.
- **Models know themselves a little better than others know them.** This is the one place in phase 2 where a
  self-advantage shows, and it is a small one.

## Cost

| Experiment | Spent |
|---|---|
| E1 | $3.09 |
| E3 | $39.44 |
| E4 | $13.86 |
| E5 | $29.43 |
| E6 | $19.81 for answers, plus coding (unrecorded, estimated $14.40) |
| E2 | $53.50 |
| **Total recorded** | **$159.13**, plus coding (estimated $14.40) |

Overruns against the estimates:
- **The impression contest:** about 13% over.
- **Contagion:** about 10% over.
- **The portrait games:** about 63% over ($53.50 against $32.84). Portrait briefs led argon to longer reasoning
  than the plain-profile calls the estimate was based on.

On 2026-10-25, during the runs, the user raised the cap above the approved $144 and asked to continue.
