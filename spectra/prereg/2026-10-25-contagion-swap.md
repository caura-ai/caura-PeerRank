# Pre-registration: is it the voice that spreads? A style swap for contagion

Registered 2026-10-25.
- **The registration:** the commit that adds this file, together with its code (`swap_rewrite_jobs`,
  `swap_answer_jobs`, `score_swap` and `swap` in `spectra/voice.py`), pushed before any call.
- **No calls yet:** none of the calls below will be made until the user approves the budget.
- **Changes from here on:** no hypothesis, metric, prompt list or message changes once data exist. Any deviation is
  appended at the end with its reason.

## Why

The phase 2 contagion test (E5, `2026-10-25-voice-phase2.md`) found that after reading five earlier answers by
another model as notes, a model's answers moved 20% of the way toward that model's accent (14.6% to 25.5%),
compared with reading its own notes.

**Length is an unlikely explanation.** A post-hoc check, not pre-registered, found the effect is 14.7% with length
left out (7.5% to 21.9%). It is also weakest for the source with the longest notes, gemini-4-argon.

**Content is still open.** Another model's notes differ from a model's own in what they say as well as in how they
say it, so E5 cannot tell whether the voice spreads or the content's form does. This experiment holds content fixed
and swaps the voice.

## Design

Readers, sources, prompts, notes, framing and answering settings are those of E5:
- **Readers:** the six models.
- **Sources:** claude-fable-5.1, gemini-4-argon and gpt-6-astra.
- **Prompts:** the same 40 contagion prompts.
- **Notes:** the same five note prompts.
- **Framing:** the same "Notes from earlier work" message (`contagion_message`).

**Stage 1, rewrites (150 calls).**
- **The instruction** (`rewrite_message`): rewrite the answer "in your own words and your own usual style, keeping
  all of its content and facts".
- **Who rewrites what:** each model rewrites its own five notes, each source rewrites each reader's notes, and each
  reader rewrites each source's notes.
- **No answers until stage 1 is complete:** the code refuses to collect any answer while a rewrite is missing.

**Stage 2, answers (2,040 calls).** For each reader R, source S (S ≠ R) and prompt, R answers after one of four note
sets. Every set is a rewrite, so all four went through the same rewriting step:

| Cell | Content of the notes | Voice of the notes | Notes |
|---|---|---|---|
| OO | the reader's | the reader's | R rewrites its own notes |
| OS | the reader's | the source's | S rewrites R's notes |
| SO | the source's | the reader's | R rewrites S's notes |
| SS | the source's | the source's | S rewrites its own notes |

OO depends only on the reader, so each reader answers it once per prompt.

## Measures

The measures are those of E5:
- **Traits:** the everyday style traits of `spectra.analyze.features`, in SD units of oct-23-26.
- **Projection:** onto the axis from the reader's accent to the source's, as a fraction of the way.
- **Classifier:** the character classifier, trained on the real answers of oct-14-26 and oct-23-26 to every prompt
  except the contagion and note prompts.
- **Intervals:** 95%, from 2,000 resamples of prompts, seed 7.

For each (R, S, prompt), with v the answer's trait vector after each note set:
- **Voice effect** = ½[(v_OS − v_OO) + (v_SS − v_SO)]: the source's voice against the reader's, content held fixed.
- **Content effect** = ½[(v_SO − v_OO) + (v_SS − v_OS)]: the source's content against the reader's, voice held fixed.

## Hypotheses

| | Hypothesis | Confirmed if |
|---|---|---|
| **W1 (primary)** | The voice spreads: answers move toward the source when the notes are in its voice, content held fixed | The voice effect's interval lies above zero |
| W2 | The voice spreads beyond length | The voice effect with length left out of the projection has an interval above zero |
| W3 | The classifier hears it | The classifier's probability for the source, ½[(P_OS − P_OO) + (P_SS − P_SO)], has an interval above zero |

**The manipulation check, a gate.** Among cross rewrites (one model voicing another's content), the classifier must
name the rewriter more often than the content's author. If it does not, the rewrites did not change the voice, and
W1–W3 are reported as inconclusive whatever their values.

**Exploratory, reported without a prediction:**
- the content effect;
- SS − OO, which repeats E5 with rewritten notes;
- the voice effect per source and per trait.

## What the outcomes would mean

- **W1 and W2 hold:** the voice itself is contagious, beyond length and with content fixed. The paper can say
  "style is contagious".
- **W1 fails, but the content effect is clearly positive:** E5's contagion was mostly the content's form, and the
  paper says so.
- **Both are small:** the effect needs repeated exposure or more prompts to show, and the paper keeps E5's wording,
  "reading another model's notes shifts answers toward its accent".

## Cost

From `python -m spectra.voice swap --revision oct-23-26 --dry-run`, using each writer's measured cost per answer in
oct-23-26 plus the added text at its input price:

| Stage | Calls | Estimate |
|---|---|---|
| Rewrites | 150 | $3.03 |
| Answers: OO | 240 | $6.35 |
| Answers: OS | 600 | $15.14 |
| Answers: SO | 600 | $16.34 |
| Answers: SS | 600 | $16.34 |
| **Total** | **2,190** | **$57.19** |

The answer cells are priced with the original notes, which are about as long as the rewrites. Phase 2 ran 10–13%
over its estimates on experiments of this kind, so expect about $60–65.

## Deviations

None so far.
