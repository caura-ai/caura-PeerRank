# Literature review: model voice, self-knowledge, imitation and contagion

Compiled 2026-10-25 for the voice-first direction of the Spectra paper. Each entry was checked against its arXiv
abstract page (or, for the one blog post, the post itself) on that date.

- **Checked:** title, authors, date and venue.
- **Not checked:** claims that appear only in a paper's body. These are marked "not verified" and must be read in
  the full text before we cite them.

## 1. The ten pasted papers, verified

| Paper | Verified | Notes on the pasted summary |
|---|---|---|
| Serapio-García, Safdari, Crepy et al., *Personality Traits in Large Language Models*, arXiv 2307.00184, July 2023 | Yes | The abstract matches: 18 models, reliable under specific prompts, personality can be shaped. The arXiv page names no journal, so the claimed *Nature Machine Intelligence* 2025 version is not verified. |
| Lee, Lim, Han et al., *Do LLMs Have Distinct and Consistent Personality? TRAIT*, arXiv 2406.14703, June 2024, NAACL 2025 Findings | Yes | Matches: 8K questions, Big Five and Dark Triad, profiles shaped by alignment tuning. The "GPT-4 most steerable" detail is in the body, not verified. |
| Gupta, Song, Anumanchipalli, *Self-Assessment Tests are Unreliable Measures of LLM Personality*, arXiv 2309.08163, Sept 2023 | Yes | Matches: equivalent wordings and reordered options give different scores (ChatGPT, three Llama 2 models). |
| Sharma, Tong, Korbak et al., *Towards Understanding Sycophancy in Language Models*, arXiv 2310.13548, Oct 2023 | Yes | Matches: five assistants, human preference data rewards agreement. |
| Reinhart, Markey, Laudenbach et al., *Do LLMs write like humans? Variation in grammatical and rhetorical styles*, arXiv 2410.16107, Oct 2024, PNAS 122 (2025) | Yes | Matches: GPT-4o and Llama 3 variants, consistent differences from humans and between models, larger in instruction-tuned models. |
| Li, Liu, Bashkansky et al., *Measuring and Controlling Instruction (In)Stability in Language Model Dialogs*, arXiv 2402.10962, Feb 2024, COLM 2024 | Yes | Matches: drift within a few exchanges (LLaMA2-chat-70B, GPT-3.5); split-softmax mitigation. |
| Chen, Arditi, Sleight et al., *Persona Vectors*, arXiv 2507.21509, July 2025 | Yes | Matches: activation directions for evil, sycophancy and hallucination; monitoring and preventative steering. "Demonstrations use Qwen and Llama" is not in the abstract, not verified. |
| Duan, Tang, Bai et al., *The Power of Personality*, arXiv 2502.20859, Feb 2025 | Yes | The abstract matches in general: Big Five profiles affect reasoning, creativity and collaboration. The specific openness and conscientiousness correlations by model are not verified. |
| Rudnicka and Juzek, *Beyond "AI Language": The case for the idiolectal nature of LLM output*, arXiv 2608.06589, Aug 2026 (chapter for *Corpus Linguistics 2040*) | Yes | Matches: a 2024 cohort of six models and a 2026 cohort of six newer models, same prompts, stylometric PCA and contractions. They find a generational shift and a stable, identifiable profile per model. The per-model contraction figures (for example Haiku 4.5 at 30.6 per 1,000 words) are not verified. |
| Kwon, Shrestha, Han et al., *Can LLMs Truly Embody Human Personality?*, arXiv 2602.07414, Feb 2026, AAAI 2026 | Yes | Matches in general: personality-prompted dispute dialogues diverge from human data across three closed models. The per-model claims (Sonnet 3.7 de-escalates, GPT-4o Mini escalates) are not verified. |

The in-context paper cited at the end of the paste is also real: Afonin, Andriianov, Hovhannisyan et al.,
*Emergent Misalignment via In-Context Learning*, arXiv 2510.11288, Oct 2025. Narrow in-context examples produce
misaligned answers on unrelated questions in four model families.

## 2. What is already known, by Spectra claim

### A. Models have distinct, stable voices: established

- **Reinhart et al. (PNAS 2025):** grammatical and rhetorical signatures, model against model and model against
  human.
- **Milička, Marklová, Cvrček**, *Benchmark of stylistic variation in LLM-generated texts* (arXiv 2509.10179;
  *International Journal of Corpus Linguistics* 2026): Biber dimensions, 16 models, base against instruction-tuned.
- **Rudnicka and Juzek (2026):** idiolects within a cohort, plus a shift from one model generation to the next.
- **Sun, Yin, Xu et al.**, *Idiosyncrasies in Large Language Models* (arXiv 2502.12150, ICML 2025): 97.1% five-way
  attribution with fine-tuned embeddings. The signal sits in word distributions and survives rewriting.
- **Kobak, González-Márquez, Horvát et al.**, *Delving into LLM-assisted writing in biomedical publications through
  excess vocabulary* (arXiv 2406.07016; *Science Advances* 2025): style words as a population-level trace of LLM use.

**What is left for Spectra:** none of these splits a trait's variation into a model share and a prompt share, or
checks the model order across categories and between runs. The accent result, a small share per answer (median 3%)
with averages that hold from run to run (0.89–1.00 on most traits), is new as a measurement. The existence of
voices is not new.

**What not to claim:** that we discovered model-specific voices, or a stability over time that rests on more than
two runs nine days apart.

### B. Self-knowledge: partly known, so the claim needs care

- **Gupta et al. (2023):** personality self-assessments shift with wording and option order. This anticipates our
  sample 0 to sample 1 agreement of 0.62.
- **Binder, Chua, Korbak et al.**, *Looking Inward* (arXiv 2410.13787): after fine-tuning, a model predicts its own
  behavior better than a second model trained on that behavior. This holds on simple tasks only.
- **Betley, Bao, Soto et al.**, *Tell me about yourself* (arXiv 2501.11120): models fine-tuned on a behavior can
  describe it without being told.
- **Panickssery, Bowman, Feng**, *LLM Evaluators Recognize and Favor Their Own Generations* (arXiv 2404.13076):
  self-recognition is above chance and linked to self-preference.
- **Ackerman and Panickssery**, *Investigating the Ability of LLMs to Recognize Their Own Writing* (LessWrong, July
  2024; blog, not peer-reviewed): Llama 3 8B chat recognizes its own text, but cannot give consistent reasons why.
- **Zhou, Zhu, Wen et al.**, *From Implicit to Explicit* (arXiv 2508.14408): self-recognition works between two
  texts but not on one text alone; the authors trace the gap to a mismatch between representation and response.
- **Barkhordar and Thapa**, *Style, Not Self* (arXiv 2609.30048): apparent self-recognition of code is mostly
  surface style.

**What is left for Spectra:** three things together, which none of these does:
- the descriptions are scored against measured writing habits as ground truth;
- every model describes every other model, so describing oneself has a control;
- no fine-tuning, on frontier API models.

The direction of the error is also unreported: models describe the generic AI stereotype (caveats ×15) and know
least the habits that carry their accent.

**The risk:** recognizing a voice without being able to explain it has precedent, for self-recognition (Ackerman and
Panickssery; Zhou et al.). Our describe/recognize/imitate claim has to be about voices in general, other models'
included, and scored against measured habits. It can't rest on self-recognition alone.

### C. Imitation: of humans, yes; of other models, no study found

- **Wang, Tripto, Park et al.**, *Catch Me If You Can? Not Yet* (arXiv 2509.14543, EMNLP 2025 Findings): LLMs imitate
  formal human writing reasonably well and casual writing poorly, from a few samples.
- **Jemama and Kumar**, *How Well Do LLMs Imitate Human Writing Style?* (arXiv 2509.24930, IEEE UEMCON 2025): the
  prompting strategy matters more than model size, and completion prompts imitate best.

**What is left for Spectra:** one model imitating another model's voice, from its name alone and then from a measured
profile, scored against the target's measured habits. We found no study of this, so the impression contest is new.

### D. Contagion and accommodation: accommodation to users is known; model-to-model contagion is not

- **Blevins, Schmalwieser, Roth**, *Do language models accommodate their users?* (arXiv 2508.03276, EACL 2026): over
  16 models, models converge to a conversation's style, more than humans do. Larger and instruction-tuned models
  converge less.
- **Blevins**, *Accommodation Goes Both Ways* (arXiv 2605.29278, May 2026): on WildChat, models over-converge toward
  their users across eight languages.
- **Thejaswi, Kulshreshta, Oettershagen**, *Linguistic Monoculture in LLM-Assisted Language Use* (arXiv 2607.27134,
  July 2026): a theory paper about humans drifting toward a model's style. Models do not influence each other there.
- **Afonin et al. (2025):** narrow in-context examples change behavior on unrelated tasks.
- **Li et al. (COLM 2024):** instructions drift over a dialogue.

**What is left for Spectra:** whether a model's voice moves toward another model's after it reads that model's text
as notes or memory, without being asked to imitate. We found no study of this. Which habits travel (the accent, or
only the stereotype), and whether reading one's own notes amplifies one's accent, are open too. Blevins' results
make some convergence likely. The novelty is model-to-model contagion through stored context, which bears on
multi-agent systems that share memory.

### E. Personality and behavior: context only

Serapio-García et al., TRAIT, Duan et al. and Kwon et al. study personality questionnaires and personality
prompting. Persona vectors need model internals. Sharma et al. on sycophancy is the closest of these to our
"agrees with the user" trap. These are cited for context. Spectra measures habits in ordinary writing, not
personality.

## 3. Claims we can make, strongest novelty first

1. **Describe, recognize, imitate.** Models recognize, and may imitate, voices they cannot describe, including other
   models' voices, all scored against measured habits. This builds on the self-recognition and introspection work
   in B.
2. **Accent contagion.** A model's voice drifts toward another model's after reading its notes, and possibly toward
   its own accent after reading its own. This is new as model-to-model contagion; accommodation to users (D) is
   known.
3. **A small, steady accent.** Each model explains a small share per answer, but its averages hold across categories
   and runs. The measurement is new; the voices themselves are not (A).
4. **Wrong self-portraits.** Models describe themselves and each other with the AI stereotype. This extends Gupta et
   al. with measured ground truth and peer controls (B).

## 4. Still to read in full before writing

- Rudnicka and Juzek: their stability measure and per-model figures.
- Blevins et al.: effect sizes, and which features converge.
- Zhou et al. and Ackerman and Panickssery: the exact implicit and explicit gap.
- Wang et al. and Jemama and Kumar: imitation metrics we could reuse.
- O'Sullivan (2025, *Humanities & Social Sciences Communications*): found in an earlier search, not yet checked
  against the source.
