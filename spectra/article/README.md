# Spectra article: *Language Models Mishear Their Own Accents*

The Spectra paper for arXiv (cs.CL). Since 2026-10-25 it is built around the accent story:
- every model has a small, stable accent, and larger, repeatable reactions to particular prompts;
- the models mishear it (they describe the stereotype of AI writing);
- measurement makes it usable;
- it is contagious, mostly through content but also through voice.

It draws on four runs:
- `oct-23-26`: the main run, with the guessing games and the pre-registered voice experiments.
- `oct-14-26`: the replication run.
- `v2-oct-25-26`: the locked second prompt set.
- `oct-23-26-s3`: the main run's prompts answered twice more (E7), for the reactions.

The paper is standalone; PeerRank appears only as related work.

```
article/
  main.tex          the paper
  references.bib    bibliography; entries marked TODO still need checking against their source
  make_assets.py    writes everything in generated/ from data/spectra/<run>/
  generated/        numbers.tex (one macro per number), tab_*.tex, fig_*.pdf (+ .png previews)
```

## Rule: no hand-typed results

Every result in `main.tex` is a macro from `generated/numbers.tex` or a generated table or figure.
To change a number, change the analysis or the run, then regenerate. Do not edit `generated/` by hand.
The only literal numbers in the text are method settings (2,000 bootstrap resamples, 50-word windows,
z >= 2.5) and figures quoted from other papers.

## Regenerate the assets

Needs the run data, which is not in git:
- `data/spectra/oct-23-26/`, including `voice.json`, `descriptions-s1..4.json` and `attribution.json`
- `data/spectra/oct-14-26/`
- `data/spectra/v2-oct-25-26/`

It also needs the `spectra` extra (`pip install -e ".[spectra]"`), plus matplotlib and scipy.
Run `python -m spectra.voice score --revision oct-23-26` first, so that `voice.json` is current.

```
python spectra/article/make_assets.py
```

The script asserts the facts the text states in words (for example, that one prompt is the only
false-premise miss in both runs). If a new run breaks one, the script stops instead of
writing a number into a sentence that no longer fits it.

## Build the PDF

No LaTeX engine is installed on the machine this draft was written on, so it has not been compiled yet.
Either upload this folder (with `generated/`) to Overleaf, or with a local TeX installation run:

```
cd spectra/article
latexmk -pdf main.tex
```

## Open items in the draft

Search `main.tex` for `\todo` and `references.bib` for `TODO`:

- author list and order; repository link and license
- coder costs (not recorded by the pipeline)
- bibliography entries cited from memory or with incomplete titles and author lists

The pre-registrations and their results are in `spectra/prereg/`. Earlier work, done since the first draft: the classifier is also scored jointly per prompt and on the game's
halves (`analyze.distinctiveness`), the held-out game was played with each model used once, a
fixed-effects analysis separates prompts, guessers and authors (`guess.prompt_effects`), votes are
also weighted by each guesser's skill on the other half, and a third coder settled the disputed
labels (`spectra.tiebreak`; the summary is read from `data/spectra/oct-23-26/tiebreak.json`).
