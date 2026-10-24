# Spectra article (first draft)

A first draft of the Spectra paper for arXiv (cs.CL), built from the two most recent runs:
`oct-23-26` (main run, with the three guessing games) and `oct-14-26` (replication).
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

Needs the run data in `data/spectra/oct-23-26/` and `data/spectra/oct-14-26/` (not in git) and the
`spectra` extra (`pip install -e ".[spectra]"`), plus matplotlib and scipy.

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
- third-coder tie-break on the disputed labels (plan gap 1, next step)
- bibliography entries cited from memory or with incomplete titles and author lists

Done since the first draft: the classifier is also scored jointly per prompt and on the game's
halves (`analyze.distinctiveness`), the held-out game was played with each model used once, and a
fixed-effects analysis separates prompts, guessers and authors (`guess.prompt_effects`).
