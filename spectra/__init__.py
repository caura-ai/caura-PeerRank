"""
Spectra - behavioral profiles of the PeerRank roster

Profiles how each model answers 200 everyday prompts: style, habits, and behavior when a
prompt carries a trap (false premise, pasted error, vague request, stated opinion,
emotional framing). It is not a ranking. Models are named by their PeerRank names and called
through PeerRank's providers with Phase 2's settings, so a Spectra revision describes the same
deployment as the PeerRank revision of the same name.

Run from the repo root:

  python -m spectra.run --revision oct-14-26 --limit 5   # smoke test: 5 prompts x every model
  python -m spectra.run --revision oct-14-26             # full run (resumable; re-run to retry errors)
  python -m spectra.code_behaviors --revision oct-14-26  # optional: behavior codes from coder models
  python -m spectra.analyze --revision oct-14-26         # features, traits, signature words, classifier
  python -m spectra.report --revision oct-14-26          # -> data/spectra/oct-14-26/report.html

--revision defaults to config.yaml's revision. Output goes to data/spectra/<revision>/.

Caveats to state when publishing: the prompts are synthetic and English-only; effect sizes are
relative to the models in the run, so adding or removing a model shifts everyone; tag-based
behaviors rest on 5-11 prompts each, enough for large differences and not for fine ones; hand-check
100-200 coder labels before trusting the behavior traits.
"""

from pathlib import Path

SPECTRA_DIR = Path(__file__).resolve().parent
DATA_ROOT = SPECTRA_DIR.parent / "data" / "spectra"
DEFAULT_CONFIG = SPECTRA_DIR / "config.yaml"


def _entry(m) -> dict:
    """A model or coder entry is a PeerRank name, or a dict with at least a name."""
    return {"name": m} if isinstance(m, str) else dict(m)


def load_config(path=None) -> dict:
    import yaml  # here so --help works without it

    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    cfg.setdefault("samples", 1)
    cfg.setdefault("concurrency_per_model", 4)
    cfg.setdefault("timeout_seconds", 180)
    cfg.setdefault("retries", 3)
    cfg["models"] = [_entry(m) for m in cfg.get("models") or []]
    cfg["coders"] = [_entry(c) for c in cfg.get("coders") or []]
    prompts = Path(cfg["prompts"])
    cfg["prompts"] = str(prompts if prompts.is_absolute() else SPECTRA_DIR / prompts)
    return cfg


def add_run_args(ap):
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--revision", default=None, help="PeerRank-style revision tag, e.g. oct-14-26")


def run_dir(cfg: dict, revision: str | None = None) -> Path:
    rev = (revision or cfg.get("revision") or "").strip()
    if not rev:
        raise SystemExit("No revision: pass --revision or set revision in config.yaml")
    cfg["revision"] = rev
    return DATA_ROOT / rev
