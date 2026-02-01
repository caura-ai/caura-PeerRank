"""
validation_utils.py - Shared utilities for validation scripts (GSM8K, TruthfulQA, MMLU)

Consolidates duplicate functions from validate_*.py scripts:
- File I/O (load/save JSON with revision suffix)
- Progress display
- Confidence interval calculations
- Phase detection
"""

import json
from math import sqrt
from pathlib import Path


def load_validation_json(directory: Path, filename: str, revision: str) -> dict:
    """Load JSON file with revision suffix from specified directory.

    Args:
        directory: Path to the validation data directory (e.g., DATA_DIR / "GSM8K")
        filename: Base filename (e.g., "phase1_questions.json")
        revision: Revision tag (e.g., "GSM8K", "TFQ", "MMLU")

    Returns:
        Parsed JSON as dict

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    base, ext = filename.rsplit(".", 1)
    filepath = directory / f"{base}_{revision}.{ext}"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath.name} not found")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_validation_json(directory: Path, filename: str, revision: str, data: dict):
    """Save JSON file with revision suffix to specified directory.

    Args:
        directory: Path to the validation data directory
        filename: Base filename (e.g., "phase2_answers.json")
        revision: Revision tag
        data: Data to save
    """
    directory.mkdir(parents=True, exist_ok=True)
    base, ext = filename.rsplit(".", 1)
    filepath = directory / f"{base}_{revision}.{ext}"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath.name}")


def progress_bar(completed: int, total: int, width: int = 40) -> str:
    """Generate ASCII progress bar.

    Args:
        completed: Number of completed items
        total: Total number of items
        width: Bar width in characters (default 40)

    Returns:
        Formatted progress bar string like "[=====>....] 50% (5/10)"
    """
    if total == 0:
        return "[" + "." * width + "] 0%"
    pct = completed * 100 // total
    filled = pct * width // 100
    bar = "=" * filled + ">" + "." * (width - filled - 1) if filled < width else "=" * width
    return f"[{bar}] {pct:3}% ({completed}/{total})"


def get_last_completed_phase(directory: Path, revision: str, phase_files: list[tuple[int, str]]) -> int:
    """Detect highest completed phase by checking for output files.

    Args:
        directory: Path to the validation data directory
        revision: Revision tag
        phase_files: List of (phase_number, filename_base) tuples, ordered from highest to lowest.
                    Example: [(5, "GSM8K_analysis"), (4, "phase4_GSM8K_scores"), ...]

    Returns:
        Highest completed phase number (0 if none)
    """
    for phase, fn in phase_files:
        if (directory / f"{fn}_{revision}.json").exists():
            return phase
    return 0


def correlation_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute confidence interval for Pearson r using Fisher z-transformation.

    Args:
        r: Pearson correlation coefficient
        n: Sample size
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the CI
    """
    from scipy.stats import norm
    import math

    if abs(r) >= 1.0 or n < 4:
        return (r, r)

    z = 0.5 * math.log((1 + r) / (1 - r))  # Fisher z
    se = 1 / math.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)

    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    r_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
    r_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)

    return (round(r_lo, 4), round(r_hi, 4))


def wilson_ci(correct: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (accuracy).

    More accurate than normal approximation for small samples or extreme proportions.

    Args:
        correct: Number of correct items
        total: Total number of items
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the CI
    """
    from scipy.stats import norm

    if total == 0:
        return (0.0, 0.0)

    p = correct / total
    z = norm.ppf(1 - alpha / 2)

    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom

    return (round(max(0, center - margin), 4), round(min(1, center + margin), 4))


def peer_score_ci(scores: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval for mean peer score using t-distribution.

    Args:
        scores: List of peer scores
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the CI
    """
    from scipy.stats import t
    from statistics import mean, stdev

    if len(scores) < 2:
        m = scores[0] if scores else 0
        return (m, m)

    n = len(scores)
    m = mean(scores)
    se = stdev(scores) / sqrt(n)
    t_crit = t.ppf(1 - alpha / 2, n - 1)

    return (round(m - t_crit * se, 2), round(m + t_crit * se, 2))
