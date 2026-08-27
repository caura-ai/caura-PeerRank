"""
figure_utils.py - Shared helpers for the validation figure generators.

Centralizes the publication style config, model color palette, and the small
plotting/analysis helpers that were previously duplicated verbatim across
generate_figures_TFQ.py and generate_figures_GSM8K.py (and, for save_figure and
get_text_color_for_background, generate_figures_PeerRank.py).

Kept here rather than in peerrank/validation_utils.py on purpose: validation_utils
is imported by the validate_*.py scripts, which must run without matplotlib
installed (matplotlib is an optional [figures] extra). Importing this module pulls
in matplotlib, so only the figure generators should import it.

MODEL_COLORS is the single palette for every figure generator (PeerRank, TFQ,
GSM8K). Keys are model DISPLAY names (ALL_MODELS[*]['name']), not model_ids, and
it covers the whole roster - active and cost-tracking-only alike. Colors are
assigned explicitly rather than derived from ALL_MODELS order so a model keeps
its color across paper revisions when the roster changes; the cost of that is
that a rename needs a key here too, which _warn_missing_model_colors flags at
import time.
"""

import warnings
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from peerrank.config import calculate_scores_from_evaluations, MODELS
from peerrank.models import ALL_MODELS


# Publication-Quality Style Settings
STYLE_CONFIG = {
    # Fonts
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,

    # Figure size (inches)
    'figure.figsize': (7, 4.5),
    'figure.dpi': 150,

    # Export settings
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,

    # PDF font embedding
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    # Clean style
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',

    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
}

# Model palette, keyed by display name (see module docstring).
#
# Each provider gets one hue and a dark->light ramp inside it, so families read
# as a group in a legend and same-provider models separate by lightness, which
# is the one channel dichromacy leaves intact.
#
# Verified with CIE76 dE on the palette and on Vienot-1999 protanope/deuteranope
# simulations of it. Worst pair, minimum over {normal, deutan, protan}:
#   active roster (10 models):  dE 11.0  - nothing below 10 in any vision
#   the rev-z2 roster (9):      dE  8.4  (claude-sonnet-5 / gemini-3.1-flash-lite, protan)
#   all 22 entries at once:     dE  2.9  (deepseek-v4-flash / gemini-3.1-pro-preview, protan)
# So this is dichromat-safe at the ~10 series a real run plots, NOT at 22 - 22
# mutually distinguishable categorical colors do not exist under dichromacy.
# Any figure that activates most of the roster must carry direct labels rather
# than lean on the legend. Re-measure before reshuffling these values.
MODEL_COLORS = {
    # OpenAI - blues
    'gpt-5.6-sol': '#08306B',            # Navy
    'gpt-5.6-terra': '#2171B5',          # Medium Blue
    'gpt-5.6-luna': '#6BAED6',           # Light Blue
    'gpt-5-mini': '#9ECAE1',             # Pale Blue
    'gpt-5-nano': '#DEEBF7',             # Ice Blue

    # Anthropic - greens
    'claude-fable-5': '#00441B',         # Dark Forest
    'claude-opus-5': '#238B45',          # Green
    'claude-sonnet-5': '#41AB5D',        # Medium Green
    'claude-haiku-4-5': '#A1D99B',       # Light Green

    # Google - orange ramp topped with yellow
    'gemini-3.1-pro-preview': '#7F2704', # Dark Rust
    'gemini-3.5-flash': '#D94801',       # Burnt Orange
    'gemini-3.5-flash-lite': '#F16913',  # Orange
    'gemini-3.1-flash-lite': '#FDAE6B',  # Peach
    'gemini-3.7-flash': '#F0E442',       # Yellow

    # xAI - pinks
    'grok-4.6': '#CC79A7',               # Pink
    'grok-code-fast-1': '#88356A',       # Dark Magenta

    # DeepSeek - golds
    'deepseek-v4-pro': '#E69F00',        # Gold
    'deepseek-v4-flash': '#663300',      # Dark Brown

    # Single-model providers
    'llama-3.3-70b': '#4D4D4D',          # Dark Gray  (Meta / Together)
    'kimi-k3': '#8C564B',                # Brown      (Moonshot)
    'mistral-large': '#17BECF',          # Cyan       (Mistral)
    'pplx-agent-medium': '#4B0082',      # Indigo     (Perplexity)
}

# Default color for unknown models
DEFAULT_COLOR = '#666666'


def get_color(model: str) -> str:
    """Get color for a model with fallback."""
    return MODEL_COLORS.get(model, DEFAULT_COLOR)


def missing_model_colors() -> list[str]:
    """Display names in ALL_MODELS that have no MODEL_COLORS entry."""
    return [m["name"] for m in ALL_MODELS if m["name"] not in MODEL_COLORS]


def _warn_missing_model_colors() -> None:
    """Warn once at import if the roster has outgrown the palette.

    Without this a rename is silent: get_color falls back to DEFAULT_COLOR and
    the figures just render grey. That is how this map drifted a whole model
    generation behind the roster.
    """
    missing = missing_model_colors()
    if missing:
        warnings.warn(
            "MODEL_COLORS has no entry for: " + ", ".join(missing)
            + " - these will render as DEFAULT_COLOR. Add them in peerrank/figure_utils.py.",
            stacklevel=2,
        )


_warn_missing_model_colors()


def get_text_color_for_background(hex_color: str) -> str:
    """Determine if white or black text is more readable on a given background."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if luminance < 0.5 else 'black'


def get_comparison_data(data: dict) -> list:
    """Extract comparison data from analysis or compute from raw data."""
    if 'analysis' in data and 'comparison' in data['analysis']:
        return data['analysis']['comparison']

    # Compute from raw data if analysis not available
    if 'phase3' not in data or 'phase4' not in data:
        return []

    evaluations = data['phase3'].get('evaluations_by_mode', {}).get(
        'shuffle_blind', data['phase3'].get('evaluations', {})
    )
    model_names = [n for _, _, n in MODELS]
    scores_result = calculate_scores_from_evaluations(evaluations, model_names)

    peer_means = {m: mean(s) for m, s in scores_result['peer_scores'].items() if s}
    truth_summary = data['phase4'].get('summary', {})

    comparison = []
    for model in peer_means:
        if model in truth_summary:
            comparison.append({
                'model': model,
                'peer_score': round(peer_means[model], 2),
                'truth_score': truth_summary[model].get('mean', 0),
                'accuracy': truth_summary[model].get('accuracy', 0),
            })

    comparison.sort(key=lambda x: -x['peer_score'])
    for i, row in enumerate(comparison):
        row['peer_rank'] = i + 1

    # Calculate truth ranks with tie handling
    by_truth = sorted(comparison, key=lambda x: -x['truth_score'])
    i = 0
    while i < len(by_truth):
        score = by_truth[i]['truth_score']
        tied = [by_truth[i]]
        j = i + 1
        while j < len(by_truth) and by_truth[j]['truth_score'] == score:
            tied.append(by_truth[j])
            j += 1
        avg_rank = (i + 1 + j) / 2
        for item in tied:
            item['truth_rank'] = avg_rank if len(tied) > 1 else i + 1
        i = j

    return comparison


def get_correlation_stats(data: dict) -> dict:
    """Get or compute correlation statistics."""
    if 'analysis' in data and 'correlation' in data['analysis']:
        return data['analysis']['correlation']

    comparison = get_comparison_data(data)
    if len(comparison) < 3 or not HAS_SCIPY:
        return {}

    peer_arr = [c['peer_score'] for c in comparison]
    truth_arr = [c['truth_score'] for c in comparison]

    if len(set(truth_arr)) == 1:
        return {'pearson_r': 0, 'pearson_p': 1, 'spearman_r': 0, 'spearman_p': 1}

    pearson_r, pearson_p = pearsonr(peer_arr, truth_arr)
    spearman_r, spearman_p = spearmanr(peer_arr, truth_arr)

    return {
        'pearson_r': round(pearson_r, 4),
        'pearson_p': round(pearson_p, 4),
        'spearman_r': round(spearman_r, 4),
        'spearman_p': round(spearman_p, 4),
    }


def save_figure(fig, output_dir: Path, name: str):
    """Save figure in both PDF and PNG formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"  Saved: {pdf_path}")

    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=600)
    print(f"  Saved: {png_path}")

    plt.close(fig)
