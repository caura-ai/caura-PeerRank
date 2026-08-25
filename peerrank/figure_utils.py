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

NOTE: the MODEL_COLORS map below is the shared copy moved verbatim from the two
validation figure scripts. Its keys are still keyed to the older (arXiv-run) model
names, so several current-roster models fall back to DEFAULT_COLOR — that content
staleness is a separate fix, but it now only needs to be made in one place.
"""

from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from peerrank.config import calculate_scores_from_evaluations, MODELS


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

# Colorblind-safe palette for models
MODEL_COLORS = {
    'gpt-5.5': '#0047AB',              # Cobalt Blue (darker)
    'gpt-5-mini': '#56B4E9',
    'claude-opus-5': '#029E73',
    'claude-sonnet-5': '#78C679',
    'gemini-3-pro-preview': '#D55E00',
    'gemini-3.6-flash': '#F0E442',
    'grok-4.3': '#CC79A7',
    'deepseek-v4-flash': '#E69F00',
    'llama-3.3-70b': '#999999',
    'sonar-pro': '#9467BD',
    'kimi-k2.6': '#8C564B',
    'mistral-large': '#17BECF',
}

# Default color for unknown models
DEFAULT_COLOR = '#666666'


def get_color(model: str) -> str:
    """Get color for a model with fallback."""
    return MODEL_COLORS.get(model, DEFAULT_COLOR)


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
