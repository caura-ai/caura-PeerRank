"""
PeerRank Figure Generator for Scientific Publication

Generates publication-quality figures from PeerRank evaluation data.
Output: PDF (vector) + PNG (600 DPI) for Overleaf/LaTeX compatibility.

Usage:
    python generate_figures.py                    # Use latest revision
    python generate_figures.py --revision v11    # Specific revision
    python generate_figures.py --output figures  # Custom output dir
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats

from peerrank.config import (
    calculate_scores_from_evaluations, calculate_judge_agreement, calculate_question_stats,
    calculate_elo_ratings, _pearson_correlation, _spearman_correlation,
)

# =============================================================================
# Publication-Quality Style Settings
# =============================================================================

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

    # Figure size (inches) - single column ~3.5", double column ~7"
    'figure.figsize': (7, 4.5),
    'figure.dpi': 150,

    # Export settings
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,

    # PDF font embedding (Type 42 = TrueType, preferred by journals)
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

# Colorblind-safe palette for 12 models (expanded Paul Tol palette)
MODEL_COLORS = {
    'gpt-5.2': '#0173B2',              # Blue
    'gpt-5-mini': '#56B4E9',           # Light Blue
    'claude-opus-4-5': '#029E73',      # Green
    'claude-sonnet-4-5': '#78C679',    # Light Green
    'gemini-3-pro-preview': '#D55E00', # Orange
    'gemini-3-flash-thinking': '#F0E442', # Yellow
    'grok-4-1-fast': '#CC79A7',        # Pink
    'deepseek-chat': '#E69F00',        # Orange-Brown
    'llama-4-maverick': '#999999',     # Gray
    'sonar-pro': '#9467BD',            # Purple
    'kimi-k2-0905': '#8C564B',         # Brown
    'mistral-large': '#17BECF',        # Cyan
}

def get_color(model):
    """Get color for a model, with fallback."""
    return MODEL_COLORS.get(model, '#666666')


def get_text_color_for_background(hex_color):
    """Determine if white or black text is more readable on a given background color.

    Uses relative luminance calculation (WCAG standard).
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert to RGB
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    # Calculate relative luminance
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Use white text for dark backgrounds, black for light backgrounds
    return 'white' if luminance < 0.5 else 'black'


# =============================================================================
# Data Loading
# =============================================================================

def load_data(revision: str, data_dir: Path):
    """Load all phase data for a revision."""

    phase1_path = data_dir / f"phase1_questions_{revision}.json"
    phase2_path = data_dir / f"phase2_answers_{revision}.json"
    phase3_path = data_dir / f"phase3_rankings_{revision}.json"

    data = {}

    if phase1_path.exists():
        with open(phase1_path, encoding='utf-8') as f:
            data['phase1'] = json.load(f)

    if phase2_path.exists():
        with open(phase2_path, encoding='utf-8') as f:
            data['phase2'] = json.load(f)

    if phase3_path.exists():
        with open(phase3_path, encoding='utf-8') as f:
            data['phase3'] = json.load(f)

    return data


def get_rankings(data: dict):
    """Extract ranked model data from phase3 evaluations."""

    evaluations = data['phase3'].get('evaluations', {})
    models = list(evaluations.keys())
    scores = calculate_scores_from_evaluations(evaluations, models)

    rankings = []
    for model in models:
        peer = scores['peer_scores'].get(model, [])
        self_ = scores['self_scores'].get(model, [])
        given = scores['judge_given'].get(model, [])

        rankings.append({
            'model': model,
            'peer_score': np.mean(peer) if peer else 0,
            'peer_std': np.std(peer) if peer else 0,
            'self_score': np.mean(self_) if self_ else 0,
            'self_bias': (np.mean(self_) - np.mean(peer)) if (peer and self_) else 0,
            'judge_avg': np.mean(given) if given else 0,
            'judge_std': np.std(given) if given else 0,
        })

    rankings.sort(key=lambda x: x['peer_score'], reverse=True)
    return rankings


def get_timing_data(data: dict):
    """Extract timing data from phase2 and phase3."""

    timing = {}

    # Phase 2 answer times
    if 'phase2' in data and 'timing_stats' in data['phase2']:
        timing['answer'] = {
            model: stats['avg']
            for model, stats in data['phase2']['timing_stats'].items()
        }

    # Phase 3 evaluation times
    if 'phase3' in data and 'timing_stats' in data['phase3']:
        timing['eval'] = {
            model: stats['avg']
            for model, stats in data['phase3']['timing_stats'].items()
        }

    return timing


def get_cross_eval_matrix(data: dict):
    """Build evaluator x evaluated score matrix."""

    evaluations = data['phase3'].get('evaluations', {})
    models = list(evaluations.keys())
    n = len(models)

    matrix = np.zeros((n, n))
    counts = np.zeros((n, n))

    for i, evaluator in enumerate(models):
        for question, ratings in evaluations.get(evaluator, {}).items():
            for j, rated_model in enumerate(models):
                if rated_model in ratings:
                    rating = ratings[rated_model]
                    score = rating.get('score', rating) if isinstance(rating, dict) else rating
                    if score is not None:
                        try:
                            matrix[i, j] += float(score)
                            counts[i, j] += 1
                        except (ValueError, TypeError):
                            continue

    # Average (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_matrix = np.zeros_like(matrix)
        np.divide(matrix, counts, out=avg_matrix, where=counts > 0)
        avg_matrix[counts == 0] = np.nan
        matrix = avg_matrix

    return matrix, models


# =============================================================================
# Figure Generation
# =============================================================================

def save_figure(fig, output_dir: Path, name: str):
    """Save figure in both PDF and PNG formats."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # PDF (vector, primary)
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"  Saved: {pdf_path}")

    # PNG (raster, fallback)
    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=600)
    print(f"  Saved: {png_path}")

    plt.close(fig)


def generate_fig4_peer_rankings(data: dict, output_dir: Path):
    """Figure 3: Horizontal bar chart of peer rankings with error bars (RESULTS)."""

    print("\nGenerating Figure 3: Peer Rankings...")

    rankings = get_rankings(data)

    fig, ax = plt.subplots(figsize=(7, 4))

    y_pos = np.arange(len(rankings))
    models = [r['model'] for r in rankings]
    scores = [r['peer_score'] for r in rankings]
    stds = [r['peer_std'] for r in rankings]
    colors = [get_color(r['model']) for r in rankings]

    bars = ax.barh(y_pos, scores, xerr=stds, color=colors,
                   edgecolor='white', linewidth=0.5,
                   error_kw={'capsize': 3, 'capthick': 1, 'elinewidth': 1},
                   alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlabel('Peer Score (mean ± std)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=11)

    # Add score labels
    for i, (score, std) in enumerate(zip(scores, stds)):
        ax.text(score + std + 0.15, i, f'{score:.2f}', va='center', fontsize=11)

    ax.set_title('Peer Evaluation Rankings', fontweight='bold', fontsize=14, pad=10)

    save_figure(fig, output_dir, 'fig4_peer_rankings')


def generate_fig7_peer_score_vs_time(data: dict, output_dir: Path):
    """Figure 7: Scatter plot of peer score vs response time (RESULTS)."""

    print("\nGenerating Figure 7: Peer Score vs Response Time...")

    rankings = get_rankings(data)
    timing = get_timing_data(data)

    if 'answer' not in timing:
        print("  Skipping: No timing data available")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Collect all points first
    points = []
    for r in rankings:
        model = r['model']
        if model not in timing['answer']:
            continue
        x = timing['answer'][model]
        y = r['peer_score']
        points.append({'model': model, 'x': x, 'y': y, 'color': get_color(model)})

    # Sort by x position for consistent labeling
    points.sort(key=lambda p: p['x'])

    # Detect and fix overlapping points (jitter them slightly)
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i >= j:
                continue
            # Check if points are too close
            x_dist = abs(p1['x'] - p2['x'])
            y_dist = abs(p1['y'] - p2['y'])
            if x_dist < 0.5 and y_dist < 0.15:
                # Jitter the second point slightly
                p2['y'] += 0.12
                p2['jittered'] = True

    # Plot all points
    for i, p in enumerate(points):
        ax.scatter(p['x'], p['y'], s=150, c=p['color'],
                   edgecolor='white', linewidth=1.5, zorder=3 + i)  # Increment zorder

    # Get axis limits for boundary checking
    x_values = [p['x'] for p in points]
    x_min, x_max = min(x_values), max(x_values)
    x_range = x_max - x_min
    x_pad = x_range * 0.15
    ax.set_xlim(max(0, x_min - x_pad), x_max + x_pad)  # Tight fit with small padding

    # Smart label placement - alternate above/below and use connecting lines
    placed_labels = []  # Track (x, y, width, height) of placed labels
    right_edge = x_max + x_pad * 2  # Don't place labels past this

    for i, p in enumerate(points):
        x, y, model = p['x'], p['y'], p['model']

        # Check if point is near edges
        near_right_edge = x > (x_max - x_range * 0.2)
        near_left_edge = x < (x_min + x_range * 0.2)

        # Base offset scales with data range
        x_off = x_range * 0.08
        y_off = 0.15

        # Try different positions based on location
        if near_right_edge:
            positions = [
                (x - x_off, y, 'right', 'center'),           # left
                (x, y + y_off, 'center', 'bottom'),          # above
                (x, y - y_off, 'center', 'top'),             # below
                (x - x_off, y + y_off, 'right', 'bottom'),   # left-above
                (x - x_off, y - y_off, 'right', 'top'),      # left-below
                (x - x_off*1.5, y, 'right', 'center'),       # far left
                (x, y + y_off*2, 'center', 'bottom'),        # far above
                (x, y - y_off*2, 'center', 'top'),           # far below
            ]
        elif near_left_edge:
            positions = [
                (x + x_off, y, 'left', 'center'),            # right
                (x, y + y_off, 'center', 'bottom'),          # above
                (x, y - y_off, 'center', 'top'),             # below
                (x + x_off, y + y_off, 'left', 'bottom'),    # right-above
                (x + x_off, y - y_off, 'left', 'top'),       # right-below
                (x + x_off*1.5, y, 'left', 'center'),        # far right
                (x, y + y_off*2, 'center', 'bottom'),        # far above
                (x, y - y_off*2, 'center', 'top'),           # far below
            ]
        else:
            positions = [
                (x + x_off, y, 'left', 'center'),            # right
                (x - x_off, y, 'right', 'center'),           # left
                (x, y + y_off, 'center', 'bottom'),          # above
                (x, y - y_off, 'center', 'top'),             # below
                (x + x_off, y + y_off, 'left', 'bottom'),    # right-above
                (x + x_off, y - y_off, 'left', 'top'),       # right-below
                (x - x_off, y + y_off, 'right', 'bottom'),   # left-above
                (x - x_off, y - y_off, 'right', 'top'),      # left-below
                (x + x_off*1.5, y, 'left', 'center'),        # far right
                (x - x_off*1.5, y, 'right', 'center'),       # far left
                (x, y + y_off*2, 'center', 'bottom'),        # far above
                (x, y - y_off*2, 'center', 'top'),           # far below
            ]

        # Find best position (least overlap with other labels AND points)
        best_pos = positions[-1]  # Default to last (furthest)
        for tx, ty, ha, va in positions:
            overlap = False
            # Check overlap with existing labels
            for lx, ly, lw, lh in placed_labels:
                if abs(tx - lx) < x_range * 0.12 and abs(ty - ly) < 0.22:
                    overlap = True
                    break
            # Check overlap with other points
            if not overlap:
                for other in points:
                    if other['model'] != model:
                        if abs(tx - other['x']) < x_range * 0.06 and abs(ty - other['y']) < 0.18:
                            overlap = True
                            break
            if not overlap:
                best_pos = (tx, ty, ha, va)
                break

        tx, ty, ha, va = best_pos

        ax.annotate(model, (x, y),
                    xytext=(tx, ty),
                    fontsize=12, ha=ha, va=va,
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5,
                                   shrinkA=0, shrinkB=3),
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                             edgecolor='none', alpha=0.8),
                    zorder=4)

        # Track this label position (approximate)
        placed_labels.append((tx, ty, 2, 0.2))

    ax.set_xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peer Score', fontsize=12, fontweight='bold')
    ax.set_ylim(6, 10)
    ax.tick_params(axis='both', labelsize=11)

    # Add quadrant shading
    mid_x = np.median([p['x'] for p in points])
    mid_y = np.median([p['y'] for p in points])

    ax.axhline(mid_y, color='gray', linestyle=':', alpha=0.5, zorder=1)
    ax.axvline(mid_x, color='gray', linestyle=':', alpha=0.5, zorder=1)

    # Quadrant labels
    ax.text(0.02, 0.98, 'Fast & High Quality', transform=ax.transAxes,
            fontsize=11, va='top', ha='left', color='#009988', alpha=0.8, fontweight='bold')
    ax.text(0.98, 0.02, 'Slow & Low Quality', transform=ax.transAxes,
            fontsize=11, va='bottom', ha='right', color='#CC3311', alpha=0.8)

    ax.set_title('Peer Score vs Response Time', fontweight='bold', fontsize=14, pad=10)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig7_peer_score_vs_time')


def generate_fig5_cross_eval_heatmap(data: dict, output_dir: Path):
    """Figure 4: Heatmap of evaluator x evaluated scores (RESULTS)."""

    print("\nGenerating Figure 4: Cross-Evaluation Heatmap...")

    matrix, models = get_cross_eval_matrix(data)
    short_labels = models

    fig, ax = plt.subplots(figsize=(8, 6))

    # Mask for diagonal (self-ratings)
    mask = np.eye(len(models), dtype=bool)

    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=short_labels, yticklabels=short_labels,
                vmin=5, vmax=10, center=7.5,
                annot_kws={'fontsize': 11},
                cbar_kws={'label': 'Score', 'shrink': 0.8},
                linewidths=0.5, linecolor='white',
                ax=ax)

    # Highlight diagonal
    for i in range(len(models)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor='black', linewidth=2))

    ax.set_xlabel('Evaluated Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Evaluator Model', fontweight='bold', fontsize=12)
    ax.set_title('Cross-Evaluation Matrix\n(diagonal = self-ratings)',
                 fontweight='bold', fontsize=14, pad=10)

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    # Increase colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Score', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    save_figure(fig, output_dir, 'fig5_cross_eval_heatmap')


# =============================================================================
# Bias Analysis Figures (requires multi-mode data)
# =============================================================================

def has_multimode_data(data: dict) -> bool:
    """Check if data has evaluations from all 3 bias modes."""
    return 'evaluations_by_mode' in data.get('phase3', {})


def calculate_mode_scores(data: dict):
    """Calculate scores for each model across all 3 evaluation modes."""

    if not has_multimode_data(data):
        return None

    evals_by_mode = data['phase3']['evaluations_by_mode']
    models = list(evals_by_mode.get('shuffle_blind', {}).keys())

    mode_scores = {mode: {} for mode in ['shuffle_only', 'blind_only', 'shuffle_blind']}

    for mode_name, evaluations in evals_by_mode.items():
        scores = calculate_scores_from_evaluations(evaluations, models)
        for model in models:
            peer = scores['peer_scores'].get(model, [])
            self_ = scores['self_scores'].get(model, [])
            peer_se = np.std(peer, ddof=1) / np.sqrt(len(peer)) if len(peer) > 1 else 0
            self_se = np.std(self_, ddof=1) / np.sqrt(len(self_)) if len(self_) > 1 else 0
            mode_scores[mode_name][model] = {
                'peer_avg': np.mean(peer) if peer else 0,
                'peer_std': np.std(peer) if peer else 0,
                'peer_se': peer_se,
                'self_avg': np.mean(self_) if self_ else 0,
                'self_se': self_se,
                'n_peer': len(peer),
                'n_self': len(self_),
            }

    return mode_scores


def calculate_position_bias(data: dict):
    """Calculate scores by position in blind evaluation mode.

    UNIFIED BIAS CONVENTION: Positive = factor HELPED the model
    Position Bias = Blind − Peer (positive = fixed position helped)
    """

    if not has_multimode_data(data):
        return None

    # In blind_only mode, order is fixed: model list order = position
    evals_by_mode = data['phase3']['evaluations_by_mode']
    blind_evals = evals_by_mode.get('blind_only', {})
    shuffle_blind_evals = evals_by_mode.get('shuffle_blind', {})

    # Get model order (fixed in blind mode)
    models = list(blind_evals.keys())

    # Calculate average score per position in blind mode
    position_scores = {i: [] for i in range(len(models))}

    for evaluator, questions in blind_evals.items():
        for question, ratings in questions.items():
            for i, model in enumerate(models):
                if model in ratings:
                    rating = ratings[model]
                    score = rating.get('score', rating) if isinstance(rating, dict) else rating
                    if score is not None and evaluator != model:  # Exclude self-ratings
                        try:
                            position_scores[i].append(float(score))
                        except (ValueError, TypeError):
                            pass

    # Calculate peer scores from shuffle_blind for comparison
    peer_scores = {}
    for evaluator, questions in shuffle_blind_evals.items():
        for question, ratings in questions.items():
            for model, rating in ratings.items():
                score = rating.get('score', rating) if isinstance(rating, dict) else rating
                if score is not None and evaluator != model:
                    try:
                        if model not in peer_scores:
                            peer_scores[model] = []
                        peer_scores[model].append(float(score))
                    except (ValueError, TypeError):
                        pass

    # Build position bias data
    # Position Bias = Blind − Peer (positive = position helped)
    result = []
    for i, model in enumerate(models):
        blind_scores = position_scores[i]
        peer_scores_list = peer_scores.get(model, [])
        blind_avg = np.mean(blind_scores) if blind_scores else 0
        peer_avg = np.mean(peer_scores_list) if peer_scores_list else 0

        # Calculate standard error for confidence intervals
        blind_se = np.std(blind_scores, ddof=1) / np.sqrt(len(blind_scores)) if len(blind_scores) > 1 else 0
        peer_se = np.std(peer_scores_list, ddof=1) / np.sqrt(len(peer_scores_list)) if len(peer_scores_list) > 1 else 0
        # Combined SE for difference (assuming independence)
        bias_se = np.sqrt(blind_se**2 + peer_se**2)

        result.append({
            'position': i + 1,
            'model': model,
            'blind_score': blind_avg,
            'peer_score': peer_avg,
            'pos_bias': blind_avg - peer_avg,  # Positive = position helped
            'bias_se': bias_se,
            'n_blind': len(blind_scores),
            'n_peer': len(peer_scores_list),
        })

    return result


def calculate_name_bias(data: dict):
    """Calculate name bias: effect of showing model names.

    UNIFIED BIAS CONVENTION: Positive = factor HELPED the model
    Name Bias = Shuffle − Peer (positive = name recognition helped)
    """

    if not has_multimode_data(data):
        return None

    mode_scores = calculate_mode_scores(data)
    if not mode_scores:
        return None

    models = list(mode_scores['shuffle_blind'].keys())

    result = []
    for model in models:
        peer = mode_scores['shuffle_blind'][model]['peer_avg']
        shuffle = mode_scores['shuffle_only'][model]['peer_avg']
        self_score = mode_scores['shuffle_blind'][model]['self_avg']

        # Standard errors for CI calculation
        peer_se = mode_scores['shuffle_blind'][model]['peer_se']
        shuffle_se = mode_scores['shuffle_only'][model]['peer_se']
        self_se = mode_scores['shuffle_blind'][model]['self_se']

        # Combined SE for differences (assuming independence)
        name_bias_se = np.sqrt(shuffle_se**2 + peer_se**2)
        self_bias_se = np.sqrt(self_se**2 + peer_se**2)

        result.append({
            'model': model,
            'peer_score': peer,
            'shuffle_score': shuffle,
            'self_score': self_score,
            'name_bias': shuffle - peer,  # Positive = name helped
            'self_bias': self_score - peer,  # Positive = overrates self
            'name_bias_se': name_bias_se,
            'self_bias_se': self_bias_se,
        })

    # Sort by peer score
    result.sort(key=lambda x: x['peer_score'], reverse=True)
    return result


def generate_fig14_judge_generosity(data: dict, output_dir: Path):
    """Figure 14: Judge generosity - how lenient/strict each model judges (DISCUSSION)."""

    print("\nGenerating Figure 14: Judge Generosity...")

    rankings = get_rankings(data)

    # Sort by judge_avg (most generous at top)
    by_gen = sorted(rankings, key=lambda x: x['judge_avg'], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    models = [r['model'] for r in by_gen]
    avgs = [r['judge_avg'] for r in by_gen]
    stds = [r['judge_std'] for r in by_gen]
    colors = [get_color(r['model']) for r in by_gen]

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, avgs, xerr=stds, color=colors, edgecolor='white', alpha=0.85,
                   error_kw={'capsize': 3, 'capthick': 1, 'elinewidth': 1})

    mean_avg = np.mean(avgs)
    ax.axvline(mean_avg, color='black', linewidth=0.8, linestyle='--',
               label=f'Mean: {mean_avg:.2f}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Average Score Given (± std)')
    ax.set_xlim(5, 10)
    ax.set_title('Judge Generosity\n(higher = more lenient)', fontweight='bold', pad=10)
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=8)

    # Add value labels (clip to stay within xlim)
    xlim_max = 10
    for i, (avg, std) in enumerate(zip(avgs, stds)):
        label_x = avg + std + 0.1
        if label_x > xlim_max - 0.3:
            # Place inside the bar if would go off-chart
            ax.text(avg - 0.1, i, f'{avg:.2f}', va='center', ha='right', fontsize=9, color='white', fontweight='bold')
        else:
            ax.text(label_x, i, f'{avg:.2f}', va='center', ha='left', fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig14_judge_generosity')


def generate_fig15_judge_generosity_vs_peer(data: dict, output_dir: Path):
    """Figure 15: Judge Generosity vs Peer Ranking scatter plot (DISCUSSION)."""

    print("\nGenerating Figure 15: Judge Generosity vs Peer Ranking...")

    rankings = get_rankings(data)

    # Prepare data
    models = [r['model'] for r in rankings]
    peer_scores = [r['peer_score'] for r in rankings]
    judge_generosity = [r['judge_avg'] for r in rankings]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Collect points for collision detection
    points = []
    for i, r in enumerate(rankings):
        points.append({
            'model': r['model'],
            'x': r['peer_score'],
            'y': r['judge_avg'],
            'rank': i + 1,
            'color': get_color(r['model'])
        })

    # Plot all points with rank numbers inside
    for p in points:
        ax.scatter(p['x'], p['y'], s=300, color=p['color'], alpha=0.7,
                  edgecolors='black', linewidths=1.5, zorder=3)
        text_color = get_text_color_for_background(p['color'])
        ax.text(p['x'], p['y'], str(p['rank']),
               ha='center', va='center',
               fontsize=12, fontweight='bold',
               color=text_color, zorder=4)

    # Smart label placement with collision detection
    placed_labels = []
    x_range = max(peer_scores) - min(peer_scores)
    y_range = max(judge_generosity) - min(judge_generosity)

    for p in points:
        x, y, model = p['x'], p['y'], p['model']

        # Try different positions - increased offsets for larger fonts
        positions = [
            # Close positions
            (x + x_range*0.05, y, 'left', 'center'),
            (x - x_range*0.05, y, 'right', 'center'),
            (x, y + y_range*0.08, 'center', 'bottom'),
            (x, y - y_range*0.08, 'center', 'top'),
            # Diagonal close
            (x + x_range*0.05, y + y_range*0.06, 'left', 'bottom'),
            (x + x_range*0.05, y - y_range*0.06, 'left', 'top'),
            (x - x_range*0.05, y + y_range*0.06, 'right', 'bottom'),
            (x - x_range*0.05, y - y_range*0.06, 'right', 'top'),
            # Further positions for crowded areas
            (x + x_range*0.10, y, 'left', 'center'),
            (x - x_range*0.10, y, 'right', 'center'),
            (x, y + y_range*0.12, 'center', 'bottom'),
            (x, y - y_range*0.12, 'center', 'top'),
            # Diagonal far
            (x + x_range*0.08, y + y_range*0.10, 'left', 'bottom'),
            (x + x_range*0.08, y - y_range*0.10, 'left', 'top'),
            (x - x_range*0.08, y + y_range*0.10, 'right', 'bottom'),
            (x - x_range*0.08, y - y_range*0.10, 'right', 'top'),
        ]

        best_pos = positions[0]
        for tx, ty, ha, va in positions:
            overlap = False
            # Check overlap with placed labels - increased thresholds for larger fonts
            for lx, ly in placed_labels:
                if abs(tx - lx) < x_range*0.12 and abs(ty - ly) < y_range*0.10:
                    overlap = True
                    break
            # Also check overlap with other points
            if not overlap:
                for other in points:
                    if other['model'] != model:
                        if abs(tx - other['x']) < x_range*0.04 and abs(ty - other['y']) < y_range*0.06:
                            overlap = True
                            break
            if not overlap:
                best_pos = (tx, ty, ha, va)
                break

        tx, ty, ha, va = best_pos

        ax.annotate(model, xy=(x, y), xytext=(tx, ty),
                   fontsize=14, ha=ha, va=va,
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5),
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='gray', alpha=0.9),
                   zorder=5)
        placed_labels.append((tx, ty))

    # Add trend line
    z = np.polyfit(peer_scores, judge_generosity, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(peer_scores), max(peer_scores), 100)
    ax.plot(x_trend, p(x_trend), "k--", alpha=0.4, linewidth=2)

    # Calculate correlation
    correlation = np.corrcoef(peer_scores, judge_generosity)[0, 1]

    # Styling
    ax.set_xlabel('Peer Score (Performance Ranking)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Judge Generosity (Average Score Given)', fontweight='bold', fontsize=12)
    ax.set_title('Judge Generosity vs Peer Ranking\nDo better models judge more harshly?',
                fontweight='bold', fontsize=14, pad=10)
    ax.tick_params(axis='both', labelsize=11)

    # Add correlation text box in bottom-left (less cluttered area)
    textstr = f'r = {correlation:.3f} (n={len(rankings)})'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes,
           fontsize=12, verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig15_judge_generosity_vs_peer')


def generate_fig17_radar_chart(data: dict, output_dir: Path):
    """Figure 17: Radar/spider chart for multi-dimensional model comparison (DISCUSSION)."""

    print("\nGenerating Figure 17: Radar Chart...")

    rankings = get_rankings(data)
    timing = get_timing_data(data)

    if 'answer' not in timing:
        print("  Skipping: Incomplete timing data")
        return

    # Normalize metrics to 0-1 scale
    metrics = ['Peer Score', 'Consistency', 'Speed', 'Humility', 'Strictness']

    # Calculate normalized values
    model_data = {}

    max_time = max(timing['answer'].values())
    max_std = max(r['peer_std'] for r in rankings)

    for r in rankings:
        model = r['model']
        if model not in timing['answer']:
            continue

        # Peer Score: normalized (0-10 -> 0-1)
        peer_score_norm = r['peer_score'] / 10

        # Consistency: inverse of std (lower std = more consistent)
        consistency = 1 - (r['peer_std'] / max_std) if max_std > 0 else 0.5

        # Speed: inverse of time (faster = higher)
        speed = 1 - (timing['answer'][model] / max_time) if max_time > 0 else 0.5

        # Humility: inverse of self-bias (negative bias = humble)
        humility = 0.5 - (r['self_bias'] / 4)  # Scale bias to roughly 0-1
        humility = max(0, min(1, humility))

        # Strictness: inverse of judge generosity (stricter = lower avg given)
        strictness = 1 - ((r['judge_avg'] - 5) / 5)  # Normalize 5-10 range
        strictness = max(0, min(1, strictness))

        model_data[model] = [peer_score_norm, consistency, speed, humility, strictness]

    # Create radar chart
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, values in model_data.items():
        values_closed = values + values[:1]  # Close the polygon
        ax.plot(angles, values_closed, 'o-', linewidth=2,
                label=model, color=get_color(model), alpha=0.8)
        ax.fill(angles, values_closed, alpha=0.1, color=get_color(model))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=12)

    ax.set_title('Multi-Dimensional Model Comparison', fontweight='bold', pad=20, fontsize=16)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig17_radar_chart')


def generate_fig18_elo_vs_peer(data: dict, output_dir: Path):
    """Figure 18: Elo Ranking vs Peer Ranking comparison (slope graph)."""

    print("\nGenerating Figure 18: Elo vs Peer Ranking...")

    # Get peer rankings
    rankings = get_rankings(data)
    if not rankings:
        print("  Skipping: No ranking data available")
        return

    # Get Elo ratings from evaluations
    evaluations = data['phase3'].get('evaluations', {})
    if not evaluations:
        print("  Skipping: No evaluation data available")
        return

    models = [r['model'] for r in rankings]
    elo_data = calculate_elo_ratings(evaluations, models)

    if not elo_data or not elo_data['ratings']:
        print("  Skipping: Could not calculate Elo ratings")
        return

    # Build comparison data
    peer_scores = [r['peer_score'] for r in rankings]
    elo_ratings = [elo_data['ratings'].get(r['model'], 1500) for r in rankings]

    # Calculate correlations
    pearson_r = _pearson_correlation(peer_scores, elo_ratings)
    spearman_r = _spearman_correlation(peer_scores, elo_ratings)

    # Sort by Elo for Elo ranking
    elo_sorted = sorted(enumerate(rankings), key=lambda x: elo_data['ratings'].get(x[1]['model'], 1500), reverse=True)
    elo_rank = {r['model']: i + 1 for i, (_, r) in enumerate(elo_sorted)}

    # Peer rank is already sorted
    peer_rank = {r['model']: i + 1 for i, r in enumerate(rankings)}

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create slope graph
    left_x, right_x = 0.2, 0.8

    for r in rankings:
        model = r['model']
        pr = peer_rank[model]
        er = elo_rank[model]
        elo_rating = elo_data['ratings'].get(model, 1500)
        peer_score = r['peer_score']
        color = get_color(model)

        # Draw line connecting peer rank to elo rank
        ax.plot([left_x, right_x], [pr, er], '-', color=color, linewidth=2.5, alpha=0.8)

        # Draw points
        ax.scatter(left_x, pr, s=350, color=color, edgecolor='white', linewidth=2, zorder=3)
        ax.scatter(right_x, er, s=350, color=color, edgecolor='white', linewidth=2, zorder=3)

        # Rank numbers inside the points
        text_color = get_text_color_for_background(color)
        ax.text(left_x, pr, str(pr), ha='center', va='center', fontsize=14,
                color=text_color, fontweight='bold', zorder=4)
        ax.text(right_x, er, str(er), ha='center', va='center', fontsize=14,
                color=text_color, fontweight='bold', zorder=4)

        # Model name and peer score on the left
        ax.text(left_x - 0.05, pr, f"{model} ({peer_score:.2f})", ha='right', va='center', fontsize=12,
                color=color, fontweight='bold')

        # Model name and Elo rating on the right with rank change indicator
        diff = pr - er
        if diff != 0:
            diff_str = f" ({diff:+d})"
        else:
            diff_str = ""
        ax.text(right_x + 0.05, er, f"{model} ({elo_rating}){diff_str}", ha='left', va='center', fontsize=12,
                color=color, fontweight='bold')

    # Axis styling
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(len(rankings) + 0.5, -0.8)  # Extended top for headers

    # Column headers (positioned above rank 1)
    ax.text(left_x, -0.4, 'Peer Rank\n(Mean Score)', ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(right_x, -0.4, 'Elo Rank\n(Pairwise)', ha='center', va='top', fontsize=16, fontweight='bold')

    # Remove spines and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add correlation stats (top center, between headers and rank 1)
    corr_text = f"Pearson r = {pearson_r:.3f}  |  Spearman ρ = {spearman_r:.3f}"
    ax.text(0.5, 0.5, corr_text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.95, zorder=10))

    ax.set_title('Elo Ranking vs Peer Ranking',
                 fontweight='bold', fontsize=18, pad=20)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig18_elo_vs_peer')


def generate_fig16_judge_agreement_matrix(data: dict, output_dir: Path):
    """Figure 16: Judge Agreement Matrix - pairwise correlation heatmap (DISCUSSION)."""

    print("\nGenerating Figure 16: Judge Agreement Matrix...")

    evaluations = data['phase3'].get('evaluations', {})
    if not evaluations:
        print("  Skipping: No evaluation data available")
        return

    # Use shared function from config.py
    agreement = calculate_judge_agreement(evaluations)
    if not agreement or not agreement['judges']:
        print("  Skipping: Could not calculate judge agreement")
        return

    judges = agreement['judges']
    matrix = agreement['matrix']
    n = len(judges)

    # Build numpy matrix (NaN on diagonal for display)
    corr_matrix = np.zeros((n, n))
    for i, j1 in enumerate(judges):
        for j, j2 in enumerate(judges):
            if j1 == j2:
                corr_matrix[i, j] = np.nan  # Diagonal shows "—"
            elif j2 in matrix.get(j1, {}):
                corr_matrix[i, j] = matrix[j1][j2]
            elif j1 in matrix.get(j2, {}):
                corr_matrix[i, j] = matrix[j2][j1]
            else:
                corr_matrix[i, j] = np.nan

    # Calculate stats for subtitle
    valid_pairs = [p for p in agreement['pairs'] if p[0] != p[1]]
    avg_corr = np.mean([p[2] for p in valid_pairs]) if valid_pairs else 0

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create custom annotation array with "—" for diagonal
    annot_array = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot_array[i, j] = "—"
            elif np.isnan(corr_matrix[i, j]):
                annot_array[i, j] = ""
            else:
                annot_array[i, j] = f"{corr_matrix[i, j]:.2f}"

    # Create heatmap with 0-1 scale (correlations are positive)
    sns.heatmap(corr_matrix, annot=annot_array, fmt='', cmap='RdYlGn',
                xticklabels=judges, yticklabels=judges,
                vmin=0, vmax=1,
                annot_kws={'fontsize': 11},
                cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
                linewidths=0.5, linecolor='white',
                ax=ax)

    # Highlight diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor='black', linewidth=2))

    ax.set_xlabel('Judge B', fontweight='bold', fontsize=12)
    ax.set_ylabel('Judge A', fontweight='bold', fontsize=12)
    ax.set_title(f'Judge Agreement Matrix\n(Avg agreement: r = {avg_corr:.3f}, n = {len(valid_pairs)} pairs)',
                 fontweight='bold', fontsize=14, pad=10)

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    # Increase colorbar font size
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Pearson r', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig16_judge_agreement_matrix')


def generate_fig6_question_autopsy(data: dict, output_dir: Path):
    """Figure 6: Question Autopsy - Difficulty vs Controversy scatter (RESULTS)."""

    print("\nGenerating Figure 6: Question Autopsy...")

    evaluations = data['phase3'].get('evaluations', {})

    # Flatten questions_by_model into a list
    questions = []
    for model, model_qs in data.get('phase1', {}).get('questions_by_model', {}).items():
        for q in model_qs:
            questions.append({
                "question": q.get("question", ""),
                "category": q.get("category", ""),
                "source_model": model,
                "id": q.get("question", "")[:50],
            })

    if not evaluations:
        print("  Skipping: No evaluation data available")
        return

    # Calculate question stats
    q_stats = calculate_question_stats(evaluations, questions)
    if not q_stats or not q_stats['questions']:
        print("  Skipping: Could not calculate question stats")
        return

    # Extract data for scatter plot
    q_data = list(q_stats['questions'].values())
    avgs = [q['avg'] for q in q_data]
    stds = [q['std'] for q in q_data]
    categories = [q.get('category', 'unknown') for q in q_data]

    # Create category color mapping
    unique_cats = list(set(categories))
    cat_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_cats)))
    cat_color_map = {cat: cat_colors[i] for i, cat in enumerate(unique_cats)}
    colors = [cat_color_map[c] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Main scatter plot
    scatter = ax.scatter(avgs, stds, c=colors, s=80, alpha=0.7,
                         edgecolors='white', linewidths=0.5)

    # Highlight extremes with markers (no emojis - Times New Roman doesn't support them)
    hardest = q_stats['hardest'][:3]
    controversial = q_stats['controversial'][:3]

    for q_id, qs in hardest:
        ax.scatter(qs['avg'], qs['std'], s=200, marker='v', c='#CC3311',
                  edgecolors='black', linewidths=1.5, zorder=5, label='_nolegend_')

    for q_id, qs in controversial:
        ax.scatter(qs['avg'], qs['std'], s=200, marker='X', c='#EE7733',
                  edgecolors='black', linewidths=1.5, zorder=5, label='_nolegend_')

    # Add quadrant lines at medians
    median_avg = np.median(avgs)
    median_std = np.median(stds)
    ax.axvline(median_avg, color='gray', linestyle=':', alpha=0.5, zorder=1)
    ax.axhline(median_std, color='gray', linestyle=':', alpha=0.5, zorder=1)

    # Quadrant labels
    ax.text(0.02, 0.98, 'Hard & Controversial', transform=ax.transAxes,
            fontsize=13, va='top', ha='left', color='#CC3311', alpha=0.8,
            fontweight='bold')
    ax.text(0.98, 0.98, 'Easy & Controversial', transform=ax.transAxes,
            fontsize=13, va='top', ha='right', color='#EE7733', alpha=0.8)
    ax.text(0.02, 0.02, 'Hard & Consensus', transform=ax.transAxes,
            fontsize=13, va='bottom', ha='left', color='#0077BB', alpha=0.8)
    ax.text(0.98, 0.02, 'Easy & Consensus', transform=ax.transAxes,
            fontsize=13, va='bottom', ha='right', color='#009988', alpha=0.8,
            fontweight='bold')

    # Legend for categories
    legend_handles = [mpatches.Patch(color=cat_color_map[cat], label=cat[:20])
                     for cat in unique_cats]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=12,
             title='Category', title_fontsize=13)

    ax.set_xlabel('Difficulty (avg score, lower = harder)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Controversy (std, higher = more disagreement)', fontweight='bold', fontsize=14)
    ax.set_title(f'Question Autopsy: Difficulty vs Controversy (n = {len(q_data)})\n'
                 r'($\blacktriangledown$ = hardest, $\times$ = controversial)',
                 fontweight='bold', fontsize=16, pad=10)
    ax.tick_params(axis='both', labelsize=12)

    # Set axis limits with padding
    ax.set_xlim(min(avgs) - 0.5, max(avgs) + 0.5)
    ax.set_ylim(0, max(stds) + 0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig6_question_autopsy')


def generate_fig11_self_bias(data: dict, output_dir: Path):
    """Figure 11: Self Bias - tendency to rate own responses higher, with 95% CI."""

    print("\nGenerating Figure 11: Self Bias...")

    name_data = calculate_name_bias(data)
    if not name_data:
        print("  Skipping: No multi-mode data available")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    by_self = sorted(name_data, key=lambda x: x['self_bias'], reverse=True)
    models = [d['model'] for d in by_self]
    biases = [d['self_bias'] for d in by_self]
    errors = [d['self_bias_se'] * 1.96 for d in by_self]  # 95% CI
    colors = ['#EE7733' if b > 0 else '#0077BB' for b in biases]

    y_pos = np.arange(len(models))
    ax.barh(y_pos, biases, color=colors, edgecolor='white', alpha=0.85,
            xerr=errors, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Self − Peer Score')
    ax.set_title('Self Bias with 95% CI\n(positive = overrates own responses)', fontweight='bold', pad=10)
    ax.invert_yaxis()

    # Extend x-axis to make room for labels and error bars
    max_extent = max(abs(b) + e for b, e in zip(biases, errors))
    ax.set_xlim(-max_extent - 0.15, max_extent + 0.15)

    # Add value labels at outer end of bars (beyond error bars)
    for i, (bias, err) in enumerate(zip(biases, errors)):
        if bias >= 0:
            ax.text(bias + err + 0.03, i, f'{bias:+.2f}', va='center', ha='left', fontsize=9)
        else:
            ax.text(bias - err - 0.03, i, f'{bias:+.2f}', va='center', ha='right', fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig11_self_bias')


def generate_fig12_name_bias(data: dict, output_dir: Path):
    """Figure 12: Name Bias - effect of revealing model identity, with 95% CI."""

    print("\nGenerating Figure 12: Name Bias...")

    name_data = calculate_name_bias(data)
    if not name_data:
        print("  Skipping: No multi-mode data available")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    by_name = sorted(name_data, key=lambda x: x['name_bias'], reverse=True)
    models = [d['model'] for d in by_name]
    biases = [d['name_bias'] for d in by_name]
    errors = [d['name_bias_se'] * 1.96 for d in by_name]  # 95% CI
    colors = ['#009988' if b > 0 else '#CC3311' for b in biases]

    y_pos = np.arange(len(models))
    ax.barh(y_pos, biases, color=colors, edgecolor='white', alpha=0.85,
            xerr=errors, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Shuffle − Peer Score')
    ax.set_title('Name Bias with 95% CI\n(positive = brand recognition helped)', fontweight='bold', pad=10)
    ax.invert_yaxis()

    # Extend x-axis to make room for labels and error bars
    max_extent = max(abs(b) + e for b, e in zip(biases, errors))
    ax.set_xlim(-max_extent - 0.15, max_extent + 0.15)

    # Add value labels at outer end of bars (beyond error bars)
    for i, (bias, err) in enumerate(zip(biases, errors)):
        if bias >= 0:
            ax.text(bias + err + 0.03, i, f'{bias:+.2f}', va='center', ha='left', fontsize=9)
        else:
            ax.text(bias - err - 0.03, i, f'{bias:+.2f}', va='center', ha='right', fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig12_name_bias')


def generate_fig13_position_bias(data: dict, output_dir: Path):
    """Figure 13: Position Bias - effect of presentation order with 95% CI."""

    print("\nGenerating Figure 13: Position Bias...")

    pos_data = calculate_position_bias(data)
    if not pos_data:
        print("  Skipping: No multi-mode data available")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    positions = [d['position'] for d in pos_data]
    biases = [d['pos_bias'] for d in pos_data]
    errors = [d['bias_se'] * 1.96 for d in pos_data]  # 95% CI
    colors = ['#009988' if b > 0 else '#CC3311' for b in biases]

    x_pos = np.arange(len(positions))
    ax.bar(x_pos, biases, color=colors, edgecolor='white', alpha=0.85,
           yerr=errors, capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'#{p}' for p in positions])
    ax.set_xlabel('Presentation Position')
    ax.set_ylabel('Blind − Peer Score')
    ax.set_title('Position Bias with 95% CI\n(positive = position helped)', fontweight='bold', pad=10)

    for i, (bias, err) in enumerate(zip(biases, errors)):
        va = 'bottom' if bias >= 0 else 'top'
        offset = err + 0.03 if bias >= 0 else -err - 0.03
        ax.text(i, bias + offset, f'{bias:+.2f}', ha='center', va=va, fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig13_position_bias')


# =============================================================================
# Main
# =============================================================================

def find_latest_revision(data_dir: Path):
    """Find the most recent revision based on file modification time."""

    phase3_files = list(data_dir.glob("phase3_rankings_*.json"))
    if not phase3_files:
        return None

    latest = max(phase3_files, key=lambda p: p.stat().st_mtime)
    # Extract revision from filename
    return latest.stem.replace("phase3_rankings_", "")


def main():
    parser = argparse.ArgumentParser(description='Generate publication figures from PeerRank data')
    parser.add_argument('--revision', '-r', type=str, help='Data revision (e.g., v11)')
    parser.add_argument('--output', '-o', type=str, default='figures', help='Output directory')
    parser.add_argument('--data-dir', '-d', type=str, default='data', help='Data directory')
    args = parser.parse_args()

    # Apply style
    plt.rcParams.update(STYLE_CONFIG)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)

    # Find revision
    revision = args.revision or find_latest_revision(data_dir)
    if not revision:
        print("Error: No data files found. Run peerrank.py first.")
        return 1

    print(f"PeerRank Figure Generator")
    print(f"=" * 40)
    print(f"Revision: {revision}")
    print(f"Data dir: {data_dir}")
    print(f"Output:   {output_dir}")

    # Load data
    data = load_data(revision, data_dir)

    if 'phase3' not in data:
        print(f"Error: phase3_rankings_{revision}.json not found")
        return 1

    print(f"\nLoaded: {len(data)} phase files")

    # ==========================================================================
    # FIGURE ORGANIZATION (matching article structure)
    # ==========================================================================
    # Fig 1: Pipeline diagram (user-created, not generated here)
    #
    # RESULTS SECTION (Figures 4-7, 18)
    # Fig 4: Main rankings - the primary finding
    # Fig 5: Cross-evaluation matrix - who rated whom
    # Fig 6: Question autopsy - difficulty vs controversy
    # Fig 7: Peer score vs time - efficiency trade-off
    # Fig 18: Elo vs Peer ranking - slope graph comparison
    #
    # DISCUSSION SECTION (Figures 11-17) - Bias Analysis
    # Fig 11: Self bias - tendency to overrate own responses
    # Fig 12: Name bias - effect of revealing model identity
    # Fig 13: Position bias - effect of presentation order
    # Fig 14: Judge generosity - how lenient/strict each model judges
    # Fig 15: Judge generosity vs peer ranking - correlation analysis
    # Fig 16: Judge agreement matrix - pairwise correlation heatmap
    # Fig 17: Radar chart - multi-dimensional summary
    # ==========================================================================

    print("\n--- RESULTS SECTION ---")
    generate_fig4_peer_rankings(data, output_dir)
    generate_fig5_cross_eval_heatmap(data, output_dir)
    generate_fig6_question_autopsy(data, output_dir)
    generate_fig7_peer_score_vs_time(data, output_dir)
    generate_fig18_elo_vs_peer(data, output_dir)

    # Bias analysis figures (require multi-mode data)
    if has_multimode_data(data):
        print("\n--- DISCUSSION SECTION (Bias Analysis) ---")
        generate_fig11_self_bias(data, output_dir)
        generate_fig12_name_bias(data, output_dir)
        generate_fig13_position_bias(data, output_dir)
        generate_fig14_judge_generosity(data, output_dir)
        generate_fig15_judge_generosity_vs_peer(data, output_dir)
        generate_fig16_judge_agreement_matrix(data, output_dir)
        generate_fig17_radar_chart(data, output_dir)
    else:
        print("\n[Note: Figures 11-17 skipped - requires multi-mode bias data]")
        print("  Run Phase 3 with all 3 modes to generate bias analysis figures.")

    print(f"\n{'=' * 40}")
    print(f"Done! Figures saved to: {output_dir}/")

    # Provider clustering statistical test
    clustering_result = compute_provider_clustering(data)
    if clustering_result:
        print(f"\n{clustering_result}")

    # Generate LaTeX file with figure templates
    generate_latex_templates(output_dir, has_multimode_data(data))

    return 0


def compute_provider_clustering(data: dict) -> str:
    """Compute Kruskal-Wallis test for peer scores grouped by provider."""

    # Provider mapping from model names
    PROVIDER_MAP = {
        'gpt-5.2': 'OpenAI', 'gpt-5-mini': 'OpenAI',
        'claude-opus-4-5': 'Anthropic', 'claude-sonnet-4-5': 'Anthropic',
        'gemini-3-pro-preview': 'Google', 'gemini-3-flash-thinking': 'Google',
        'gemini-2.5-pro': 'Google', 'gemini-2.5-flash': 'Google',
        'grok-4-1-fast': 'xAI',
        'deepseek-chat': 'DeepSeek',
        'llama-4-maverick': 'Meta',
        'sonar-pro': 'Perplexity',
        'kimi-k2-0905': 'Moonshot',
        'mistral-large': 'Mistral',
    }

    evaluations = data['phase3'].get('evaluations', {})
    models = list(evaluations.keys())
    scores = calculate_scores_from_evaluations(evaluations, models)

    # Group peer scores by provider
    provider_scores = {}
    for model in models:
        provider = PROVIDER_MAP.get(model, 'Other')
        peer = scores['peer_scores'].get(model, [])
        if peer:
            if provider not in provider_scores:
                provider_scores[provider] = []
            provider_scores[provider].extend(peer)

    # Need at least 2 providers with data
    groups = [v for v in provider_scores.values() if len(v) >= 2]
    if len(groups) < 2:
        return None

    # Kruskal-Wallis H-test (non-parametric ANOVA)
    h_stat, p_value = stats.kruskal(*groups)

    # Effect size: eta-squared approximation = H / (N - 1)
    n_total = sum(len(g) for g in groups)
    eta_sq = h_stat / (n_total - 1) if n_total > 1 else 0

    # Format result
    n_providers = len(groups)
    sig = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"

    return f"Provider clustering: Kruskal-Wallis H({n_providers-1}) = {h_stat:.2f}, {sig}, eta^2 = {eta_sq:.3f}"


def generate_latex_templates(output_dir: Path, has_bias_figs: bool):
    """Generate LaTeX file with figure templates and captions."""

    latex_content = r"""%% PeerRank Figures - LaTeX Templates
%% Auto-generated by generate_figures.py
%% Copy these into your Overleaf document
%%
%% FIGURE ORGANIZATION:
%% - Fig 1: Pipeline diagram (user-created, in Methodology)
%% - Figs 4-7, 18: Results section
%% - Figs 11-17: Discussion section (Bias Analysis)

\usepackage{graphicx}
\usepackage{subcaption}  % For subfigures

%% ============================================================================
%% METHODOLOGY SECTION
%% ============================================================================
%% Figure 1: Pipeline diagram (user-created, not auto-generated)
%% \begin{figure}[htbp]
%%     \centering
%%     \includegraphics[width=\linewidth]{figures/fig1_pipeline.pdf}
%%     \caption{Fully endogenous evaluation pipeline...}
%%     \label{fig:pipeline}
%% \end{figure}

%% ============================================================================
%% RESULTS SECTION
%% ============================================================================

%% FIGURE 4: Peer Rankings (Main Result)
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig4_peer_rankings.pdf}
    \caption{Peer evaluation rankings of seven large language models. Bars represent mean peer scores (excluding self-ratings) on a 1--10 scale, with error bars indicating standard deviation. Models are ranked by peer consensus.}
    \label{fig:peer-rankings}
\end{figure}

%% FIGURE 5: Cross-Evaluation Heatmap
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\linewidth]{figures/fig5_cross_eval_heatmap.pdf}
    \caption{Cross-evaluation matrix showing average scores assigned by each evaluator model (rows) to each evaluated model (columns). Diagonal cells (highlighted) represent self-ratings. Color scale ranges from red (low scores) to green (high scores), centered at 7.5.}
    \label{fig:cross-eval}
\end{figure}

%% FIGURE 6: Question Autopsy
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig6_question_autopsy.pdf}
    \caption{Question difficulty versus controversy scatter plot. Each point represents a question, with x-axis showing average score (lower = harder) and y-axis showing score standard deviation (higher = more judge disagreement). Points are colored by question category.}
    \label{fig:question-autopsy}
\end{figure}

%% FIGURE 7: Peer Score vs Speed Trade-off
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig7_peer_score_vs_time.pdf}
    \caption{Peer score versus response time trade-off across evaluated models. The upper-left quadrant represents optimal performance (high peer score, low latency). Dashed lines indicate median values for each axis.}
    \label{fig:peer-score-speed}
\end{figure}

%% FIGURE 18: Elo vs Peer Ranking
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig18_elo_vs_peer.pdf}
    \caption{Comparison of Elo-based ranking (computed from pairwise comparisons) and mean peer score ranking. Lines connect each model's position in both ranking systems; steeper slopes indicate larger rank changes. Correlation statistics shown at bottom.}
    \label{fig:elo-vs-peer}
\end{figure}

"""

    if has_bias_figs:
        latex_content += r"""
%% ============================================================================
%% DISCUSSION SECTION - Bias Analysis
%% ============================================================================

%% FIGURE 11: Self Bias
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig11_self_bias.pdf}
    \caption{Self bias across models. Bars show the difference between self-ratings and peer ratings (Self $-$ Peer). Positive values (orange) indicate models that overrate their own responses; negative values (blue) indicate models that underrate themselves.}
    \label{fig:self-bias}
\end{figure}

%% FIGURE 12: Name Bias
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig12_name_bias.pdf}
    \caption{Name bias across models. Bars show score change when model identity is revealed (Shuffle $-$ Peer). Positive values (green) indicate brand recognition helped; negative values (red) indicate name hurt scores.}
    \label{fig:name-bias}
\end{figure}

%% FIGURE 13: Position Bias
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig13_position_bias.pdf}
    \caption{Position bias by presentation order. Bars show score change due to fixed position in blind evaluation (Blind $-$ Peer). Positive values indicate the position helped; negative values indicate the position hurt scores.}
    \label{fig:position-bias}
\end{figure}

%% FIGURE 14: Judge Generosity
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig14_judge_generosity.pdf}
    \caption{Judge generosity across models. Bars show average score given by each model when evaluating peers, with error bars indicating standard deviation. Higher values indicate more lenient judging; the dashed line marks the overall mean.}
    \label{fig:judge-generosity}
\end{figure}

%% FIGURE 15: Judge Generosity vs Peer Ranking
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig15_judge_generosity_vs_peer.pdf}
    \caption{Relationship between peer ranking and judge generosity. Each point represents a model (numbered by rank), showing its performance (peer score) on the x-axis and its judging leniency (average score given) on the y-axis.}
    \label{fig:judge-vs-peer}
\end{figure}

%% FIGURE 16: Judge Agreement Matrix
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\linewidth]{figures/fig16_judge_agreement_matrix.pdf}
    \caption{Judge agreement matrix showing pairwise Pearson correlation between models' scoring patterns. Higher values (green) indicate judges that rate responses similarly; lower values (red) indicate divergent evaluation criteria.}
    \label{fig:judge-agreement}
\end{figure}

%% FIGURE 17: Radar Chart (multi-dimensional summary)
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\linewidth]{figures/fig17_radar_chart.pdf}
    \caption{Multi-dimensional model comparison across five normalized metrics: Peer Score, Consistency (inverse of score variance), Speed (inverse of response time), Humility (inverse of self-bias), and Strictness (inverse of judge generosity). All metrics scaled to 0--1.}
    \label{fig:radar-chart}
\end{figure}

"""

    latex_content += r"""
%% ============================================================================
%% CROSS-REFERENCE GUIDE
%% ============================================================================
%% Usage: Figure~\ref{fig:peer-rankings} shows...
%%
%% Label                 | Figure | Section
%% ----------------------|--------|------------
%% fig:pipeline          | 1      | Methodology
%% fig:peer-rankings     | 4      | Results
%% fig:cross-eval        | 5      | Results
%% fig:question-autopsy  | 6      | Results
%% fig:peer-score-speed  | 7      | Results
%% fig:elo-vs-peer       | 18     | Results
%% fig:self-bias         | 11     | Discussion
%% fig:name-bias         | 12     | Discussion
%% fig:position-bias     | 13     | Discussion
%% fig:judge-generosity  | 14     | Discussion
%% fig:judge-vs-peer     | 15     | Discussion
%% fig:judge-agreement   | 16     | Discussion
%% fig:radar-chart       | 17     | Discussion
"""

    # Write LaTeX file
    latex_path = output_dir / "figures_latex.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"\nLaTeX templates saved to: {latex_path}")
    print("  Copy the figure blocks into your Overleaf document.")


if __name__ == '__main__':
    exit(main())
