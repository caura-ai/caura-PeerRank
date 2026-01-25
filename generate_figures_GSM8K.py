"""
GSM8K Validation Figure Generator

Generates publication-quality figures from GSM8K validation data:
- Fig 20: Correlation scatter (peer scores vs ground truth math accuracy)
- Fig 21: Rank agreement visualization
- Fig 22: Score comparison bar chart
- Statistical analysis reports

Output: PDF (vector) + PNG (600 DPI) for Overleaf/LaTeX compatibility.

Usage:
    python generate_figures_GSM8K.py                    # Generate all figures
    python generate_figures_GSM8K.py --output figures   # Custom output dir
    python generate_figures_GSM8K.py --stats-only       # Print stats without figures
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from peerrank.config import DATA_DIR, calculate_scores_from_evaluations, MODELS

# =============================================================================
# Configuration
# =============================================================================

GSM8K_DIR = DATA_DIR / "GSM8K"
VALIDATION_REVISION = "GSM8K"

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
    'gpt-5.2': '#0047AB',              # Cobalt Blue (darker)
    'gpt-5-mini': '#56B4E9',
    'claude-opus-4-5': '#029E73',
    'claude-sonnet-4-5': '#78C679',
    'gemini-3-pro-preview': '#D55E00',
    'gemini-3-flash-preview': '#F0E442',
    'grok-4-1-fast': '#CC79A7',
    'deepseek-chat': '#E69F00',
    'llama-4-maverick': '#999999',
    'sonar-pro': '#9467BD',
    'kimi-k2-0905': '#8C564B',
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


# =============================================================================
# Data Loading
# =============================================================================

def load_gsm8k_data() -> dict:
    """Load all GSM8K validation data."""
    data = {}

    # GSM8K analysis (correlation results)
    analysis_path = GSM8K_DIR / f"GSM8K_analysis_{VALIDATION_REVISION}.json"
    if analysis_path.exists():
        with open(analysis_path, encoding='utf-8') as f:
            data['analysis'] = json.load(f)

    # Phase 4 ground truth scores
    phase4_path = GSM8K_DIR / f"phase4_GSM8K_scores_{VALIDATION_REVISION}.json"
    if phase4_path.exists():
        with open(phase4_path, encoding='utf-8') as f:
            data['phase4'] = json.load(f)

    # Phase 3 peer evaluations
    phase3_path = GSM8K_DIR / f"phase3_rankings_{VALIDATION_REVISION}.json"
    if phase3_path.exists():
        with open(phase3_path, encoding='utf-8') as f:
            data['phase3'] = json.load(f)

    # Phase 2 answers (for timing data)
    phase2_path = GSM8K_DIR / f"phase2_answers_{VALIDATION_REVISION}.json"
    if phase2_path.exists():
        with open(phase2_path, encoding='utf-8') as f:
            data['phase2'] = json.load(f)

    return data


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


# =============================================================================
# Figure Generation
# =============================================================================

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


def generate_fig11_correlation_scatter(data: dict, output_dir: Path):
    """Figure 20: Correlation scatter plot - Peer Score vs Ground Truth Math Accuracy."""
    print("\nGenerating Figure 20: Peer vs Truth Correlation...")

    comparison = get_comparison_data(data)
    correlation = get_correlation_stats(data)

    if not comparison:
        print("  Skipping: No comparison data available")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))

    peer_scores = [c['peer_score'] for c in comparison]
    truth_scores = [c['truth_score'] for c in comparison]
    models = [c['model'] for c in comparison]

    # Scatter plot with model colors
    for i, c in enumerate(comparison):
        color = get_color(c['model'])
        ax.scatter(c['truth_score'], c['peer_score'], s=200, c=color,
                   edgecolor='white', linewidth=1.5, zorder=3, alpha=0.9)

        # Add rank number inside point
        text_color = get_text_color_for_background(color)
        ax.text(c['truth_score'], c['peer_score'], str(c['peer_rank']),
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=text_color, zorder=4)

    # Add model labels
    for c in comparison:
        offset_x = 0.05
        offset_y = 0.04
        ax.annotate(c['model'],
                    xy=(c['truth_score'], c['peer_score']),
                    xytext=(c['truth_score'] + offset_x, c['peer_score'] + offset_y),
                    fontsize=8, ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='gray', alpha=0.8),
                    zorder=5)

    # Regression line
    if len(peer_scores) > 2:
        z = np.polyfit(truth_scores, peer_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(truth_scores) - 0.2, max(truth_scores) + 0.2, 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2,
                label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')

    # Perfect correlation reference line
    min_val = min(min(peer_scores), min(truth_scores)) - 0.2
    max_val = max(max(peer_scores), max(truth_scores)) + 0.2
    ax.plot([min_val, max_val], [min_val, max_val], 'g:', alpha=0.4,
            linewidth=1.5, label='Perfect correlation')

    # Correlation stats box
    if correlation:
        r = correlation.get('pearson_r', 0)
        p_val = correlation.get('pearson_p', 1)
        rho = correlation.get('spearman_r', 0)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        interp = "Strong" if abs(r) >= 0.7 else "Moderate" if abs(r) >= 0.5 else "Weak"

        textstr = f'Pearson r = {r:.3f}{sig}\nSpearman \u03c1 = {rho:.3f}\np = {p_val:.4f}\n({interp} correlation)'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    ax.set_xlabel('Ground Truth Score (GSM8K Math Accuracy \xd7 10)', fontweight='bold')
    ax.set_ylabel('Peer Score (Mean Peer Evaluation)', fontweight='bold')
    ax.set_title('Peer Evaluation vs Ground Truth Accuracy\nGSM8K Math Validation',
                 fontweight='bold', pad=10)

    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(min(truth_scores) - 0.3, max(truth_scores) + 0.3)
    ax.set_ylim(min(peer_scores) - 0.3, max(peer_scores) + 0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig11_gsm8k_correlation')


def generate_fig101_score_comparison(data: dict, output_dir: Path):
    """Figure 22: Side-by-side bar chart comparing Peer vs Truth scores."""
    print("\nGenerating Figure 22: Score Comparison Bars...")

    comparison = get_comparison_data(data)
    if not comparison:
        print("  Skipping: No comparison data available")
        return

    # Sort by peer score
    comparison = sorted(comparison, key=lambda x: -x['peer_score'])

    fig, ax = plt.subplots(figsize=(10, 5))

    models = [c['model'] for c in comparison]
    peer_scores = [c['peer_score'] for c in comparison]
    truth_scores = [c['truth_score'] for c in comparison]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, peer_scores, width, label='Peer Score',
                   color='#0173B2', edgecolor='white', alpha=0.85)
    bars2 = ax.bar(x + width/2, truth_scores, width, label='Truth Score',
                   color='#029E73', edgecolor='white', alpha=0.85)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model (sorted by Peer Score)', fontweight='bold')
    ax.set_ylabel('Score (0-10 scale)', fontweight='bold')
    ax.set_title('Peer Score vs Ground Truth Score by Model\nGSM8K Math Validation', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(min(min(peer_scores), min(truth_scores)) - 0.5, 10.2)
    ax.legend(loc='upper right')

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig101_gsm8k_score_comparison')


def generate_fig100_rank_agreement(data: dict, output_dir: Path):
    """Figure 21: Rank agreement visualization - peer rank vs truth rank."""
    print("\nGenerating Figure 21: Rank Agreement...")

    comparison = get_comparison_data(data)
    if not comparison:
        print("  Skipping: No comparison data available")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    n = len(comparison)
    comparison = sorted(comparison, key=lambda x: x['peer_rank'])

    for c in comparison:
        color = get_color(c['model'])
        peer_rank = c['peer_rank']
        truth_rank = c.get('truth_rank', c['peer_rank'])

        # Draw connecting line
        ax.plot([0, 1], [peer_rank, truth_rank], 'k-', alpha=0.3, linewidth=1)

        # Peer rank point (left)
        ax.scatter(0, peer_rank, s=200, c=color, edgecolor='white',
                   linewidth=1.5, zorder=3)
        ax.text(-0.08, peer_rank, c['model'], ha='right', va='center', fontsize=12)

        # Truth rank point (right)
        ax.scatter(1, truth_rank, s=200, c=color, edgecolor='white',
                   linewidth=1.5, zorder=3)

        # Rank change indicator
        diff = peer_rank - truth_rank
        if abs(diff) > 0.5:
            arrow_color = '#029E73' if diff > 0 else '#D55E00'  # Green if improved, orange if worse
            ax.annotate('', xy=(1, truth_rank), xytext=(0, peer_rank),
                        arrowprops=dict(arrowstyle='->', color=arrow_color,
                                        alpha=0.6, lw=1.5))

    ax.set_xlim(-0.5, 1.3)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Peer Rank', 'Truth Rank'], fontweight='bold', fontsize=12)
    ax.set_ylabel('Rank (1 = Best)', fontweight='bold', fontsize=12)
    ax.set_title('Rank Agreement: Peer vs Ground Truth\nGSM8K Math Validation (lower = better)',
                 fontweight='bold', fontsize=14, pad=10)
    ax.tick_params(axis='y', labelsize=11)

    # Add perfect agreement line
    ax.axhline(y=(n + 1) / 2, color='gray', linestyle=':', alpha=0.3)

    # Add legend for rank change
    green_patch = mpatches.Patch(color='#029E73', alpha=0.6, label='Underranked by peers')
    orange_patch = mpatches.Patch(color='#D55E00', alpha=0.6, label='Overranked by peers')
    ax.legend(handles=[green_patch, orange_patch], loc='lower right', fontsize=10)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig100_gsm8k_rank_agreement')


# =============================================================================
# Statistical Report Generation
# =============================================================================

def generate_stats_report(data: dict, output_dir: Path):
    """Generate comprehensive statistical analysis report."""
    print("\nGenerating Statistical Report...")

    comparison = get_comparison_data(data)
    correlation = get_correlation_stats(data)
    phase4 = data.get('phase4', {})

    if not comparison:
        print("  Skipping: No data available")
        return

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("GSM8K VALIDATION - STATISTICAL ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Revision:  {VALIDATION_REVISION}")
    report_lines.append(f"Models:    {len(comparison)}")

    if phase4.get('summary'):
        total_q = list(phase4['summary'].values())[0].get('total', 0)
        report_lines.append(f"Questions: {total_q}")
    report_lines.append("")

    # Correlation Analysis
    report_lines.append("-" * 70)
    report_lines.append("1. CORRELATION ANALYSIS")
    report_lines.append("-" * 70)

    if correlation:
        r = correlation.get('pearson_r', 0)
        p = correlation.get('pearson_p', 1)
        rho = correlation.get('spearman_r', 0)
        rho_p = correlation.get('spearman_p', 1)

        report_lines.append(f"  Pearson r:     {r:>7.4f}  (p = {p:.4f})")
        report_lines.append(f"  Spearman rho:  {rho:>7.4f}  (p = {rho_p:.4f})")
        report_lines.append("")

        # Interpretation
        if r >= 0.8:
            interp = "STRONG positive correlation"
        elif r >= 0.6:
            interp = "MODERATE positive correlation"
        elif r >= 0.4:
            interp = "WEAK positive correlation"
        elif r >= -0.4:
            interp = "NO significant correlation"
        else:
            interp = "NEGATIVE correlation"

        sig = "statistically significant (p < 0.05)" if p < 0.05 else "NOT statistically significant"
        report_lines.append(f"  Interpretation: {interp}")
        report_lines.append(f"  Significance:   {sig}")
        report_lines.append("")

        # R-squared
        r_sq = r ** 2
        report_lines.append(f"  R-squared: {r_sq:.4f}")
        report_lines.append(f"    -> {r_sq*100:.1f}% of peer score variance explained by math accuracy")
    else:
        report_lines.append("  [Correlation data not available - install scipy]")
    report_lines.append("")

    # Score Comparison Table
    report_lines.append("-" * 70)
    report_lines.append("2. SCORE COMPARISON TABLE")
    report_lines.append("-" * 70)
    report_lines.append("")
    report_lines.append(f"  {'Rank':<5} {'Model':<25} {'Peer':>7} {'Truth':>7} {'Diff':>7} {'T.Rank':>7}")
    report_lines.append(f"  {'-'*5} {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    comparison_sorted = sorted(comparison, key=lambda x: x['peer_rank'])
    for c in comparison_sorted:
        diff = c['peer_score'] - c['truth_score']
        tr = c.get('truth_rank', '-')
        tr_str = f"{tr:.1f}" if isinstance(tr, float) and tr != int(tr) else str(int(tr)) if isinstance(tr, (int, float)) else str(tr)
        report_lines.append(
            f"  {c['peer_rank']:<5} {c['model']:<25} {c['peer_score']:>7.2f} "
            f"{c['truth_score']:>7.2f} {diff:>+7.2f} {tr_str:>7}"
        )
    report_lines.append("")

    # Rank Agreement Analysis
    report_lines.append("-" * 70)
    report_lines.append("3. RANK AGREEMENT ANALYSIS")
    report_lines.append("-" * 70)

    rank_diffs = [abs(c['peer_rank'] - c.get('truth_rank', c['peer_rank'])) for c in comparison]
    exact_matches = sum(1 for d in rank_diffs if d < 0.5)
    close_matches = sum(1 for d in rank_diffs if d <= 1.5)

    report_lines.append(f"  Exact rank matches:    {exact_matches}/{len(comparison)} ({100*exact_matches/len(comparison):.0f}%)")
    report_lines.append(f"  Within 1 rank:         {close_matches}/{len(comparison)} ({100*close_matches/len(comparison):.0f}%)")
    report_lines.append(f"  Mean rank difference:  {np.mean(rank_diffs):.2f}")
    report_lines.append(f"  Max rank difference:   {max(rank_diffs):.1f}")
    report_lines.append("")

    # Most over/under-rated
    over_rated = max(comparison, key=lambda x: x['peer_score'] - x['truth_score'])
    under_rated = min(comparison, key=lambda x: x['peer_score'] - x['truth_score'])
    report_lines.append(f"  Most overrated:   {over_rated['model']} (+{over_rated['peer_score']-over_rated['truth_score']:.2f})")
    report_lines.append(f"  Most underrated:  {under_rated['model']} ({under_rated['peer_score']-under_rated['truth_score']:.2f})")
    report_lines.append("")

    # Ground Truth Accuracy Summary
    if phase4.get('summary'):
        report_lines.append("-" * 70)
        report_lines.append("4. GROUND TRUTH MATH ACCURACY SUMMARY")
        report_lines.append("-" * 70)
        report_lines.append("")

        summary = phase4['summary']
        ranked = sorted(summary.items(), key=lambda x: -x[1]['accuracy'])

        accs = [s['accuracy'] for _, s in ranked]
        report_lines.append(f"  Mean accuracy:   {np.mean(accs):.1f}%")
        report_lines.append(f"  Std deviation:   {np.std(accs):.1f}%")
        report_lines.append(f"  Range:           {min(accs):.0f}% - {max(accs):.0f}%")
        report_lines.append("")

        report_lines.append(f"  {'Model':<25} {'Accuracy':>10} {'Correct':>10}")
        report_lines.append(f"  {'-'*25} {'-'*10} {'-'*10}")
        for model, stats in ranked:
            report_lines.append(
                f"  {model:<25} {stats['accuracy']:>9.1f}% "
                f"{stats['correct']:>4}/{stats['total']}"
            )
        report_lines.append("")

    # Conclusion
    report_lines.append("-" * 70)
    report_lines.append("5. CONCLUSION")
    report_lines.append("-" * 70)

    if correlation:
        r = correlation.get('pearson_r', 0)
        p = correlation.get('pearson_p', 1)

        if r >= 0.7 and p < 0.05:
            conclusion = "VALIDATED: Peer evaluation strongly correlates with ground truth math accuracy."
        elif r >= 0.5 and p < 0.05:
            conclusion = "PARTIALLY VALIDATED: Moderate correlation between peer and truth scores."
        elif p >= 0.05:
            conclusion = "INCONCLUSIVE: Correlation not statistically significant."
        else:
            conclusion = "NOT VALIDATED: Weak or negative correlation with ground truth."

        report_lines.append(f"  {conclusion}")
        report_lines.append("")
        report_lines.append(f"  Peer evaluation explains {(r**2)*100:.1f}% of the variance in math accuracy,")
        report_lines.append(f"  suggesting that peer rankings {'can' if r >= 0.5 else 'may not'} serve as a proxy")
        report_lines.append(f"  for objective mathematical reasoning ability on GSM8K-style problems.")
    else:
        report_lines.append("  [Cannot compute correlation - scipy required]")

    report_lines.append("")
    report_lines.append("=" * 70)

    # Write report
    report_text = "\n".join(report_lines)

    # Print to console
    print("\n" + report_text)

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "GSM8K_stats_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  Saved: {report_path}")

    # Also save JSON summary
    json_summary = {
        'timestamp': datetime.now().isoformat(),
        'revision': VALIDATION_REVISION,
        'n_models': len(comparison),
        'correlation': correlation,
        'rank_agreement': {
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'mean_diff': round(np.mean(rank_diffs), 2),
            'max_diff': round(max(rank_diffs), 1),
        },
        'comparison': comparison,
    }

    json_path = output_dir / "GSM8K_stats_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2)
    print(f"  Saved: {json_path}")


# =============================================================================
# LaTeX Templates
# =============================================================================

def generate_latex_templates(output_dir: Path):
    """Generate LaTeX file with figure templates."""

    latex_content = r"""%% GSM8K Validation Figures - LaTeX Templates
%% Auto-generated by generate_figures_GSM8K.py
%% Copy these into your Overleaf document

\usepackage{graphicx}
\usepackage{subcaption}

%% ============================================================================
%% VALIDATION SECTION - GSM8K Math Analysis (Figures 20-22)
%% ============================================================================

%% FIGURE 20: Correlation Scatter Plot (Main Validation Result)
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig11_gsm8k_correlation.pdf}
    \caption{Correlation between peer evaluation scores and ground truth accuracy on GSM8K math problems. Each point represents a model, numbered by peer rank. The strong positive correlation validates that peer evaluation can assess mathematical reasoning ability.}
    \label{fig:gsm8k-correlation}
\end{figure}

%% FIGURE 21: Rank Agreement
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/fig100_gsm8k_rank_agreement.pdf}
    \caption{Rank agreement between peer evaluation and ground truth rankings on GSM8K. Lines connect each model's peer rank (left) to its truth rank (right). Green arrows indicate models underrated by peers; orange arrows indicate overrated models.}
    \label{fig:gsm8k-rank-agreement}
\end{figure}

%% FIGURE 22: Score Comparison Bars
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/fig101_gsm8k_score_comparison.pdf}
    \caption{Side-by-side comparison of peer evaluation scores (blue) and ground truth math accuracy scores (green) for each model on GSM8K. Both metrics use a 0--10 scale for direct comparison.}
    \label{fig:gsm8k-score-comparison}
\end{figure}

%% ============================================================================
%% CROSS-REFERENCE GUIDE
%% ============================================================================
%% Usage: Figure~\ref{fig:gsm8k-correlation} shows...
%%
%% Label                      | Figure | Description
%% ---------------------------|--------|---------------------------
%% fig:gsm8k-correlation      | 20     | Main correlation scatter
%% fig:gsm8k-rank-agreement   | 21     | Rank slope graph
%% fig:gsm8k-score-comparison | 22     | Peer vs truth bar chart
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    latex_path = output_dir / "GSM8K_figures_latex.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"\nLaTeX templates saved to: {latex_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication figures from GSM8K validation data'
    )
    parser.add_argument('--output', '-o', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('--stats-only', action='store_true',
                        help='Print statistics without generating figures')
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("GSM8K Validation Figure Generator")
    print("=" * 50)
    print(f"Data dir: {GSM8K_DIR}")
    print(f"Output:   {output_dir}")

    # Load data
    data = load_gsm8k_data()

    if not data:
        print(f"\nError: No GSM8K data found in {GSM8K_DIR}")
        print("Run validate_gsm8k.py first to generate data.")
        return 1

    print(f"\nLoaded: {len(data)} data files")

    # Apply matplotlib style
    plt.rcParams.update(STYLE_CONFIG)

    if args.stats_only:
        generate_stats_report(data, output_dir)
        return 0

    # Generate all figures
    print("\n--- GSM8K VALIDATION FIGURES (20-22) ---")
    generate_fig11_correlation_scatter(data, output_dir)
    generate_fig100_rank_agreement(data, output_dir)
    generate_fig101_score_comparison(data, output_dir)

    # Generate statistical report
    generate_stats_report(data, output_dir)

    # Generate LaTeX templates
    generate_latex_templates(output_dir)

    print(f"\n{'=' * 50}")
    print(f"Done! Figures saved to: {output_dir}/")

    return 0


if __name__ == '__main__':
    exit(main())
