#!/usr/bin/env python3
"""
Generate static HTML webpage from PeerRank Phase 4 report.
Reads markdown, parses tables, renders Jinja2 template to docs/index.html.
"""

import re
import os
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader


def parse_markdown_table(text: str, table_start: str) -> list[dict]:
    """Parse a markdown table following a header line containing table_start."""
    lines = text.split('\n')

    # Find the table start
    table_idx = None
    for i, line in enumerate(lines):
        if table_start in line:
            table_idx = i
            break

    if table_idx is None:
        return []

    # Find the actual table (skip empty lines, find header row with |)
    header_idx = None
    for i in range(table_idx, min(table_idx + 10, len(lines))):
        if '|' in lines[i] and not lines[i].strip().startswith('*'):
            header_idx = i
            break

    if header_idx is None:
        return []

    # Parse header
    header_line = lines[header_idx]
    headers = [h.strip() for h in header_line.split('|') if h.strip()]

    # Skip separator line (|---|---|)
    data_start = header_idx + 2

    # Parse rows until empty line or non-table line
    rows = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or not line.startswith('|'):
            break
        if line.startswith('|--') or line.startswith('| --'):
            continue

        cells = [c.strip() for c in line.split('|') if c.strip() or c == '']
        # Filter out empty strings from split edges
        cells = [c for c in cells if c or cells.index(c) not in [0, len(cells)-1]]

        if len(cells) >= len(headers):
            row = {headers[j]: cells[j] for j in range(len(headers))}
            rows.append(row)

    return rows


def parse_peer_rankings(text: str) -> list[dict]:
    """Parse the Final Peer Rankings table."""
    pattern = r'\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'

    # Find the section
    section_start = text.find('## Final Peer Rankings')
    if section_start == -1:
        return []

    section_end = text.find('\n## ', section_start + 1)
    section = text[section_start:section_end] if section_end != -1 else text[section_start:]

    matches = re.findall(pattern, section)
    results = []
    for m in matches:
        results.append({
            'rank': int(m[0]),
            'model': m[1].strip(),
            'peer_score': float(m[2]),
            'std': float(m[3]),
            'raw': float(m[4])
        })
    return results


def parse_elo_ratings(text: str) -> list[dict]:
    """Parse the Elo Ratings table."""
    # Pattern: | 1 | model | 1639 | 59.8% | 1720-959-1201 | 8.45 | 5 | +4 |
    pattern = r'\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|\s*([\d.]+)%\s*\|\s*([^|]+)\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|'

    section_start = text.find('## Elo Ratings')
    if section_start == -1:
        return []

    section_end = text.find('\n## ', section_start + 1)
    section = text[section_start:section_end] if section_end != -1 else text[section_start:]

    # Also get total matches
    total_matches_match = re.search(r'Total matches:\s*([\d,]+)', section)
    total_matches = total_matches_match.group(1) if total_matches_match else "0"

    matches = re.findall(pattern, section)
    results = []
    for m in matches:
        diff = m[7].strip()
        if diff == '—':
            diff = '0'
        results.append({
            'rank': int(m[0]),
            'model': m[1].strip(),
            'elo': int(m[2]),
            'win_pct': float(m[3]),
            'wlt': m[4].strip(),
            'peer': float(m[5]),
            'peer_rank': int(m[6]),
            'diff': diff
        })
    return results, total_matches


def parse_position_bias(text: str) -> list[dict]:
    """Parse the Position Bias table."""
    pattern = r'\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([+-]?[\d.]+)\s*\|'

    section_start = text.find('### Position Bias')
    if section_start == -1:
        return []

    section_end = text.find('\n### ', section_start + 1)
    if section_end == -1:
        section_end = text.find('\n## ', section_start + 1)
    section = text[section_start:section_end] if section_end != -1 else text[section_start:]

    matches = re.findall(pattern, section)
    results = []
    for m in matches:
        results.append({
            'position': int(m[0]),
            'blind_score': float(m[1]),
            'pos_bias': m[2]
        })
    return results


def parse_model_bias(text: str) -> list[dict]:
    """Parse the Model Bias table."""
    # Pattern: | 1 | model | 8.73 | 8.67 | -0.06 | 8.78 | +0.05 |
    pattern = r'\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([+-]?[\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([+-]?[\d.]+)\s*\|'

    section_start = text.find('### Model Bias')
    if section_start == -1:
        return []

    section_end = text.find('\n## ', section_start + 1)
    section = text[section_start:section_end] if section_end != -1 else text[section_start:]

    matches = re.findall(pattern, section)
    results = []
    for m in matches:
        results.append({
            'rank': int(m[0]),
            'model': m[1].strip(),
            'peer': float(m[2]),
            'self': float(m[3]),
            'self_bias': m[4],
            'shuffle': float(m[5]),
            'name_bias': m[6]
        })
    return results


def parse_judge_generosity(text: str) -> list[dict]:
    """Parse the Judge Generosity table."""
    pattern = r'\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|'

    section_start = text.find('## Judge Generosity')
    if section_start == -1:
        return []

    section_end = text.find('\n## ', section_start + 1)
    section = text[section_start:section_end] if section_end != -1 else text[section_start:]

    matches = re.findall(pattern, section)
    results = []
    for m in matches:
        results.append({
            'rank': int(m[0]),
            'model': m[1].strip(),
            'avg_given': float(m[2]),
            'std': float(m[3]),
            'count': int(m[4])
        })
    return results


def parse_question_autopsy(text: str, section_name: str) -> list[dict]:
    """Parse a question autopsy section (Hardest, Easiest, etc.)."""
    # Find section by emoji/name
    section_markers = {
        'hardest': '### 🔥 Hardest',
        'controversial': '### ⚔️ Most Controversial',
        'easiest': '### ✅ Easiest',
        'consensus': '### 🤝 Consensus'
    }

    marker = section_markers.get(section_name, '')
    section_start = text.find(marker)
    if section_start == -1:
        return []

    # Find next section
    section_end = text.find('\n### ', section_start + 1)
    if section_end == -1:
        section_end = text.find('\n## ', section_start + 1)
    section = text[section_start:section_end] if section_end != -1 else text[section_start:]

    # Parse table rows - format varies by section type
    if section_name in ['hardest', 'easiest']:
        # | Avg | Std | Category | Creator | Question |
        pattern = r'\|\s*([\d.]+)\s*\|\s*±([\d.]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|'
    else:
        # | Std | Avg | Category | Creator | Question |
        pattern = r'\|\s*±([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|'

    matches = re.findall(pattern, section)
    results = []
    for m in matches:
        if section_name in ['hardest', 'easiest']:
            results.append({
                'avg': float(m[0]),
                'std': float(m[1]),
                'category': m[2].strip(),
                'creator': m[3].strip(),
                'question': m[4].strip()
            })
        else:
            results.append({
                'std': float(m[0]),
                'avg': float(m[1]),
                'category': m[2].strip(),
                'creator': m[3].strip(),
                'question': m[4].strip()
            })
    return results


def parse_metadata(text: str) -> dict:
    """Parse header metadata from the report."""
    metadata = {
        'revision': 'v1',
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'models_count': 0,
        'questions_count': 0,
        'web_search': 'ON'
    }

    # Revision: **v1**
    rev_match = re.search(r'Revision:\s*\*\*([^*]+)\*\*', text)
    if rev_match:
        metadata['revision'] = rev_match.group(1)

    # Generated: 2026-01-19 22:02:45
    gen_match = re.search(r'Generated:\s*([\d-]+\s+[\d:]+)', text)
    if gen_match:
        metadata['generated'] = gen_match.group(1)

    # Models evaluated: 12
    models_match = re.search(r'Models evaluated:\s*(\d+)', text)
    if models_match:
        metadata['models_count'] = int(models_match.group(1))

    # Questions: 36
    q_match = re.search(r'Questions:\s*(\d+)', text)
    if q_match:
        metadata['questions_count'] = int(q_match.group(1))

    # Web search: **ON**
    ws_match = re.search(r'Web search:\s*\*\*([^*]+)\*\*', text)
    if ws_match:
        metadata['web_search'] = ws_match.group(1)

    return metadata


def generate_webpage():
    """Main function to generate the webpage."""
    # Paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'data' / 'phase4_report_WEBPAGE.md'
    template_dir = project_root / 'web'
    output_file = project_root / 'docs' / 'index.html'

    # Read markdown
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return False

    with open(data_file, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Parse all sections
    metadata = parse_metadata(markdown_text)
    peer_rankings = parse_peer_rankings(markdown_text)
    elo_result = parse_elo_ratings(markdown_text)
    elo_rankings = elo_result[0] if isinstance(elo_result, tuple) else elo_result
    total_matches = elo_result[1] if isinstance(elo_result, tuple) else "0"
    position_bias = parse_position_bias(markdown_text)
    model_bias = parse_model_bias(markdown_text)
    judge_generosity = parse_judge_generosity(markdown_text)

    # Question autopsy
    hardest = parse_question_autopsy(markdown_text, 'hardest')
    controversial = parse_question_autopsy(markdown_text, 'controversial')
    easiest = parse_question_autopsy(markdown_text, 'easiest')
    consensus = parse_question_autopsy(markdown_text, 'consensus')

    # Create lookup dicts for templates
    generosity_by_model = {g['model']: g for g in judge_generosity}

    # Set up Jinja2
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('template.html')

    # Render template
    html = template.render(
        metadata=metadata,
        peer_rankings=peer_rankings,
        elo_rankings=elo_rankings,
        total_matches=total_matches,
        position_bias=position_bias,
        model_bias=model_bias,
        judge_generosity=judge_generosity,
        generosity_by_model=generosity_by_model,
        hardest_questions=hardest,
        controversial_questions=controversial,
        easiest_questions=easiest,
        consensus_questions=consensus,
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated: {output_file}")
    print(f"  - Peer Rankings: {len(peer_rankings)} models")
    print(f"  - Elo Rankings: {len(elo_rankings)} models")
    print(f"  - Position Bias: {len(position_bias)} positions")
    print(f"  - Model Bias: {len(model_bias)} models")
    print(f"  - Questions: {len(hardest)} hardest, {len(easiest)} easiest")

    return True


if __name__ == '__main__':
    generate_webpage()
