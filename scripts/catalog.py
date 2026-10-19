"""Join execution settings with separately stored historical measurements."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
def load_groups():
    groups = json.loads((ROOT / 'scripts/config/experiments.json').read_text())
    references = json.loads((ROOT / 'results/raw_results/reference/hpr_case_results.json').read_text())
    for group in groups:
        for case in group['cases']:
            case['reference'] = references[group['id']][case['name']]
    return groups

def load_catalog():
    catalog = json.loads((ROOT / 'scripts/config/reviewer_experiments.json').read_text())
    archives = json.loads((ROOT / 'results/raw_results/reference/archived_results.json').read_text())
    for experiment in catalog:
        for solver, spec in experiment['solvers'].items():
            for case in spec['cases']:
                case['archived'] = archives[experiment['name']][solver][case['key']]
    return catalog
