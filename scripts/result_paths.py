"""One result layout shared by runners, reports and package checks."""
from pathlib import Path

def experiment_name(name):
    return ('fig_' if name == 'Restart-Sigma' else 'exp_') + name

def experiment_path(output, name, paper=False):
    return Path(output) / ('paper_results' if paper else 'reproduced_results') / experiment_name(name)

def solver_path(output, name, solver, paper=False):
    return experiment_path(output, name, paper) / ('solver_' + solver)

def group_folder(group):
    return {'synthetic_denser':'denser', 'synthetic_standard':'standard',
            'trust_region':'regular', 'trust_region_large':'large'}.get(group, group)

def hpr_raw_path(output, experiment, solver, group, groups, paper=False):
    folder = solver_path(output, experiment, solver, paper) / 'raw'
    if len(groups) > 1:
        folder /= group_folder(group)
    return folder
