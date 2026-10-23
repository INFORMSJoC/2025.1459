"""Locate per-family inputs while accepting historical data installations."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
ENTRIES = [x for name in ('manifest.json', 'competitor_qap_manifest.json')
           if (ROOT / 'scripts/config/data' / name).exists()
           for x in json.loads((ROOT / 'scripts/config/data' / name).read_text())]
LEGACY = {x['path']: x.get('legacy_paths', []) for x in ENTRIES}

def locate_input(relative, data_root):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Input path must be relative to the data root')
    candidates = []
    for root in [Path(data_root), ROOT / 'data']:
        candidates.append(root / relative)
        for old in LEGACY.get(str(relative), []):
            alias = Path(old)
            if alias.is_absolute() or '..' in alias.parts:
                raise ValueError('Unsafe legacy input path')
            candidates.append(root / alias)
    return next((p.resolve() for p in candidates if p.is_file()), None)
