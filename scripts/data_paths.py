"""Canonical reviewer paths with explicit lookup of unchanged historical data copies."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
LEGACY={x['path']:x['legacy_path'] for x in json.loads((ROOT/'data/manifest.json').read_text()) if x.get('legacy_path')}
def locate_input(relative,data_root):
 relative=Path(relative)
 if relative.is_absolute() or '..' in relative.parts:raise ValueError('Input path must be relative to the data root')
 candidates=[Path(data_root)/relative]
 if str(relative) in LEGACY:candidates.append(Path(data_root)/LEGACY[str(relative)])
 candidates.append(ROOT/'data/inputs'/relative)
 return next((p.resolve() for p in candidates if p.is_file()),None)
