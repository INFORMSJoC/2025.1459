#!/usr/bin/env python3
"""Check relocation invariants, data inventories, licenses and local documentation links."""
from pathlib import Path
import csv,json,re,sys
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1]
issues=[]
for obsolete in ['config','environment','experiments','run.sh','SHA256SUMS.json','data/inputs','data/archives']:
 if (ROOT/obsolete).exists():issues.append('Obsolete package path '+obsolete)
for path in ['scripts/run.sh','scripts/config/SHA256SUMS.json','src/competitors/PDQP/LICENSE','results/raw_results/README.md','results/paper_results/README.md']:
 if not (ROOT/path).is_file():issues.append('Missing '+path)
entries=[x for n in ['manifest.json','competitor_qap_manifest.json'] for x in json.loads((ROOT/'scripts/config/data'/n).read_text())]
rows=[x for p in (ROOT/'data').glob('*/instances.csv') for x in csv.DictReader(p.open())]
index={x['path']:x for x in entries}
if len(rows)!=len(index) or {x['path'] for x in rows}!=set(index):issues.append('Family instance lists differ from exact input inventory')
for row in rows:
 if row['sha256']!=index[row['path']]['sha256'] or int(row['bytes'])!=index[row['path']]['bytes']:issues.append('Instance list identity mismatch '+row['path'])
for name in ['Synthetic-CQP','Lasso-Synthetic','Large-QAP','Trust-Region']:
 if not (ROOT/'data'/name/'LICENSE').is_file():issues.append('Missing author-generated data license '+name)
for name in ['experiments.json','reviewer_experiments.json']:
 text=(ROOT/'scripts/config'/name).read_text()
 if '"reference":' in text or '"archived":' in text:issues.append('Numerical rows remain in runner configuration '+name)
readme=(ROOT/'README.md').read_text()
if not readme.startswith('[![INFORMS Journal on Computing Logo]'):issues.append('Missing IJOC banner')
if re.search(r'^## (.+)$',readme,re.M).group(1)!='Cite':issues.append('Cite must be first README subheading')
# Check maintained documentation, not third-party source docs or historical raw records.
docs=[ROOT/'README.md']+list((ROOT/'docs').glob('*.md'))+list((ROOT/'data').rglob('*.md'))+list((ROOT/'scripts').rglob('README.md'))+[ROOT/'src/README.md',ROOT/'src/variants/README.md',ROOT/'src/competitors/PDQP/README.md']+list((ROOT/'results').glob('README.md'))+list((ROOT/'results').glob('*/README.md'))
links=0
for doc in docs:
 text=re.sub(r'```.*?```','',doc.read_text(),flags=re.S)
 for target in re.findall(r'\]\(([^)]+)\)',text):
  target=target.strip().split(' "',1)[0].strip('<>')
  url=urlsplit(target)
  if url.scheme or target.startswith('//') or not url.path:continue
  path=(doc.parent/unquote(url.path)).resolve();links+=1
  if not path.exists():issues.append(str(doc.relative_to(ROOT))+': broken link '+target)
print(json.dumps(dict(instance_representations=len(rows),local_links_checked=links,issues=issues),indent=2))
sys.exit(bool(issues))
