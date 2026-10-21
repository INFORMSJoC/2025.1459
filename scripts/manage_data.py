#!/usr/bin/env python3
"""Verify exact inputs, or download and safely unpack indexed, checksummed volumes."""
import argparse,hashlib,json,shutil,tarfile,urllib.request,os,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify(p,x):return p.is_file() and p.stat().st_size==x['bytes'] and bool(x.get('sha256')) and digest(p)==x['sha256']
class Joined(io.RawIOBase):
 def __init__(self,paths):self.paths=iter(paths);self.f=None
 def readable(self):return True
 def read(self,n=-1):
  if n<0:raise ValueError('Streaming reader requires a bounded read')
  out=bytearray()
  while len(out)<n:
   if self.f is None:
    try:self.f=next(self.paths).open('rb')
    except StopIteration:break
   b=self.f.read(n-len(out))
   if b:out.extend(b)
   else:self.f.close();self.f=None
  return bytes(out)
p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','fetch','extract']);p.add_argument('--data-root',type=Path,default=Path(os.environ.get('HPR_DATA_ROOT',str(ROOT/'data'))));p.add_argument('--groups',nargs='+');p.add_argument('--bundled',action='store_true');p.add_argument('--include-competitors',action='store_true',help='Also verify/fetch/extract explicit QAP inputs');p.add_argument('--index',type=Path,default=ROOT/'scripts/config/data/archives/archive_index.json');p.add_argument('--cache',type=Path,default=ROOT/'downloads');a=p.parse_args()
entries=json.loads((ROOT/'scripts/config/data/manifest.json').read_text())
if a.include_competitors:entries+=json.loads((ROOT/'scripts/config/data/competitor_qap_manifest.json').read_text())
chosen=[x for x in entries if (not a.groups or set(a.groups)&set(x['groups'])) and (not a.bundled or x.get('bundled'))]
if not chosen:
 if a.bundled and not a.groups:
  print('No third-party benchmark inputs are bundled. See data/README.md for acquisition.');raise SystemExit(0)
 raise SystemExit('No matching inputs')
if a.action!='verify':
 index=json.loads(a.index.read_text());wanted={x['path'] for x in chosen};a.cache.mkdir(parents=True,exist_ok=True)
 if a.include_competitors and a.index.resolve()==(ROOT/'scripts/config/data/archives/archive_index.json').resolve():
  extra=json.loads((ROOT/'scripts/config/data/archives/competitor_qap_index.json').read_text())
  index['datasets']+=extra['datasets']
 for d in index['datasets']:
  if not wanted & {x['path'] for x in d['inputs']}:continue
  if all(verify(a.data_root/x['path'],x) for x in d['inputs'] if x['path'] in wanted):continue
  paths=[]
  for part in d['parts']:
   name=part['file'];assert Path(name).name==name,'Unsafe archive filename'
   rel=Path(part.get('download_path',name))
   if rel.is_absolute() or '..' in rel.parts or rel.name!=name:raise SystemExit('Unsafe archive download path: '+str(rel))
   # Prefer the organized Dropbox layout; accept older flat caches as well.
   dest=a.cache/rel
   for old in [*part.get('legacy_download_paths',[]),name]:
    old=Path(old)
    if old.is_absolute() or '..' in old.parts or old.name!=name:raise SystemExit('Unsafe legacy archive path: '+str(old))
    if not dest.is_file() and (a.cache/old).is_file():dest=a.cache/old
   if not verify(dest,part):
    if a.action=='extract':raise SystemExit('Missing or corrupted volume: '+str(dest))
    url=part.get('url')
    if not url:raise SystemExit('No direct download URL is configured for '+name+'. Download the archive from the Dropbox collection linked in data/README.md into --cache, then use extract.')
    if not url.startswith('https://'):raise SystemExit('Only HTTPS download URLs are accepted')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.partial')
    with urllib.request.urlopen(url) as src,tmp.open('wb') as f:shutil.copyfileobj(src,f,8*1024*1024)
    if not verify(tmp,part):raise SystemExit('Download checksum failed: '+name)
    tmp.replace(dest)
   paths.append(dest)
  allowed={x.get('archive_path',x['path']):x for x in d['inputs']}
  with tarfile.open(fileobj=Joined(paths),mode='r|gz') as tar:
   for member in tar:
    rel=Path(member.name)
    if rel.is_absolute() or '..' in rel.parts or not member.isfile() or member.name not in allowed:raise SystemExit('Unsafe or unlisted tar member: '+member.name)
    spec=allowed[member.name]
    if spec['path'] not in wanted:continue
    if member.size!=spec['bytes']:raise SystemExit('Tar member size mismatch')
    target=(a.data_root/spec['path']).resolve();root=a.data_root.resolve()
    if root not in target.parents:raise SystemExit('Output escapes data root')
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(target.suffix+'.partial')
    with tar.extractfile(member) as src,tmp.open('wb') as f:shutil.copyfileobj(src,f,8*1024*1024)
    if not verify(tmp,spec):raise SystemExit('Extracted checksum failed: '+member.name)
    tmp.replace(target)
 # The index can provide hashes completed after the release manifest was frozen.
 known={x['path']:x for d in index['datasets'] for x in d['inputs']}
 for x in chosen:
  if x['path'] in known:x.update(known[x['path']])
counts={};issues=[]
for x in chosen:
 # Use the same canonical / legacy / bundled precedence as the launcher.
 from data_paths import locate_input
 f=locate_input(x['path'],a.data_root) or a.data_root/x['path']
 state='missing' if not f.is_file() else 'size_mismatch' if f.stat().st_size!=x['bytes'] else 'hash_pending' if not x.get('sha256') else 'verified' if digest(f)==x['sha256'] else 'hash_mismatch'
 counts[state]=counts.get(state,0)+1
 if state!='verified':issues.append(dict(path=x['path'],status=state))
print(json.dumps(dict(counts=counts,issues=issues),indent=2));raise SystemExit(0 if not issues else 2)
