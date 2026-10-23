#!/usr/bin/env python3
"""External process-group wall-clock watchdog, including startup/loading/warmup."""
import os,signal,subprocess,time
from pathlib import Path

def group_rss_bytes(pgid):
 total=0;pagesize=os.sysconf('SC_PAGE_SIZE')
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   tail=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(tail[2])==pgid:total+=int((p/'statm').read_text().split()[1])*pagesize
  except (OSError,ValueError,IndexError):pass
 return total

def terminate_group(proc,grace=3):
 for sig,wait in [(signal.SIGTERM,grace),(signal.SIGKILL,3)]:
  try:os.killpg(proc.pid,sig)
  except ProcessLookupError:pass
  try:proc.wait(timeout=wait)
  except subprocess.TimeoutExpired:continue
  # Kill any child remaining in the same process group after its parent exits.
  try:os.killpg(proc.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  return

def run_guarded(command,env,log,limit,stop=None,memory_gb=0,cwd=None):
 start=time.monotonic();state='completed';peak=0
 Path(log).parent.mkdir(parents=True,exist_ok=True)
 with open(log,'w') as f:
  proc=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=cwd)
  try:
   while proc.poll() is None:
    elapsed=time.monotonic()-start
    if stop is not None and stop.is_set():state='INTERRUPTED';break
    if elapsed>=limit:state='HARD_TIME_LIMIT';break
    if memory_gb:
     peak=max(peak,group_rss_bytes(proc.pid))
     if peak>memory_gb*1024**3:state='MEMORY_LIMIT';break
    time.sleep(0.2)
  except BaseException:
   terminate_group(proc);raise
  if state!='completed':terminate_group(proc)
  elif proc.returncode!=0:state='EXECUTION_ERROR'
 return dict(state=state,exit_code=proc.returncode,wall_seconds=time.monotonic()-start,hard_time_limit=limit,host_memory_limit_gb=memory_gb,peak_sampled_rss_bytes=peak,command=command)
