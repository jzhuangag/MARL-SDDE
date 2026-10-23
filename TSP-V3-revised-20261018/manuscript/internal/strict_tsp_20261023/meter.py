"""Preserve paper-tool attempts and exact Windows process CPU, not study reruns."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
label,*command=sys.argv[1:]
record=HERE/(label+'.cpu.json')
if record.exists() or (HERE/(label+'.stdout.log')).exists():
    raise RuntimeError('Preserve existing attempt '+label)
started=time.process_time()
with (HERE/(label+'.stdout.log')).open('xb') as out,(HERE/(label+'.stderr.log')).open('xb') as err:
    child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
    code=child.wait()
    times=[wintypes.FILETIME() for _ in range(4)]
    get_times=ctypes.windll.kernel32.GetProcessTimes
    get_times.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    if not get_times(wintypes.HANDLE(int(child._handle)),*[ctypes.byref(v) for v in times]):
        raise ctypes.WinError()
    ticks=lambda t:(t.dwHighDateTime<<32)|t.dwLowDateTime
    cpu=(ticks(times[2])+ticks(times[3]))/1e7
value=dict(label=label,command=command,cwd=str(ROOT),pid=child.pid,exit_code=code,
    child_cpu_seconds=cpu,helper_cpu_seconds=time.process_time()-started,
    scope='Ordinary paper/literature work; separate from immutable completed native study ledgers; grandchildren excluded.')
with record.open('x',encoding='utf-8') as out:json.dump(value,out,indent=2)
print(json.dumps(value));sys.exit(code)
