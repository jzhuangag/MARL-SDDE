"""Local immutable registration, exact subprocess CPU, and audited predecessor gate."""
import ctypes
from ctypes import wintypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/"protocol.json"
SPEC = HERE/"execution_spec.json"
LOCK = HERE/"lock.json"
ADMISSION = HERE/"admission.json"
RESULT = ROOT/"experiments/v3_risk_gate8h/results/formal_20261019"
PREDECESSOR = ROOT/"ICML-2027/results/sealed_bridge_learning_dev140h_20261018"
RECEIPT = ROOT/"ICML-2027/internal/sealed_bridge_learning_140h_audited_terminal.json"


def dump(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def child_cpu(handle):
    times = [wintypes.FILETIME() for _ in range(4)]
    k = ctypes.WinDLL("kernel32",use_last_error=True)
    k.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    if not k.GetProcessTimes(int(handle),*(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    return sum((int(t.dwHighDateTime)<<32)+int(t.dwLowDateTime) for t in times[2:])/1e7


def pid_alive(pid):
    k = ctypes.WinDLL("kernel32",use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    handle = k.OpenProcess(0x1000,False,int(pid))
    if not handle:
        # Access denied is not proof of exit.
        if ctypes.get_last_error()==5:
            return True
        return False
    code = wintypes.DWORD()
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    try:
        if not k.GetExitCodeProcess(handle,ctypes.byref(code)):
            return True
        return code.value==259
    finally:
        k.CloseHandle(handle)


class SingleOwner:
    def __init__(self,name="Local\\SDDESealedBridgeLearning20261016"):
        self.name = name

    def __enter__(self):
        k = ctypes.WinDLL("kernel32",use_last_error=True)
        k.CreateMutexW.restype = wintypes.HANDLE
        k.CreateMutexW.argtypes = [ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR]
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
        k.ReleaseMutex.argtypes = [wintypes.HANDLE]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.k,self.handle = k,k.CreateMutexW(None,False,self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        value = k.WaitForSingleObject(self.handle,0)
        if value not in (0,0x80):
            k.CloseHandle(self.handle)
            raise RuntimeError("Another native CPU owner holds the shared mutex")
        return self

    def __exit__(self,*args):
        self.k.ReleaseMutex(self.handle)
        self.k.CloseHandle(self.handle)


def runtime():
    import numpy, scipy, pandas, matplotlib
    import numpy.core._multiarray_umath as multi
    import numpy.random.mtrand as random_binary
    import scipy.signal._sigtools as signal_binary
    import scipy.special._ufuncs as special_binary
    import scipy.stats._continuous_distns as distribution_source
    import pandas._libs.json as json_binary
    modules = (numpy,scipy,pandas,matplotlib,multi,random_binary,signal_binary,
               special_binary,distribution_source,json_binary)
    paths = [Path(sys.executable)]+[Path(module.__file__) for module in modules]
    return dict(python=platform.python_version(),executable=str(Path(sys.executable).resolve()),
                platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,
                pandas=pandas.__version__,matplotlib=matplotlib.__version__,
                binaries={str(p):sha(p) for p in paths},
                threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})


def source_paths():
    return sorted([*HERE.glob("*.py"),PROTOCOL,SPEC,HERE/".gitattributes",
                   ROOT/"TSP-V3-revised-20261018/manuscript/tools/risk_gate_core.py"])


def run_command(args):
    p = subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err = p.communicate()
    cost = child_cpu(p._handle) if os.name=="nt" else 0.
    if p.returncode:
        raise RuntimeError(f"Command failed: {args[:2]}: {err.decode(errors='replace')[-1000:]}")
    return out,cost


def verify_sources(lock):
    for rel,digest in lock["source_sha256"].items():
        if sha(ROOT/rel)!=digest:
            raise AssertionError("Source changed: "+rel)


def verify_registration():
    lock,admission = json.loads(LOCK.read_text(encoding="utf-8")),json.loads(ADMISSION.read_text(encoding="utf-8"))
    verify_sources(lock)
    if runtime()!=lock["runtime"]:
        raise AssertionError("Runtime mismatch")
    if sha(LOCK)!=admission["lock_sha256"] or sha(PROTOCOL)!=admission["protocol_sha256"]:
        raise AssertionError("Admission mismatch")
    paths = [*lock["source_sha256"],LOCK.relative_to(ROOT).as_posix(),ADMISSION.relative_to(ROOT).as_posix()]
    cpu = 0.
    for rel in paths:
        data,cost = run_command(["git","show",f"HEAD:{rel}"])
        cpu += cost
        if hashlib.sha256(data).hexdigest()!=sha(ROOT/rel):
            raise AssertionError("Uncommitted registration: "+rel)
    ahead,cost = run_command(["git","rev-list","--count","@{upstream}..HEAD"])
    cpu += cost
    if ahead.strip()!=b"0":
        raise AssertionError("Registration not pushed")
    return lock,admission,cpu


def predecessor_ready(receipt_path=RECEIPT, root=PREDECESSOR, alive=pid_alive, workspace=ROOT):
    if not Path(receipt_path).exists():
        return False,"awaiting independent 140h final audit receipt"
    r = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    required = ("audit_complete","all_preserved_evidence_reviewed","terminal_decision_recorded")
    if not all(r.get(k) is True for k in required):
        return False,"predecessor audit incomplete"
    if not isinstance(r.get("terminal_decision"),str) or not r["terminal_decision"].strip():
        return False,"predecessor scientific/resource decision missing"
    if Path(r.get("native_root","")).resolve()!=Path(root).resolve():
        return False,"predecessor root mismatch"
    if alive(int(r["driver_pid"])):
        return False,"predecessor driver still alive"
    marker = Path(root)/r["terminal_marker"]
    if r["terminal_marker"] not in ("workflow_complete.json","failure.json"):
        return False,"invalid predecessor terminal marker"
    if not marker.exists() or sha(marker)!=r["terminal_marker_sha256"]:
        return False,"predecessor terminal digest mismatch"
    audit = Path(workspace)/r["audit_report"]
    if not audit.resolve().is_relative_to(Path(workspace).resolve()):
        return False,"predecessor audit path escapes workspace"
    if not audit.exists() or sha(audit)!=r["audit_report_sha256"]:
        return False,"predecessor final audit digest mismatch"
    return True,"140h terminal decision and independent audit verified; driver exited"


class Budget:
    def __init__(self,earlier,child_cpu_seconds=0.):
        self.earlier = float(earlier)
        self.child = float(child_cpu_seconds)

    @property
    def cpu(self):
        return self.earlier+self.child+time.process_time()

    def __call__(self):
        if self.cpu>=28800:
            raise TimeoutError("Separate V3 28800 CPU-second cumulative ceiling")
