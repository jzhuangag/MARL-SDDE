"""Fail-closed native authorization, immutable fingerprints and one local owner."""
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = HERE/'development.json'


def dump(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    entries = {str(p.relative_to(path)).replace('\\', '/'): sha(p)
               for p in sorted(Path(path).rglob('*')) if p.is_file()
               and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def fingerprint():
    from sealed_bridge.native import base, FARAMA
    import gymnasium, gymnasium_robotics, mujoco
    # All transitive project sources, old configuration and native runtime trees.
    paths = list(HERE.glob('*.py'))+[PLAN]
    paths += list((HERE.parent/'sealed_bridge').glob('*.py'))
    paths += [HERE.parent/'happo_batch_cpu.py', HERE.parent/'happo_batch_cpu_preregistration.json',
              HERE.parent/'sealed_bridge/qualification.json',
              ROOT/'ICML-2027/internal/sealed_bridge_learning_preregistration_20261016.md']
    files = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(set(paths))}
    harl = subprocess.check_output(['git', '-C', str(base.UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(base.UPSTREAM), 'status', '--porcelain'], text=True):
        raise AssertionError('Pinned HARL changed')
    return dict(files=files, harl_commit=harl, harl_tree=tree_hash(base.UPSTREAM/'harl'),
                pettingzoo_tree=tree_hash(base.RUNTIME), farama_tree=tree_hash(FARAMA),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                gymnasium=gymnasium.__version__, robotics=gymnasium_robotics.__version__,
                mujoco=mujoco.__version__)


def profile_plan(plan):
    result = json.loads(json.dumps(plan))
    for task in result['tasks']:
        task['training_transitions'] = task['horizon']*plan['profile']['episodes']
    result['evaluation_intervals'] = plan['profile']['evaluation_intervals']
    result['evaluation_episodes'] = plan['profile']['evaluation_episodes']
    return result


class Budget:
    def __init__(self, plan, lock_path, root, phase, earlier_cpu=0.):
        self.plan, self.root, self.phase = plan, Path(root), phase
        self.lock_path = Path(lock_path)
        lock = json.loads(self.lock_path.read_text(encoding='utf-8'))
        if lock['plan'] != plan or lock['fingerprint'] != fingerprint():
            raise AssertionError('Prospective source/config/runtime lock mismatch')
        self.fingerprint = lock['fingerprint']
        self.source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        # Verify that the complete registration was committed and pushed BEFORE data.
        committed_lock = subprocess.check_output(['git', 'show', f'{self.source_commit}:{self.lock_path.relative_to(ROOT).as_posix()}'], cwd=ROOT)
        if hashlib.sha256(committed_lock).hexdigest() != sha(self.lock_path):
            raise AssertionError('Lock is uncommitted')
        for rel, digest in self.fingerprint['files'].items():
            if hashlib.sha256(subprocess.check_output(['git', 'show', f'{self.source_commit}:{rel}'], cwd=ROOT)).hexdigest() != digest:
                raise AssertionError(f'Uncommitted source {rel}')
        ahead = subprocess.check_output(['git', 'rev-list', '--count', '@{upstream}..HEAD'], cwd=ROOT, text=True).strip()
        if ahead != '0':
            raise AssertionError('Registration is not pushed')
        # Include imports, fingerprinting and registration checks in active CPU.
        self.start, self.earlier_cpu = 0., float(earlier_cpu)
        self.active_cell = None
        self.log_index = 0

    @property
    def cpu(self):
        return time.process_time()-self.start+self.earlier_cpu

    def __call__(self):
        if self.cpu > self.plan['cpu_seconds_limit']:
            raise TimeoutError('Whole registered profile+science+replay+analysis CPU cap')
        if self.phase == 'profile' and time.process_time()-self.start > self.plan['profile']['cpu_seconds_limit']:
            raise TimeoutError('Prospective cost-only profile CPU cap')

    def authorize(self, task_index, seed, method, actual_plan):
        self()
        if self.active_cell != (task_index, seed, method):
            raise AssertionError('Cell is not owned by the locked driver')
        expected = profile_plan(self.plan) if self.phase == 'profile' else self.plan
        allowed = [self.plan['profile']['seed']] if self.phase == 'profile' else self.plan['seeds']
        if actual_plan != expected or seed not in allowed or method not in self.plan['methods']:
            raise AssertionError('Unauthorized cell/configuration')
        self.verify_sources()

    def verify_sources(self):
        for rel, digest in self.fingerprint['files'].items():
            if sha(ROOT/rel) != digest:
                raise AssertionError(f'Frozen source mutated: {rel}')

    def progress(self, phase, completed):
        self.log_index += 1
        dump(self.root/f'progress_{self.log_index:04d}.json', dict(
            phase=phase, completed=completed, cpu_seconds=self.cpu, pid=os.getpid()))

    def record_failure_metadata(self, metadata):
        dump(self.root/'failed_cell_partial_costs.json', dict(cell=self.active_cell, **metadata))


def require_cell(budget, task_index, seed, method, plan):
    if not isinstance(budget, Budget):
        raise RuntimeError('Native execution requires a committed, pushed registration and single driver')
    budget.authorize(task_index, seed, method, plan)


class SingleOwner:
    """Windows kernel mutex; no filesystem cleanup or stale-file deletion."""
    def __enter__(self):
        if os.name != 'nt':
            raise RuntimeError('This registration freezes the local Windows runtime')
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateMutexW.restype = ctypes.c_void_p
        k.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k.ReleaseMutex.argtypes = [ctypes.c_void_p]
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        self.k, self.handle = k, k.CreateMutexW(None, False, 'Local\\SDDESealedBridgeLearning20261016')
        if not self.handle:
            raise OSError(ctypes.get_last_error())
        status = k.WaitForSingleObject(self.handle, 0)
        if status not in (0, 0x80):
            k.CloseHandle(self.handle)
            raise RuntimeError('Another complete learning/profile/replay owner is active')
        return self

    def __exit__(self, *args):
        self.k.ReleaseMutex(self.handle)
        self.k.CloseHandle(self.handle)
