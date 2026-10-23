"""Reuse fixed phase logic; warm public designs before the retained pair profile."""
import os
import argparse
import json
from support import RESULT,previous_module,plan,verify_sources,Budget
from model import old,public_design
base=previous_module('worker')
base.verify_sources=lambda lock:verify_sources(lock,executed_only=True)

def run(earlier,parent_pid):
    if os.getppid()!=parent_pid:raise RuntimeError('Admitted supervisor required')
    execution=json.loads((RESULT/'execution.json').read_text())
    if execution['supervisor_pid']!=parent_pid:raise RuntimeError('Parent identity mismatch')
    cfg=plan();budget=Budget(earlier)
    for learner in cfg['learners']:
        for f in old.families(cfg):
            for scale in cfg['public_grid']['scales']:
                budget(900);public_design(learner,f,scale)
    # All warm-up CPU remains in process_time and in the unchanged full-suite cap.
    base.run(earlier,parent_pid)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--earlier',type=float,required=True);p.add_argument('--parent',type=int,required=True)
    a=p.parse_args();run(a.earlier,a.parent)
