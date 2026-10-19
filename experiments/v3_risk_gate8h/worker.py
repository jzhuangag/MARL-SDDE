"""One admitted V3 native worker. No standalone, resume, retry or reduced suite."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
from pathlib import Path
import time
import traceback

from core import families, group_rows, LEARNERS
from analysis import analyze, plots, audit_row
from guard import Budget, RESULT, LOCK, dump, sha, verify_sources


def run(earlier,parent_pid):
    if os.getppid()!=parent_pid:
        raise RuntimeError("Worker requires the admitted single-owner supervisor")
    execution = json.loads((RESULT/"execution.json").read_text(encoding="utf-8"))
    if execution["supervisor_pid"]!=parent_pid or execution["worker_authorized"] is not True:
        raise RuntimeError("Missing supervisor admission")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    verify_sources(lock)
    plan = lock["protocol"]
    budget = Budget(earlier)
    seeds = range(plan["formal_seeds"]["start"],plan["formal_seeds"]["end"]+1)
    stages = {}
    active = None
    log_index = 0
    with (RESULT/"cost_events.jsonl").open("x",encoding="utf-8",newline="\n") as ledger:
        try:
            for phase in ("science","replay"):
                directory = RESULT/phase
                directory.mkdir(exist_ok=False)
                hashes = {}
                count = 0
                for seed in seeds:
                    verify_sources(lock)
                    budget()
                    seed_start = time.process_time()
                    target = directory/f"seed_{seed}.jsonl"
                    with target.open("x",encoding="utf-8",newline="\n") as stream:
                        local_count = 0
                        for family in families(plan):
                            for regime in ("low","high"):
                                for learner in LEARNERS:
                                    active = dict(seed=seed,family=family.key,regime=regime,learner=learner,phase=phase)
                                    for row in group_rows(seed,family,regime,learner,plan["public_grid"]["scales"],budget):
                                        # Save even a malformed numerical payload before judging it.
                                        stream.write(json.dumps(row,sort_keys=True,allow_nan=True,separators=(",",":"))+"\n")
                                        audit_row(row)
                                        local_count += 1
                                        count += 1
                                    stream.flush()
                                    budget()
                        stream.flush()
                        os.fsync(stream.fileno())
                    if local_count!=1536:
                        raise AssertionError("Incomplete seed mapping")
                    hashes[target.name] = sha(target)
                    if phase=="replay" and sha(RESULT/"science"/target.name)!=hashes[target.name]:
                        raise AssertionError("Semantic replay mismatch")
                    seed_cpu = time.process_time()-seed_start
                    ledger.write(json.dumps(dict(phase=phase,seed=seed,rows=local_count,process_cpu_seconds=seed_cpu,cumulative_charged_cpu_seconds=budget.cpu),sort_keys=True)+"\n")
                    ledger.flush()
                    log_index += 1
                    dump(RESULT/f"progress_{log_index:04d}.json",dict(phase=phase,rows=count,seed_parts=len(hashes),
                         cumulative_charged_cpu_seconds=budget.cpu,pid=os.getpid()))
                    if phase=="science" and len(hashes)==1:
                        if time.process_time()>900:
                            raise TimeoutError("First-seed cost-only profile exceeded900CPU seconds")
                        projection = budget.cpu+1.25*seed_cpu*(127+128)+2700
                        dump(RESULT/"cost_admission.json",dict(cost_only=True,first_seed_preserved_as_science_prefix=True,
                             first_seed_rows=1536,first_seed_cpu_seconds=seed_cpu,remaining_seed_parts=255,
                             projection_multiplier=1.25,analysis_audit_reserve_seconds=2700,
                             projected_charged_cpu_seconds=projection,cap_seconds=28800,
                             admitted=projection<=28800,scientific_returns_inspected_for_admission=False))
                        if projection>28800:
                            raise TimeoutError("Fixed full-suite cost forecast exceeds independent8h; no shrinkage")
                if count!=196608 or len(hashes)!=128:
                    raise AssertionError("Whole phase incomplete")
                dump(directory/"hashes.json",hashes)
                stages[phase] = dict(rows=count,seed_parts=len(hashes),cumulative_charged_cpu_seconds=budget.cpu)
                dump(RESULT/f"{phase}_complete.json",stages[phase])
            for name in ("analysis_a","analysis_b"):
                budget()
                analyze(RESULT/"science",RESULT/"replay",RESULT/name,plan,budget)
                dump(RESULT/f"{name}_complete.json",dict(cumulative_charged_cpu_seconds=budget.cpu))
            names = ("metrics.csv","summary.json","integrity.json","cell_results.csv","macro_results.csv","directional_errors.csv","zero_allowance_stress.csv")
            for name in names:
                if (RESULT/"analysis_a"/name).read_bytes()!=(RESULT/"analysis_b"/name).read_bytes():
                    raise AssertionError("Double-analysis mismatch: "+name)
            plots(RESULT/"analysis_a",RESULT/"plots",budget)
            verify_sources(lock)
            budget()
            dump(RESULT/"workflow_complete.json",dict(science_rows=196608,replay_rows=196608,
                 analysis_copies=2,byte_identical_analysis=True,stages=stages,
                 analysis_sha256={n:sha(RESULT/"analysis_a"/n) for n in names},
                 plots=[p.name for p in sorted((RESULT/"plots").glob("*.png"))],
                 worker_pid=os.getpid(),supervisor_pid=parent_pid,
                 cumulative_charged_cpu_seconds=budget.cpu,manuscript_admission=False))
        except BaseException as error:
            dump(RESULT/"worker_failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                 active=active,stages=stages,cumulative_charged_cpu_seconds=budget.cpu,
                 process_cpu_seconds=time.process_time(),retry_authorized=False))
            raise


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--earlier-cpu",type=float,required=True)
    parser.add_argument("--parent-pid",type=int,required=True)
    args = parser.parse_args()
    run(args.earlier_cpu,args.parent_pid)
