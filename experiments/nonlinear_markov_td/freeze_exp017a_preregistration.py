"""Freeze outcome-free EXP-017A preregistration artifacts and code hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from exp017a_nonlinear_config import (
    EXPERIMENT,
    PARENT_HEAD,
    PILOT_SEEDS,
    PRIMARY_EFFECT_THRESHOLD,
    build_static_manifest,
    canonical_json,
    expected_runs,
    repository_root,
)


HASHED_FILES = (
    "experiments/nonlinear_markov_td/exp017a_nonlinear_config.py",
    "experiments/nonlinear_markov_td/run_exp017a_nonlinear_pilot.py",
    "experiments/nonlinear_markov_td/analyze_exp017a_nonlinear_pilot.py",
    "experiments/nonlinear_markov_td/test_exp017a_nonlinear_benchmark.py",
    "experiments/nonlinear_markov_td/requirements_exp017a_overlay.txt",
    "slurm/exp017a_pilot_a30.sbatch",
)


def sha256_file(path: Path) -> str:
    # Registry hashes bind the Git/Linux canonical text bytes.  Windows
    # checkouts may expose CRLF even though Git stores and HPC4 checks out LF.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def build_gate_table() -> dict[str, Any]:
    statements = {
        "G1": "all runs and registered metrics are finite; zero message/environment budget violations",
        "G2": "the exact registered task/mixing/correlation/delay/budget/policy/seed population is present",
        "G3": "every public joint regeneration certificate satisfies lambda_upper <= 1-gamma",
        "G4": "common/private complete-trajectory coupling and source-bank hashes are recorded; no observation-noise dependence shortcut",
        "G5": "information-only API contains no true rho, source mask, held-out error, MC return, teacher, or outcome input",
        "G6": "all paired policies receive identical message and environment budgets and remain communication matched",
        "G7": "learning-aware median selected q is strictly smaller at rho=0.9 than rho=0",
        "G8": "learning-aware median b under nonzero delay is not smaller than under zero delay",
        "G9": f"in high-correlation nonzero-delay cells, geometric terminal prediction risk learning-aware/information-only <= {1.0 - PRIMARY_EFFECT_THRESHOLD:.2f}",
        "G10": "the same primary ratio is <=1.05 separately on CartPole-v1 and Acrobot-v1",
        "G11": "learning-aware is <=1.10 versus the pilot best-fixed-q envelope in geometric mean and CVaR90",
        "G12": "controller wall fraction <=0.10 and every run reports server ticks, agent transitions, Bellman error, prediction error, AUC, resources, and q/b traces",
    }
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "all_mandatory": True,
        "formal_rule": "any failed G1-G12 gate stops formal; no threshold, seed, policy, or population amendment is allowed",
        "gates": [
            {"id": gate, "mandatory": True, "statement": statement}
            for gate, statement in statements.items()
        ],
        "scientific_outcomes_present": False,
    }


def build_registry() -> dict[str, Any]:
    root = repository_root()
    file_hashes = {relative: sha256_file(root / relative) for relative in HASHED_FILES}
    manifest = build_static_manifest()
    gates = build_gate_table()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "parent_head": PARENT_HEAD,
        "pilot_seeds": list(PILOT_SEEDS),
        "pilot_seed_sha256": hashlib.sha256(
            canonical_json(list(PILOT_SEEDS)).encode("utf-8")
        ).hexdigest(),
        "formal_seeds": None,
        "formal_status": "not preregistered; permitted only after every pilot gate passes",
        "expected_endpoint_rows": expected_runs(),
        "file_sha256": file_hashes,
        "static_manifest_sha256": manifest["static_manifest_sha256"],
        "gate_table_sha256": hashlib.sha256(canonical_json(gates).encode("utf-8")).hexdigest(),
        "environment": {
            "base_python": "/project/vincentlau/jzhuangag/MARL-SDDE/envs/exp014a-py39/bin/python",
            "base_versions": {
                "python": "3.9.21",
                "torch": "2.8.0+cu128",
                "numpy": "2.0.2",
                "pandas": "2.3.3",
            },
            "scratch_overlay": "gymnasium==1.0.0",
            "project_write_required": False,
        },
        "scientific_outcomes_present": False,
    }
    payload["configuration_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def preregistration_markdown(
    manifest: dict[str, Any], gates: dict[str, Any], registry: dict[str, Any]
) -> str:
    return f"""# EXP-017A nonlinear GPU pilot preregistration

## Outcome-free status

This commit freezes the standard-task nonlinear benchmark before any EXP-017A
trajectory exists. It does not modify EXP-016B code, seeds, gates, results, or
claims. Pilot seeds are implementation-only and permanently excluded from any
formal confirmation.

- Parent HEAD: `{PARENT_HEAD}`
- Configuration SHA-256: `{registry["configuration_sha256"]}`
- Static manifest SHA-256: `{registry["static_manifest_sha256"]}`
- Pilot seeds: `{list(PILOT_SEEDS)}`
- Formal seeds: **not assigned**
- Expected pilot endpoints: {expected_runs()}

## Tasks and learner

The benchmark uses Gymnasium `CartPole-v1` and `Acrobot-v1` under frozen
stochastic behavior policies. The learner is a two-hidden-layer ReLU neural
TD(0) predictor trained by plain SGD. There is no actor--critic, Hessian or
covariance inverse, preconditioner, or target-policy adaptation.

The held-out metrics are terminal Monte-Carlo prediction MSE and empirical
mean-squared Bellman error. Normalized prediction-error AUC, communication,
environment steps, total agent transitions, CVaR90, wall time, controller
overhead, and the complete selected `(q,b)` path are also frozen.

## Dependence and mixing construction

For every agent, a complete source trajectory is selected from either one
common source (probability `sqrt(rho)`) or an iid private source. Common and
private sources have exactly the same task and regeneration law. Thus each
single-agent marginal is invariant in `rho`, while two agents share their
source with probability `rho`. No observation noise is added to manufacture an
advantage.

At public regeneration events all source environments reset from their
standard Gymnasium initial laws. This is a joint Doeblin minorization:
`lambda_upper <= 1-gamma`, with frozen pairs `(0.8,0.2)` and `(0.95,0.05)`.
The experiment supports only this known-mixing setting.

## Scenarios and baselines

The Cartesian grid contains two tasks, two mixing profiles, correlations
`[0,0.5,0.9]`, zero/edge-jitter/WAN-bursty delay traces, and message- versus
environment-binding budgets. Every paired policy receives the same byte and
environment budgets. Mandatory arms are oracle (evaluation only), always-all,
fixed q=4/16/32 (forming the best-fixed-q envelope), single-agent,
information-only, learning-aware, no-delay, correlation-blind, and
mixing-blind.

The nonzero delay traces are realistic-shaped deterministic synthetic traces,
not claimed to be measurements from a deployed network.

## Inference and progression

The pilot uses descriptive frozen gates only. A later formal analysis, if
authorized, uses one-sided paired seed-block sign-flip maxT inference with
100,000 frozen resamples and familywise alpha 0.05. Pilot outcomes may select
the fixed-q baseline for the later formal registry, but formal seeds must be
new and independently committed before use.

All {len(gates["gates"])} gates are mandatory. Any failure stops formal without
gate, seed, population, or threshold adjustment. Active outputs and the
scratch environment must remain under `/scratch/jzhuangag`; `/project` is not
needed for this pilot.
"""


def audit_markdown() -> str:
    return """# EXP-017A read-only code and research audit

## Existing-code verdict

The existing `experiments/nonlinear_markov_td` code is useful mechanism
scaffolding but is not a standard nonlinear benchmark. EXP-013 uses Gaussian
or synthetic realizable teachers; EXP-014B uses a two-state binary MRP and its
recorded pilot failed because the conservative controller fell back on every
block. Those outcomes remain negative evidence and are not reused or relabeled.

Reusable components are limited to plain neural semi-gradient TD, pathwise
dual-budget accounting, delayed gradient queues, scalar dependence summaries,
and complete `(q,b)` traces. The new runner does not edit any old runner or
artifact.

## Benchmark choice

Gymnasium provides a maintained standardized RL interface. `CartPole-v1` has
a four-dimensional continuous observation and two actions; `Acrobot-v1` has a
six-dimensional continuous observation and three actions. Both are standard
Classic Control environments and have appeared in neural-TD evaluation work.
Their fixed behavior policies make prediction and Bellman error directly
measurable without adding actor--critic scope.

## Prior-art boundary

- DASA (Dal Fabbro et al., arXiv:2403.17247) gives delay-adaptive multi-agent
  stochastic approximation with independent agent Markov chains and N-fold
  speedup.
- AsyncMATD (Dal Fabbro et al., arXiv:2407.20441) gives asynchronous delayed
  multi-agent linear TD and N-fold speedup under independent observation
  processes.
- Neural TD convergence and Gym experiments already exist; nonlinear function
  approximation by itself is not novel.

Therefore EXP-017A can test external nonlinear breadth only. It cannot support
unrestricted unknown-mixing adaptation, global occupation optimality, or
general nonlinear MARL. The proposed distinction remains correlation-limited
participation under known mixing, dual resources, and heterogeneous delay.

## Primary sources checked

1. Gymnasium CartPole documentation:
   https://gymnasium.farama.org/environments/classic_control/cart_pole/
2. Gymnasium Acrobot documentation:
   https://gymnasium.farama.org/environments/classic_control/acrobot/
3. Towers et al., *Gymnasium: A Standard Interface for Reinforcement Learning
   Environments*, arXiv:2407.17032.
4. Dal Fabbro et al., *DASA*, arXiv:2403.17247.
5. Dal Fabbro et al., *Finite-Time Analysis of Asynchronous Multi-Agent TD
   Learning*, arXiv:2407.20441.
6. Tian, Paschalidis, and Olshevsky, *On the Performance of Temporal Difference
   Learning With Neural Networks*, arXiv:2312.05397.

The requested `academic-research-suite` skill was not installed in this Codex
workspace. This audit therefore used repository records and the primary
official/arXiv sources above; search snippets and secondary summaries were not
used as theorem evidence.
"""


def freeze() -> dict[str, Any]:
    manifest = build_static_manifest()
    gates = build_gate_table()
    registry = build_registry()
    manifest["configuration_sha256"] = registry["configuration_sha256"]
    root = repository_root()
    targets = {
        "docs/exp017a_nonlinear_preregistration.json": json.dumps(
            manifest, indent=2, sort_keys=True
        )
        + "\n",
        "docs/exp017a_pilot_gate_table.json": json.dumps(
            gates, indent=2, sort_keys=True
        )
        + "\n",
        "docs/exp017a_pilot_registry.json": json.dumps(
            registry, indent=2, sort_keys=True
        )
        + "\n",
        "docs/exp017a_nonlinear_preregistration.md": preregistration_markdown(
            manifest, gates, registry
        ),
        "docs/exp017a_nonlinear_audit.md": audit_markdown(),
    }
    for relative, content in targets.items():
        (root / relative).write_text(content, encoding="utf-8")
    return {
        "written": list(targets),
        "configuration_sha256": registry["configuration_sha256"],
        "expected_endpoints": expected_runs(),
        "scientific_outcomes_generated": False,
    }


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
