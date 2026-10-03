# TSP-V3 MaMuJoCo calibrated-catalogue development preregistration

## Status

This document freezes `TSP-V3-MARL-MAMUJOCO-CAL-DEV-001` before any new
controller trajectory is generated.  It is development only.  The preserved
fixed-q development trajectories from `TSP-MARL-MAMUJOCO-DEV-001` are reused
as baselines; neither those data nor the eight new controller runs can become
manuscript confirmation evidence.

The scientific implementation was fixed first at commit
`16abe256a82a0d5f46bb2a9fad6cb03c1ba962f0`.  The present preregistration
commit adds only the frozen configuration, execution script, and this plan.

## Motivation fixed before execution

The previous MaMuJoCo development run correctly separated low- and
high-dependence rollouts but allowed all q in `{1,2,4,8}`.  Its return-free
correlation score chose q=1 in the shared regime although the complete fixed-q
development envelope selected q=2.  The new controller therefore freezes the
development-calibrated online catalogue `{2,8}` while retaining q=1 and q=4
as external fixed comparators.  The low-complexity selection formula and all
training hyperparameters are unchanged.

This is not an outcome-free confirmation.  It is a prospective test of a
specific redesign suggested by the preserved development result.

## Frozen experiment

- Task: MaMuJoCo `HalfCheetah-v2/2x3`, shared-parameter MAPPO.
- Coupling regimes: `independent` and `shared`.
- New method: return-free Lyapunov probe-then-commit with candidates `{2,8}`.
- Baselines: preserved fixed q in `{1,2,4,8}` under the same task, budgets,
  training seeds, and upstream HARL commit.
- New controller runs: 2 regimes x 4 development seeds = 8.
- Training seeds: `125001`--`125004`.
- Disjoint new probe seeds: `136001`--`136004`.
- Probe: q=8 for 16 fully charged non-learning blocks.
- Message budget: 5,000,000.
- Environment budget: 1,000,000 ticks.
- Scratch root:
  `/scratch/jzhuangag/MARL-SDDE-TSP-V3-MAMUJOCO-CAL-DEV-001`.
- Durable `/project` writes: forbidden.

## Frozen gates

All gates are mandatory:

1. exactly eight new controller artifacts and a 40-cell combined lattice;
2. finite AUC, exact dual-budget accounting, clean pinned upstream, and valid
   per-artifact SHA-256 manifests;
3. q=8 in at least 3/4 independent runs;
4. q=2 in at least 3/4 shared runs;
5. controller no worse than 2% below the complete fixed-q envelope in each
   regime;
6. equal-regime-mixture AUC at least 1% above the strongest single fixed q;
7. probe messages at most 1% of the budget;
8. scalar selection overhead at most 2% of total wall time;
9. two analysis executions produce byte-identical CSV and JSON output.

Any failed gate stops confirmation.  No seed, task, budget, candidate,
threshold, or method may be changed after outcome access under this experiment
identifier.

## Frozen hashes

| File | SHA-256 |
|---|---|
| `TSP/experiments/run_mappo_probe_commit.py` | `771ad905ecbcff46a924efae4a8c04a5229a76d4146917aeada70409f1c1002b` |
| `TSP/experiments/lyapunov_probe_commit.py` | `37a23e91269ba29fa8c193ea5584258aebfc4c2f3cc789ed37a27003e6d60cfc` |
| `TSP-V3/experiments/analyze_marl_mamujoco_calibrated_development.py` | `68c295f45c4af5d83318afc0220f35a1df6a8028c6bf94c24fe56f2de7b112aa` |
| `TSP-V3/experiments/marl_mamujoco_calibrated_development.json` | `3a7e66dc9c82b704fcdaee205f632a7a3b462418c3c4a901ab17e21bf9075027` |
| `TSP-V3/slurm/marl_mamujoco_calibrated_development_a30.sbatch` | `bd32711f2dbbb5545e1c68e07bc921c7f96b39e4765c39c63beca47201559152` |

## Post-development rule

Only a complete pass authorizes a separate fresh-seed confirmation
preregistration.  Confirmation must rerun the controller and all fixed q in
`{1,2,4,8}` on new training/probe seeds and apply a prespecified paired
one-sided test.  Only passing confirmation may enter the TSP manuscript.

