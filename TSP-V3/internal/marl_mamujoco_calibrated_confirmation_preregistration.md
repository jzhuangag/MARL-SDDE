# TSP-V3 calibrated-catalogue MaMuJoCo confirmation preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-29
- Verification Status: PREREGISTERED-DESIGN
- Version Label: TSP-V3-MARL-MAMUJOCO-CAL-CONF-001

## Purpose

This experiment independently confirms the calibrated-catalogue Lyapunov
controller on fresh MaMuJoCo trajectories.
The controller performs a fully charged, non-learning residual probe, chooses
between q=2 and q=8 without observing benchmark return, and then commits for
the remaining dual-resource budget.
The complete fixed-q catalogue q in `{1,2,4,8}` is rerun on the same fresh
training seeds as the controller.

The predecessor `TSP-V3-MARL-MAMUJOCO-CAL-DEV-001` passed every frozen gate
and was used only to authorize confirmation, freeze q=8 as the strongest
single fixed comparator, and set the practical-effect floors below.
No development trajectory is eligible for the confirmation analysis or the
manuscript return claim.

## Frozen task, algorithm, and accounting

- Task: MaMuJoCo `HalfCheetah-v2/2x3` with shared-parameter MAPPO under
  centralized training and decentralized execution.
- Upstream HARL commit:
  `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- Coupling regimes: marginal-preserving independent and shared rollout
  innovations.
- Methods: the frozen Lyapunov probe-then-commit controller and fixed q in
  `{1,2,4,8}`.
- Controller catalogue: `{2,8}`; q=1 and q=4 remain external fixed
  comparators.
- Probe: q=8 for 16 non-learning blocks, charged to both budgets.
- Rollout length: 200; server overhead: 800.
- Message budget: 5,000,000; environment budget: 1,000,000 ticks.
- Actor and critic learning rates: 0.0005.
- Two hidden layers of width 128 with tanh activation; five actor and critic
  epochs.
- Deterministic evaluation every 200 learner updates with 20 episodes and four
  evaluation threads.
- Primary outcome: unsmoothed return AUC on 21 points of the maximum charged
  message/environment budget fraction.
- Secondary outcome: terminal return at unit charged budget fraction.

The controller cannot access evaluation return during selection.
Only the disjoint probe residuals, registered costs, and remaining horizons
enter the Lyapunov score.

## Frozen lattice and seeds

- Fresh training seeds: `138001--138008`.
- Disjoint fresh probe seeds: `139001--139008`.
- Both sets are disjoint from development training seeds `125001--125004` and
  development probe seeds `136001--136004`.
- Total runs: 2 coupling regimes x 8 seeds x 5 methods = 80.
- Scratch-only root:
  `/scratch/jzhuangag/MARL-SDDE-TSP-V3-MAMUJOCO-CAL-CONF-001`.
- Writes to `/project` and `/home` are forbidden.

## Paired estimands and inference

Let `A(r,s,m)` denote return AUC for regime r, training seed s, and method m.
The independent noninferiority estimand compares the controller with fixed
q=8 within seed.
The shared noninferiority estimand compares the controller with fixed q=2
within seed.

The primary superiority estimand first forms an equal-regime mixture within
each seed,

```text
C_s = [A(independent,s,controller) + A(shared,s,controller)] / 2,
B_s = [A(independent,s,q8) + A(shared,s,q8)] / 2,
G_s = (C_s - B_s) / |B_s|.
```

For each paired relative estimand, the registered one-sided 95% Student lower
bound is the sample mean minus
`1.894578605061305 * sample_std / sqrt(8)`.
The superiority estimand must also have at least 7/8 positive pairs, which is
equivalent to an exact one-sided sign-test p-value no greater than 9/256.
All tests form an intersection-union success rule; no failed component may be
dropped.

## Mandatory confirmation gates

Every gate must pass:

1. exactly 80 unique, finite records with exact dual-budget accounting, clean
   upstream state, and valid per-artifact hashes;
2. at least 7/8 independent controller runs select q=8;
3. at least 7/8 shared controller runs select q=2;
4. the one-sided lower bound versus q=8 in the independent regime is at least
   -2%;
5. the one-sided lower bound versus q=2 in the shared regime is at least -2%;
6. the equal-regime mixture gain versus frozen fixed q=8 has positive
   one-sided lower bound and mean gain at least 5%;
7. at least 7/8 mixture pairs are positive and the exact one-sided sign-test
   p-value is at most 0.05;
8. mean terminal-return change versus fixed q=8 under the same mixture is at
   least -2%;
9. probe messages use at most 1% of the message budget;
10. scalar selection overhead is at most 2% of training wall time;
11. two independent analysis executions produce byte-identical CSV and JSON
    results.

Any failure yields `stop`, preserves every outcome, and prohibits replacement
seeds, threshold changes, or manuscript admission under this identifier.

## Precision rationale

The four development pairs yielded a mean equal-regime gain of 13.4153% over
fixed q=8, with sample standard deviation 11.8168 percentage points and all
four paired gains positive.
At the same variance, eight confirmation pairs give an expected one-sided
lower bound near 5.5 percentage points.
The registered 5% mean-effect floor is therefore materially below the
development estimate while still requiring a practically relevant gain.
The exact sign gate separately requires broad seed-level directionality.
This planning calculation is not confirmation evidence.

## Frozen provenance

| File | SHA-256 |
|---|---|
| `TSP/experiments/run_mappo_probe_commit.py` | `771ad905ecbcff46a924efae4a8c04a5229a76d4146917aeada70409f1c1002b` |
| `TSP/experiments/lyapunov_probe_commit.py` | `37a23e91269ba29fa8c193ea5584258aebfc4c2f3cc789ed37a27003e6d60cfc` |
| `TSP/experiments/run_mappo_return_bridge.py` | `16289055f3390cf7c51ff954f54a6c47797bb00f8555c9ef7f780d53680787ed` |
| `TSP-V3/experiments/analyze_marl_mamujoco_calibrated_confirmation.py` | `1f669d7ad16cf5d0ee2c641d28bb1657708b17c51b37656d48674881b29b2f20` |
| `TSP-V3/experiments/marl_mamujoco_calibrated_confirmation.json` | `c5a00e8c82b74e9bfd5b18bdfba4b6dfb118e15688e606f590a8c07ce66326a6` |
| `TSP-V3/slurm/marl_mamujoco_calibrated_confirmation_a30.sbatch` | `652a989a47970bcdcfb9d445ad8536a2b5a4d890c80db4e1c512b783e672009c` |

Only a complete pass may be migrated into the IEEE TSP manuscript.
