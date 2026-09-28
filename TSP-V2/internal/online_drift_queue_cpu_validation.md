# Online Lyapunov-bandit CPU feasibility validation

## Material Passport

- Role: algorithmic development; not manuscript evidence
- Date: 2026-09-28
- Controller: conservative sliding-window UCB with dual Lyapunov queues
- Seeds: 256 per synthetic regime
- Horizon: 120 decisions
- Catalogue: `q={1,2,4,8}`
- Result SHA-256: `5F74B8E650134C2098DA348B187A33FDFC9F4E21E224AF1E4B4C568F732AA5BE`
- Byte-identical replay: pass

## Why this resolves the certificate dead end

The controller observes the selected action's bounded validation-risk decrease
in the same block in which that action is executed.  It does not calibrate a
deep-policy coefficient and extrapolate it over a future trajectory.  The
missing global MAPPO sensitivity constant is therefore removed from the
algorithm interface.

Only one action and one validation block are charged per decision.  A hard
reserve mask provides pathwise dual-budget safety, while the quadratic
Lyapunov queues price expensive participation actions.  Sliding-window UCB
learns task- and phase-dependent progress with sparse re-probing.

## Development result

| Regime | Median risk vs. best feasible fixed action | Win rate | Interpretation |
|---|---:|---:|---|
| High dependence | `+2.853%` risk | `0.0%` | finite exploration cost; within a 3% no-harm band |
| Low dependence | `-2.982%` risk | `89.06%` | positive value from larger participation |
| Phase switch | `-11.498%` risk | `94.14%` | strong value from online adaptation |

Here a negative risk change is an improvement.  Across all seeds and regimes,
both remaining budgets are nonnegative.  The smallest remaining message
budgets are respectively `500`, `0`, and `5`; the environment budget is used
exactly in every run.

These data are deliberately synthetic and were used to develop the controller.
They do not support a paper performance claim.  They establish that the
replacement design has a nontrivial operating region and that the previous
theorem-interface failure need not terminate TSP-V2.

## Decision

The online design passes the CPU feasibility gate and replaces long-horizon
reuse of a local drift coefficient.  The next mandatory stages are:

1. prove the constrained sliding-window UCB plus Lyapunov-queue finite-time
   bound under registered piecewise-stationary Markov gain assumptions;
2. freeze the validation sensor, costs, controller constants, final evaluation
   stream, and fresh seeds in a separate preregistration commit;
3. run a small MPE/MaMuJoCo development experiment on HPC4;
4. admit no return result to the TSP manuscript until an independent fresh-seed
   confirmation passes.

The existing `TSP/` manuscript and all G0/G0-A1/G0-B artifacts remain frozen.
