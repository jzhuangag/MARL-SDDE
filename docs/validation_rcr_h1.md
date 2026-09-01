# RCR-H1 validation: analytic survival gate failed

## Decision

RCR-H1 is an honest failure.  The reuse--correct--refresh program does not
have enough broad analytic headroom to justify a standard MARL implementation,
confirmation, formal experiment, or GPU run.  The frozen grid, gates, and
failed outcomes remain unchanged.

## Reproducibility

- preregistration commit: `3f98a907f0f8956feb7b9d17e7663c23cda99919`;
- corrected execution commit: `0c9fb7f40fcc7ea37feac0fb52315e58661a37cd`;
- configuration SHA-256:
  `875E692868E6696E5C4DD13C029A3E5E88914BB61E164A422243C9EF4C7E9D36`;
- runner SHA-256:
  `3E7E18849191C44D253F38CFFA7006E195F592A3BFBC4D72E99CF8E67C4C9CED`;
- primary and clean-reproduction summary SHA-256:
  `AB9BC684991154058CB1056019B350D256E005A345C6365D5C06A558734E7500`;
- primary and reproduction are byte-exact, each 1,394,920 bytes;
- 2,048 scenario rows, 393,216 causal events, and 64 seed-aggregated
  cells; all finite;
- local CPU only; no HARL trajectory, HPC4, GPU, or prior formal data.

The first post-preregistration attempt failed before creating a result
directory because the strong static comparator's L-BFGS-B solver returned an
abnormal status.  The pre-outcome numerical fallback and KKT audit are recorded
separately; no scientific setting changed.

## Frozen gate ledger

| Gate | Frozen threshold | Observed | Result |
|---|---:|---:|---:|
| H1 finite and structurally valid | required | yes | pass |
| H2 adaptive correction / best static vector | at most 0.95 | 0.993204 | **fail** |
| H3 causal RCR / best fixed period | at most 0.95 | 0.992718 | **fail** |
| H4 nonstationary cells strictly improved | at least 0.70 | 0.458333 | **fail** |
| H5 median oracle-gain capture | at least 0.50 | 0.000000 | **fail** |
| H6 stationary causal/fixed ratio | at most 1.01 | 1.000000 | pass |
| H7 each persistence ratio | at most 0.98 | 0.985905 / 0.995094 | **fail** |
| H8 refresh-budget overshoot | zero | zero | pass |
| H9 scalar-solver iterations | at most 128 | 40 | pass |
| H10 failed gate stops escalation | required | enforced | pass |

The result is 5/10 mandatory gates passed.

## Headroom diagnosis

The failure is not primarily a poor Lyapunov multiplier.  The clairvoyant
same-count refresh oracle has aggregate geometric risk ratio 0.969467 against
the best fixed-period schedule, only 3.05% optimistic headroom.  By profile:

| Profile | causal/fixed | oracle/fixed |
|---|---:|---:|
| stationary | 1.000000 | 1.000000 |
| bursty | 0.976507 | 0.931437 |
| mixed | 0.994919 | 0.957942 |
| rotating | 1.000197 | 0.994331 |

Only the bursty subset crosses a 5% oracle ceiling.  Selecting that subset
after seeing the scan would be outcome-driven.  The causal controller uses its
entire registered refresh allowance in every scenario, so zero median oracle
capture reflects poor event selection relative to the clairvoyant ranking, not
simple budget underuse.  Adaptive continuous correction itself improves the
best static vector by only 0.68% geometrically.

## Scientific consequence

The factorized geometric-path MSE bound, box-QP solver, and Lyapunov resource
interface remain reusable mathematical components.  They are not a viable
paper story by themselves.  The project will not alter the RCR-H1 profiles,
thresholds, or seeds and will not implement RCR on a standard benchmark merely
to search for a favorable task.

The next candidate must make dynamic value endogenous to asynchronous MARL.
The current leading question is whether heterogeneous update clocks change the
joint game dynamics and whether a Lyapunov clock-debt controller can recover
last-iterate stability and wall-clock convergence without synchronizing on the
slowest agent.  That candidate must pass a new theorem and CPU phase/headroom
audit before any standard MARL or GPU work.
