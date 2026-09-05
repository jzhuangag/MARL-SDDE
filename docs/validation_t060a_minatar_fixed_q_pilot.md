# T-060A MinAtar fixed-q CPU pilot validation

## Decision

T-060A is a reproducible prospective pilot failure.  Six of nine frozen
scientific/validity gates pass.  The result does not authorize a MinAtar
controller, formal seeds, GPU, HPC4, or `/project` output.  The preregistered
tasks, splits, thresholds, analysis, and failed decision remain unchanged.

The failure is not absence of fixed-q value.  The split-selected empirical
cell action obtains a 20.33% aggregate held-out improvement, but it strictly
improves only 39.29% of cells, is 2.23% worse on Seaquest, and is nonincreasing
with correlation on only 5/12 task-overhead-delay paths.  A small number of
large Asterix/Breakout gains dominate its geometric aggregate.  Thus the
registered empirical selector is not a stable transferable oracle.

## Provenance

- preregistration commit: `672c86c`;
- pre-outcome uint32 seed amendment: `d4e57da`;
- configuration SHA-256:
  `2a8b9b5f80174a2440c21d7c9a62c1bc372495188569596cc560eda9542604c1`;
- 32 new master seeds split 16/16 for selection/validation;
- 84 cells and 8,064 endpoints;
- official `MinAtar==1.0.15` Asterix, Breakout, and Seaquest;
- local CPU only; no GPU, HPC4, `/project`, or external artifact write.

The first attempted run produced no trajectory, moment, endpoint, or result
directory: NumPy's legacy RNG rejected the 12-digit provenance seed before
reset.  Amendment 1 froze the uniform map `seed mod 2**32` without changing a
scientific choice, after which the primary run began.

## Frozen gates

| Gate | Result | Value |
|---|---:|---:|
| V1 complete and unique | pass | 8,064/8,064 |
| V2 finite and dual-budget valid | pass | zero violations |
| V3 independent reference stability | pass | all three tasks |
| V4 held-out aggregate value | pass | ratio 0.796661 <= 0.95 |
| V5 held-out breadth | **fail** | 0.392857 < 0.60 |
| V6 taskwise value | **fail** | Seaquest 1.022345 > 0.98 |
| V7 rho direction | **fail** | 5/12 < 0.75 |
| V8 split isolation | pass | disjoint 16/16 |
| V9 no controller | pass | no controller field/outcome |

Task ratios are 0.707027 for Asterix, 0.699498 for Breakout, and 1.022345
for Seaquest.  The selected strong fixed q is q=4 except for
Seaquest/overhead-32, where it is q=16.

## Reference and numerical validity

The two independent reference banks agree closely.  Relative drift
disagreements are 0.00272--0.00401, and covariance-weighted fixed-point
prediction disagreements are 0.00587--0.07454.  Drift condition numbers are
2.13--2.49.  Every regularized symmetric drift minimum eigenvalue is at least
0.0800000, and every lifted delay-0/delay-8 spectral radius is below 0.996.
Hence the negative selector result is not explained by a vacuous or unstable
reference TD problem.

## Exact reproduction

The clean rerun reproduces all four artifacts byte for byte.  A separate
validator then reloads `endpoints.csv` and recomputes the entire summary; its
maximum numerical difference is exactly zero.

| Artifact | SHA-256 |
|---|---|
| `endpoints.csv` | `3e58807e4e200cf25b8b8162e6f4b605c8e53f11e89ab159846db27fa750c5bf` |
| `cells.csv` | `50cf50d20afa3fb5530b5430b52e279722af1805e17fe0f6a75fa0b56fc9017d` |
| `reference_moments.json` | `54f7d139202a46cb7c6264ba888d07ad213a596dcfcc330cdea9f2eb7f73e59a` |
| `summary.json` | `0de91624a57b4a88143c3098ddd1f17e4896088c018c94b3d38453567553a7ac` |

Primary scientific runtime was 1,393.62 seconds.  The identical clean rerun
took 5,197.17 seconds under lower available CPU share.  Runtime is not stored
in a scientific artifact and does not enter any gate.

## Post-result theory-rule audit

T-060A called a split-selected empirical action a cellwise oracle.  That
object is neither a population oracle nor the outcome-independent stationary
rule proved in T-050.  A read-only post-result audit therefore applied the
pre-existing action

`argmin_q (overhead+q) * (rho+(1-rho)/q)`

to the untouched validation half.  This audit is discovery evidence, not a
T-060A gate or formal paper result.  Its aggregate ratio is 0.705188 (29.48%
improvement), strict-cell fraction is 60.71%, and 50,000 seed-cluster
bootstrap one-sided 95% upper ratio is 0.807821.  Task ratios are 0.574032,
0.669491, and 0.912503.  Seaquest remains uncertain: its bootstrap 95%
interval is [0.7262, 1.1984].

This contrast localizes the issue: reward-noisy empirical cell selection can
overfit even when the low-variance closed-form correlation phase transfers.
It does not retroactively pass T-060A.  It supports only a new outcome-free
power/cost design for a prospective theorem-rule confirmation using entirely
new seeds.  Any such experiment must separately charge observable fingerprint
probes and compare against the frozen no-probe strong q.

