# T-065A preregistration: discrete joint-control mechanism pilot

T-065A is a prospective, outcome-free CPU pilot.  It asks whether independent
short Markov residual blocks and reward-free correlation fingerprints contain
enough information for the exact T-064 optimizer to recover useful online
joint `(q, eta)` actions across a broad coefficient grid.

This is deliberately a mechanism experiment, not ICML evidence.  It does not
use a standard RL task, prove a confidence sequence, or establish finite-time
TD risk.  Failure stops the online joint upgrade; success only authorizes the
next theorem-aligned delayed affine-TD experiment.

## Frozen design

- 324 coefficient cells and 64 fresh pilot seeds: 20,736 endpoints.
- `q` ranges over every integer from 1 to 64; `eta` lies in `[0.005,0.2]`.
- The observable action uses 64 pairs of independent length-16 stationary
  AR(1) residual blocks and 128 independent length-8 two-agent fingerprints.
- Probe data are disjoint from evaluation and cannot count as learning data.
- Every probe actor transition and message is included in both endpoint
  accounting fields and budgets.
- Comparators are the clairvoyant joint optimum, fixed `(8,0.05)`, q-only with
  `eta=0.05`, and eta-only with `q=8`.
- Lower robust drift score is better.  No evaluated noise or selected-action
  outcome is visible to the controller.

The exact frozen grid, seeds, equations, gates, and authorization are in
`docs/t065a_joint_mechanism_cpu_preregistration.json`.  Any mandatory gate
failure forbids treating the pilot as support for a later affine-TD run.  No
threshold, seed, or scenario may be changed after the independent
preregistration commit.

## Scope boundary

The experiment validates sensing-to-action mechanics and joint-action value.
It cannot by itself support an ICML claim.  A positive result must be followed
by a discrete finite-time theorem, predictable confidence bounds, a pathwise
budget shield, and positive comparisons on standard stochastic fixed-policy
RL tasks under identical resource accounting.
