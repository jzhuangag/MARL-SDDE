# T-080 validation of the T-079 continuous-static headroom audit

## Decision

All frozen T-079 gates H1--H14 pass.  The result establishes substantial
architecture headroom for time-varying continuous collaboration relative to a
strong continuous static graph in the registered exact-moment model.  It
justifies designing and independently preregistering a new-seed observable
controller pilot.  It does not itself authorize that pilot, formal evidence, a
nonlinear benchmark, GPU, HPC4, or a claim about standard RL performance.

## Primary result

Across all 288 preregistered nonstationary cells, the safe dynamic continuous
oracle has geometric cumulative-risk ratio `0.7771157825` relative to the
strongest-found continuous static graph, an improvement of `22.28842175%`.
The dynamic oracle is strictly better in 288/288 cells.  The ratios are
`0.8200256834` for single-switch schedules and `0.7364512498` for alternating
schedules.  Every delay group improves: ratios are `0.7617617160`,
`0.7837538888`, and `0.7860647771` for delays 0, 1, and 3.

The continuous static optimizer strengthens the frozen discrete comparator in
76.8519% of cells and improves its aggregate risk by 1.6721%.  Thus the main
headroom is not an artifact of comparing against the earlier finite graph
catalogue.  In stationary cells the dynamic/static ratio is `0.9555870252`, so
the improvement is 4.4413% and remains below the preregistered 5% diagnostic
ceiling.  The dynamic graph changes in every nonstationary cell.

## Numerical and budget audit

- 432/432 cells are finite, positive, and complete.
- Frozen-source replay has maximum relative error `5.171e-16`.
- Every cell has 10 successful static starts; the maximum row-sum residual is
  `2.220e-16`, and no infeasible negative weight occurs.
- The maximum dynamic row-simplex KKT residual is `5.528e-09`, below `1e-7`.
- Every dynamic policy uses exactly 240 learning transitions, no extra probe
  transition, and no more than 18 message units.  Mean rollback/shadow rate is
  zero.

## Clean reproduction and regression

The original and isolated clean reproduction each completed all twelve ordered
36-cell chunks.  Every per-chunk `cells.csv` hash matches.  The merged artifacts
are byte-identical:

- `cells.csv`: `434349484A23F0CBC8D2AE6FA20E90ED7AD90DC071E88EFDF7048E96D6C063B7`
- `summary.json`: `4833CA4109217F3729C990C9D77B40578A4D7CF4FBB01DA1C67725A69E8B0C7D`

The original and reproduction summed chunk runtimes were 17,960.19 s and
10,520.85 s, respectively.  Both primary orchestrator stderr files are empty.
The complete repository experiment suite passes under the project `.venv`
(Python 3.11.13, CVXPY 1.6.5): `646 passed, 7 skipped in 161.94 s`.

During the original run, a manually launched second resume process raced the
still-running original orchestrator at chunk 10 and exited when the original
process atomically created that chunk directory first.  The original process
then completed chunks 10--11 and the final merge.  This recovery-process race
did not alter a chunk or scientific artifact: all chunk manifests validate and
the independent single-orchestrator reproduction is byte-identical.

## Claim-boundary and fallacy scan

1. The dynamic policy uses exact latent moments and is an outcome-aware oracle,
   not a deployable controller.
2. The continuous static comparator is the strongest solution found by the
   frozen deterministic ten-start SLSQP procedure.  No global optimum is
   claimed for its nonconvex full-horizon objective.
3. The registered system is an exact affine personalized-learning model.  The
   audit is not evidence of improvement on a standard nonlinear RL benchmark.
4. The reported percentages concern cumulative personalized mean-square risk,
   not episodic return or sample complexity in an unregistered task.
5. No cell, threshold, optimizer start, or outcome was selected after seeing
   T-080 results.  Reproduction changes only the output path.
6. The result demonstrates available adaptation value, not that the current
   observable controller captures it.  A separately frozen pilot must test
   observability, estimation error, full sensing cost, runtime, and comparison
   against the continuous static baseline.

## Next admissible step

Prepare an outcome-free preregistration for a local-CPU observable-controller
pilot.  Its primary comparator must be the T-079 continuous static graph; it
must use new seeds, charge all observations and messages, include stationary
and low-identifiability controls, freeze a nontriviality gate, and stop formal
work after any mandatory failure.  No GPU escalation is currently justified.
