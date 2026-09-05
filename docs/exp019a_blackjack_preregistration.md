# EXP-019A preregistration: exact-Blackjack CPU learning pilot

## Authorization and scope

T-029 passed all eight zero-trajectory static gates at result commit
`b15287f91de8a0c72bbc40f6224d15f0b3cd1494`.  EXP-019A is the separately
frozen sampled-transfer test.  It runs only on the local CPU.  It does not
authorize formal seeds, MinAtar, HPC4, GPU use, or any edit to T-029.

Frozen hashes before execution:

- config SHA-256:
  `299ff2c40ace9620040e75dca3214f24a350033cbe169f73a497bb2c84b8e0d8`;
- config-file SHA-256:
  `2155c1b2f9160040b056eb4cd6133c685e67b8acf6241df8946c043a8ad3a68f`;
- runner/analyzer SHA-256:
  `26a6ba3ce458633d74cd52076a32a1e2a152e327477bd4c4b743ae12cb88e349`.

## Frozen estimator and data law

- exact continuing 280-state Blackjack chain and exact stationary
  distribution from T-029;
- exact discounted target `V^pi` at gamma 0.99;
- tabular linear semi-gradient TD(0), constant step 0.05, coordinate
  projection to `[-100,100]`;
- every registered gradient sample is separated by five charged environment
  transitions;
- a delay-D update evaluates its TD gradient at the frozen parameter vector
  from `D` updates earlier and applies that gradient to the current vector;
- whole-tape coupling: every agent independently uses the public common tape
  with probability `sqrt(rho)` and otherwise an iid private tape.  Each agent
  has the exact same marginal task law and every pair shares the complete tape
  with probability rho.

## Frozen design

- 32 new pilot seeds: integers 3,190,001 through 3,190,032;
- q `{1,2,4,8,16,32}`, rho `{0,.1,.3,.5,.7,.9}`;
- target horizons `{512,2048}`;
- message and environment budget rays;
- delay fractions `{0,.05,.2}` converted to integer update delays;
- exact 280-float payload plus 65,536-byte server overhead;
- 72 cells, comparing the public risk-proxy selected q with the strong
  horizon-by-budget fixed fallback, for 4,608 paired endpoint rows.

The selected q and fallback are recomputed solely from the frozen public
cost/risk formula using the actual 280-parameter tabular payload.  No sampled
learning outcome enters selection.

Primary endpoint is stationary normalized MSVE-AUC over 33 fixed checkpoints.
Terminal normalized MSVE is co-primary.  The initial zero-vector MSVE is the
normalizer.  CRN pairing is by seed and cell.

## Frozen pilot gates

1. all 4,608 rows are present and finite;
2. exact message and environment budgets are never exceeded;
3. geometric selected/fallback MSVE-AUC ratio over all seed-cells is at most
   0.95;
4. the selected policy strictly improves the cellwise geometric AUC in at
   least 60% of the 36 prospectively active message cells;
5. inactive environment-cell aggregate AUC ratio is at most 1.02;
6. overall terminal-MSVE ratio is at most 0.98;
7. exact Bellman residual is at most `1e-10`;
8. no GPU or external-task result is used.

Any failure stops formal.  Gates, seeds, step size, cells, payload, q values,
and endpoints may not be changed after this preregistration.  A future formal
experiment requires a separate power audit and isolated seed registry and is
limited to at most 512 seeds.

## Claim boundary

Passing EXP-019A would validate theorem-to-learning transfer for a finite
tabular Markov task under fixed participation, correlation, delay, and dual
budgets.  It would not establish nonlinear function-approximation convergence
and would not by itself make the paper ICML-complete.  A separately gated
external neural task remains mandatory.
