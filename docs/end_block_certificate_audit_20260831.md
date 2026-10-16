# End-block certificate audit — 2026-08-31

Status: one confirmed accounting defect; further theory obligations open. No frozen code changed, no scientific trajectory generated.

## Confirmed duplicate evidence

`t081_end_block_primal_dual_controller.py:114` stores the current block with `update_completed_block`. Lines 122–124 then pass that same block to `estimate_for_current_prefix`. In `t074_persistent_certificate_controller.py`, this method adds the prefix scatter, degrees, lag numerator/denominator and pair count to stored totals. Thus the current block is counted twice at each decision. It is not duplicated twice in persistent storage: the second addition is local to estimation. Earlier blocks are stored once.

Deterministic CPU check, no stochastic seed: use the four rows `(0,1),(1,0),(0,1),(1,0)`, delta=.05, rho_cap=.95. A fresh certificate with this prefix returns `(rho_upper, effective, pairs)=(0.7841002757,1.0,3)`. After first storing the same block it returns `(0.5544426221,1.1465392716,6)`. No new data were acquired; the radius nevertheless shrinks. This confirms erroneous evidence accounting, not a measured change in final learning performance.

## Consequences and next checks

T-083A remains a reproducible measurement of its frozen implemented algorithm; the result cannot establish validity of this confidence certificate. Do not silently fix historical code or report corrected performance without a new protocol. A future interface must distinguish “stored evidence only” from “past evidence plus an unseen prefix,” with an explicit invariant that each block contributes once.

Even after fixing accounting, the centered lag-ratio plus square-root radius is not automatically a valid time-uniform Markov confidence bound. Prove coverage under stated assumptions or label it an estimator, not a safety certificate. Next audit the empirical debt/true risk link, block-generation law, and the finite-time Lyapunov inequality before designing a shield or new efficacy experiment. The primary formal population is temporally uncorrelated; do not tune a replacement using its formal endpoints.
