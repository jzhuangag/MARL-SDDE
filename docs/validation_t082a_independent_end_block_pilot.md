# T-082A independent causal end-block pilot validation

## Decision

All preregistered gates P1--P14 pass.  T-082A provides independent, sampled,
byte-reproducible CPU evidence that the causal block-end primal-dual controller
captures the continuous dynamic-collaboration headroom on the registered
separated signal/mixing class.  This authorizes only a separate formal
preregistration with untouched controller and population.  It does not itself
authorize formal execution, nonlinear benchmarks, GPU, or HPC4.

## Independent primary result

The 64 pilot seeds have zero overlap with every T-071A/T-081 design seed.  On
the frozen 96-cell primary class, the observable controller has geometric
cumulative-risk ratio `0.7623942487` relative to the per-cell continuous static
graph, an improvement of `23.7606%`.  It is strictly better in 90/96 cells
(`93.75%`).

Single-switch and alternating ratios are `0.8272208881` and `0.7026478644`,
corresponding to improvements of 17.28% and 29.74%.  Delay-specific ratios are
`0.7242621097`, `0.7756389260`, and `0.7888300974` for delays 0, 1, and 3, so
every registered delay retains at least 21.12% improvement.  The
controller/local ratio is `0.7211076100`, a 27.89% improvement.

The independent estimates closely reproduce the old-seed architecture
calibration (23.89% primary improvement and the same 93.75% strict-cell
coverage), despite complete seed separation.

## Boundary controls and mechanism

The stationary controller/local ratio is `1.0516470021`, below the frozen 1.06
ceiling but still a 5.16% stationary cost; universal no-harm is not claimed.
The low-signal controller/static ratio is `1.0059131602`, while the
high-temporal-correlation ratio is `0.9844778522`.  Both contain far less value
than the primary ratio, preserving the preregistered phase ordering rather than
hiding difficult regimes in the aggregate.

Nonlocal collaboration occurs in 73.38% of recipient decisions.  Mean/max
Lyapunov debt are `0.02670`/`2.09520`, and mean projected-gradient work is
30.70 iterations per recipient decision.  Mean correlation upper bounds rise
from 0.1563 to 0.7295 as temporal correlation increases, while effective
samples fall from 7.395 to 1.679.  Every endpoint uses 240 learning
transitions, zero extra probe transitions, and no more than 18 messages.

## Reproduction and regression

The original and clean reproduction completed in 499.16 s and 556.94 s.
Execution metadata was kept outside the scientific summary.  All three frozen
scientific artifacts are byte-identical:

- `endpoints.csv`:
  `3D566DFD77C6C4DDF72D87DCFA2EC0C46141C2A8A6424404081353B88118C889`
- `cells.csv`:
  `0C8FF1BB903F401EE93BF154F7ADCDCFC2755A5000F4200E14C4FFD5C65779DB`
- `summary.json`:
  `6C91A3C55EB3288CCD820E705D84AB1C8534950513B1011D34F9B083773CCBBE`

Both stderr files are empty.  Full regression passes under the project
environment: `660 passed, 7 skipped in 110.90 s`.

## Statistical-fallacy and claim-boundary scan

Coverage is 11/11.  The controller, comparator, primary class, thresholds, and
seed-generation rule were committed before execution.  All 27,648 endpoints,
432 cells, and 64 seeds are retained.  Positive results are decomposed by both
schedule families and all delays; stationary, low-signal, and high-correlation
controls remain reported, guarding against survivorship and Simpson
aggregation.  Common random numbers pair controller, static, and local
policies without making an observational causal claim.  There is no
post-outcome subgroup selection, seed reuse, missing-run exclusion,
agent-to-population inference, collider adjustment, diagnostic-base-rate
claim, or extreme-score screening.

The evidence concerns cumulative personalized mean-square risk in the frozen
scalar affine Markov system.  It is not yet a standard nonlinear RL benchmark,
episodic-return, or broad multi-agent population result.  The stationary cost
and high-correlation attenuation prevent a universal no-harm claim.

## Next admissible step

Independently preregister a formal CPU confirmation with new formal seeds,
unchanged code hash, the same 96 primary and 336 control cells, the same
continuous static comparator, and the same P2--P11 thresholds.  Formal seeds
must remain hidden from pilot analysis until that commit.  Any formal failure
stops nonlinear/GPU escalation.  In parallel, the paper-level theory must
derive the block-end QP from a delayed drift-plus-penalty bound and state the
separated mixing/signal assumptions explicitly.
