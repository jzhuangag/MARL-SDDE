# T-081 causal end-block controller calibration validation

## Decision

T-081 is a scientifically positive but procedurally failed old-seed
calibration.  C1--C12 and C14 pass; frozen byte-identity gate C13 fails because
the runner stored nondeterministic wall-clock time in `summary.json`.  The
endpoint and cell artifacts are byte-identical, and all scientific summary
fields are equal after removing only `runtime_seconds`, but the frozen wording
did not permit that exclusion.  Therefore T-081 does not authorize a new-seed
pilot, formal evidence, nonlinear benchmarks, GPU, or HPC4.

## Positive mechanism result

On the preregistered 96-cell identifiable class, the causal observable
controller achieves geometric cumulative-risk ratio `0.7611317476` relative
to the per-cell T-079 continuous static graph: a `23.8868%` improvement.
It is strictly better in 90/96 cells (`93.75%`).  Single-switch and alternating
ratios are `0.8247931121` and `0.7023840629`, corresponding to improvements of
17.52% and 29.76%.  Delay-specific ratios are `0.7236984599`, `0.7760082176`,
and `0.7851552570` for delays 0, 1, and 3.

The controller/local ratio on the primary class is `0.7192579090`, a 28.07%
improvement.  On stationary controls it is `1.0525034886`, within the frozen
1.06 ceiling but not a no-harm result.  Relative to continuous static graphs,
the low-signal boundary ratio is `1.0040606026` and the high-temporal boundary
ratio is `0.9849506436`; both are worse than the primary ratio, as predicted by
the separated signal/mixing motivation.

## Mechanism, budget, and complexity

The action timing is causal and matches T-079: all ten observations update the
local learner before block-end mixing, and the selected graph affects only
future blocks.  Every endpoint uses exactly 240 learning transitions, zero
extra probe transitions, and at most 18 messages.  Nonlocal collaboration is
accepted in 73.40% of recipient decisions.  Mean/max Lyapunov debt are
`0.02619`/`1.71488`; mean projected-gradient work is 30.75 iterations per
recipient decision.

The persistent certificate responds in the registered direction: mean upper
correlation is 0.1564 versus 0.7298 for low versus high temporal correlation,
while effective samples fall from 7.393 to 1.677.  The two runs completed in
266.80 s and 271.02 s, below the twelve-minute gate.

## Reproduction and C13

- `endpoints.csv` in both runs:
  `FBF7C14DB16726242CF6004A41739D2D23A50C9E3FA42F56523E655D42466A91`
- `cells.csv` in both runs:
  `1C842E93FF7B1779E8BF2F784CA130F4C42E27B3FF6605B592D406FBBDC14C89`
- original `summary.json`:
  `34A798D440DB079CE02A2ECD5B323B215BF9E890D05E161709953AA4284B5D2E`
- reproduction `summary.json`:
  `7C4E15B98D1D973472D6BBC10A6E08D201ADCD47BD26987AB21D3B0B59AD00E7`

The summary difference is exactly `runtime_seconds` (266.7988665 versus
271.0202790); the scientific dictionaries are otherwise equal.  Nevertheless,
C13 remains false.  Full regression passes: `654 passed, 7 skipped in 139.95 s`.

## Statistical-fallacy and claim-boundary scan

Coverage is 11/11.  The identifiable population was frozen before execution
from target separation and mixing assumptions; no T-081 outcome selected a
cell.  All 432 cells and 32 old seeds are retained.  Positive aggregates are
decomposed by schedule and delay, while stationary, low-signal, and
high-correlation controls remain visible, preventing Simpson and survivorship
claims.  Seed reuse and architecture-calibration taint are explicit; no
independent p-value or confidence claim is made.  The controller uses only
completed observations, and target schedules precede its decisions.  No
agent-to-population, observational-causal, diagnostic-base-rate, collider, or
extreme-score inference is made.

This is still a scalar affine Markov-learning calibration rather than standard
RL benchmark evidence.  It shows that a causal, low-complexity observable
controller captures the T-080 headroom on the separated class; it does not
establish universal no-harm or ICML readiness.

## Next admissible step

Do not reinterpret C13 or run new seeds under T-081.  First freeze a generic
artifact protocol in which timing metadata is stored outside the scientific
summary and byte-identity is defined before execution.  Any future new-seed
experiment must receive a new identifier, explicitly cite T-081 as tainted
design information, preserve the controller and primary population without
further outcome tuning, and independently preregister formal stopping gates.
