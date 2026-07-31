# EXP-014A pilot validation: state/risk-aware neural TD

## Material Passport

- Evidence status: implementation-only pilot
- Source base: `4ec0c6d48ac7063a15c7b707b6e750159e4a8711`
- Pilot seeds: 20270801--20270808, permanently excluded from confirmation
- Hardware: NVIDIA A30 on HPC4
- Outcome: numerically valid; scientific progression gate not met
- Formal preregistration and confirmation: not run

## Phase-0 audit

The required ancestor `c01b900` is present. The untouched remote checkout
passed 96 tests before implementation. The isolated environment is
`/project/vincentlau/jzhuangag/MARL-SDDE/envs/exp014a-py39`; its lock is
`/project/vincentlau/jzhuangag/MARL-SDDE/artifacts/exp014a-py39-lock.txt`.
The implemented tree passed 102 tests before the first expanded pilot.

## Controller and baselines

The predictable controller chooses from a finite set of `(q,b,eta)` actions.
Its scalar risk contains observable block progress, certified effective
participation `q/[1+(q-1)rho_upper]`, a delay stability screen, message and
mixing costs, and an upper-tail uncertainty term. It stores no covariance
matrix, inverse Hessian, or preconditioner. A streaming gradient-trace probe
uses `O(d)` memory and `O(qd)` work.

The pilot compared all-agent adaptive step, fixed q=4, correlation-only,
delay-only, state/risk-aware, and a charged privileged-information policy.
Full block trajectories include q, b, eta, estimated and true effective
participation, messages, environment steps, wall time, gradient trace,
stability events, training loss, its 90th percentile, and teacher MSE.

## Implementation audit

Four small pilot iterations exposed two real implementation defects rather
than scientific evidence:

1. v1 incorrectly used temporal gap `b` to attenuate simultaneous
   cross-agent correlation.
2. v3 allowed q=1 at cold start, making cross-agent dependence
   unidentifiable and self-confirming.

Both were corrected with regression tests. The identifiable v4 controller
charges an initial q=32 probe and retains q>=4 in its adaptive candidate set.
The v5 sensitivity lowered the message price to test low-correlation recovery.
No formal seed, gate, endpoint, or exclusion rule was touched.

## Expanded pilot results

The unchanged v4 controller was run on eight pilot seeds in job `1679955`.
All 288 endpoints were finite. Paired geometric teacher-MSE ratios for
state/risk over the named baseline were:

| rho | delay | vs all-agent | vs correlation-only | vs fixed q=4 |
|---:|---:|---:|---:|---:|
| 0.0 | 0 | 2.044 | 0.188 | 0.257 |
| 0.0 | 8 | 1.016 | 0.233 | 0.282 |
| 0.5 | 0 | 1.644 | 0.603 | 1.976 |
| 0.5 | 8 | 0.765 | 0.368 | 1.029 |
| 0.9 | 0 | 0.979 | 0.523 | 1.068 |
| 0.9 | 8 | 0.284 | 0.206 | 0.271 |

At `rho=.9, delay=8`, endpoint mean/CVaR90 changed from
`0.238/0.770` for all-agent to `0.049/0.119` for state/risk. At
`rho=.9, delay=0`, the means were similar (`0.209` versus `0.210`) while
CVaR90 was lower (`0.472` versus `0.735`). However, v4 was worse than
all-agent in both zero-correlation cells and in the `rho=.5,delay=0` cell.

The preregistration sensitivity v5 restored q=32 at `rho=0,delay=0`, exactly
matching all-agent there, and retained a benefit at `rho=.9,delay=8`
(paired geometric ratios 0.501 versus all-agent and 0.364 versus
correlation-only). It still lost at `rho=0,delay=8` (1.462 versus all-agent)
and `rho=.9,delay=0` (1.376). Tuning therefore stopped.

## Decision

The pilot is a **scientific progression failure**, not a numerical failure.
It supports an identifiable state/correlation interaction in the
high-correlation delayed regime, but the current single scalar surrogate does
not robustly improve both mean and upper-tail risk relative to all-agent
across delay regimes. EXP-014A was not preregistered and no confirmation
seeds were run.

The next justified algorithmic step is to separate the delay-specific
transient model from the correlation-limited variance model and calibrate
their interaction on a new implementation-only task family. A standard MARL
benchmark was not started because the required controller smoke gate failed;
skipping directly to benchmark outcomes would make policy selection and
resource matching post hoc.

All scratch outputs are retained. Completed pilot artifacts are copied under
`/project/vincentlau/jzhuangag/MARL-SDDE/artifacts/` with per-run SHA-256
verification. No scratch data was deleted.
