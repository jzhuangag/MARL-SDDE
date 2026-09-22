# T-027 exact FrozenLake static-scan validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-02
- Verification Status: VERIFIED WITH FROZEN GATE FAILURE
- Preregistration Commit: `5d459332d01e3872d63a2ff21164543d8a33e9fc`
- Config SHA-256:
  `f9251599e0382309e5d08d115bf04def6feffb00bff2109de548543643269442`

## Decision

T-027 is a reproducible **7/8 static failure**. Slippery FrozenLake 8x8 is
not authorized for a nonlinear learning pilot under the frozen gates. No
scientific trajectory, formal seed, HPC4 job, or GPU job was created.

## Gate ledger

| Gate | Result | Value |
|---|---:|---:|
| S1 exact Markov validity | PASS | 64-state stochastic matrix; stationary residual below 1e-10 |
| S2 finite mixing certificate | PASS | max-row TV <= 0.05 at stride 52; observed 0.047492 |
| S3 aggregate oracle value | PASS | 9.4404% >= 5% |
| S4 directional cells | **FAIL** | 14/48 = 29.1667% < 60% |
| S5 distinct oracle q | PASS | q = 1, 4, 16, 32 all occur |
| S6 internal message optimum | PASS | q=4 and q=16 occur |
| S7 budget direction | PASS | environment q >= message q in 24/24 pairs |
| S8 no trajectory taint | PASS | zero trajectories/outcomes |

Both target horizons select q=4 as the message-binding strong fallback and
q=32 as the environment-binding fallback. In the 24 message cells, 14 are
strictly improved by the cellwise oracle (58.3333%); in all 24 environment
cells, q=32 is both fallback and oracle, so none is strictly improved.

## Structural gate audit

The frozen S4 denominator exposes a feasibility defect. Under the registered
environment budget, usable horizon is independent of q. Since
`rho+(1-rho)/q` is non-increasing in q, every environment-binding cell has
q=32 as its oracle, and the environment strong fallback is also q=32. With
half the registered cells environment-binding, an all-cell strict-improvement
fraction can never exceed 50%; the frozen 60% gate is structurally
unattainable for any task under this risk model.

This observation does not rescue T-027. Its frozen result remains 7/8, and
even a post-hoc message-only diagnostic is 58.3333%, still below 60%. Any
future experiment must use a new identifier and prospectively define
directional value on adaptation-active cells while retaining inactive cells
under a separate expected-tie/budget-direction gate.

## Exact reproduction

The independent rerun is byte-identical:

- `summary.json` SHA-256:
  `f69d18831142c789e3133278c35212e0512e1297614f5d55cc615f50c6f61864`;
- bytes: 73,659 in both runs;
- exact match: PASS.

FrozenLake may remain a low-cost exact-mixing calibration appendix, but it is
not the external nonlinear learning benchmark needed for submission breadth.

