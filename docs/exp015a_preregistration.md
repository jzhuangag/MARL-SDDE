# EXP-015A preregistration: paid adaptive participation

## Material passport

- Evidence class: implementation-only CPU pilot.
- Pilot seeds: `20271101`--`20271132`, permanently excluded from formal use.
- GPU/HPC4: prohibited for this pilot.
- Theory dependency: fixed-design Theorems 1--2 and Algorithm 1 in
  `theory_program_adaptation_cost.md`.
- Interpretation: mechanism/phase-transition validation, not nonlinear MARL.

This document and the runner are to be committed before the first pilot
output is generated. No gate below may be changed after inspecting output.

## Frozen model

- Low instance: common-factor variance `theta0=.05`.
- High instances: `theta1 in {.5,2,8}`.
- Equivalent correlation: `rho=theta/(1+theta)`.
- Public pilot mixing coefficient: `lambda in {0,.8,.95}`.
- Delay: `D in {0,8}`.
- Systems `(server overhead, maximum agents)`: `(4,8)` and `(16,32)`.
- Probe participation: `q in {2,4,8,16,32}`, clipped to available agents.
- Probe/optimization strides: `b in {1,2,4,8}`.
- Identification error target: `delta=.10`.
- Budget levels: `short=.5`, `near=1.1`, and `long=3.0` times the
  scenario-specific fixed-design threshold plus registered commit reserve.
- Every message and environment transition, including delay before commit,
  is charged.

The controller receives the two registered hypotheses, public `lambda`,
individual probe samples, previous actions, and remaining budgets. It does
not receive true `theta`, `rho`, regime label, latent common factor, oracle
action, or validation loss.

## Frozen policies

1. Oracle with true instance, evaluation only.
2. Always all-agent.
3. Fixed `q=2`.
4. EXP-014B strict fallback, represented honestly by always all-agent.
5. Horizon-aware paid ETC.
6. No mixing correction: designs and tests as if `lambda=0`.
7. No horizon awareness: probes whenever any partial probe fits.
8. Wrong cost model: selects probes using samples-per-agent and ignores the
   dual cost during design.

## Endpoints

- expected Gaussian terminal MSE and simulated squared error;
- CVaR90;
- oracle regret;
- safety deficit relative to always-all;
- correct identification probability;
- probe sample/message/environment cost;
- fallback, commit time, selected `q,b`, and completed commit updates;
- exact dual-budget validity;
- lower/sufficient theoretical probe thresholds.

## Frozen progression gates

All gates must pass:

1. Every row is finite.
2. Message and environment budgets are both respected.
3. No hidden-state leakage.
4. Paid ETC long-budget identification probability is at least `.90` in
   both low and high regimes.
5. Long-budget identification probability exceeds short-budget probability
   by at least `.25` in both regimes.
6. Short-budget paid ETC fallback rate is at least `.80` in both regimes.
7. In at least 75% of prespecified high-regime long-horizon scenarios, paid
   ETC has lower mean oracle regret than EXP-014B strict fallback.
8. The median long-horizon paid/strict oracle-regret ratio is below `.80`.

The pilot does not require strict per-cell improvement over the all-agent
baseline. It tests whether the predicted identification threshold separates
fallback from useful paid adaptation.

## Decision rule

- All gates pass: recommend a separately committed formal preregistration,
  while retaining AC-7--AC-9 as theory blockers before any ICML matching
  claim.
- Any gate fails: record an honest pilot failure, do not run formal seeds,
  and classify whether the failure is information threshold, algorithm,
  safety cost, or model novelty.

