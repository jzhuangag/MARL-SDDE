# EXP-016A frozen analysis plan

## Analysis populations

Every trajectory is assigned before analysis to exactly one of:

- theorem-scope `below_bn`;
- theorem-scope `gray_zone`;
- theorem-scope `above_bs`;
- mixing-boundary negative control;
- oracle-gap negative control.

The gray zone is descriptive and is excluded from every mandatory theorem
gate except the prespecified break-even localization calculation. Negative
controls are never pooled with theorem-scope cells and cannot rescue a
positive gate. Low and high directions are always analyzed separately.

The atomic cell is
`scenario_id × regime × budget_point × policy`. There is no pooling across
\(Q,\theta\)-gap, \(\lambda\), delay, overhead, budget ray, safety slack, or
regime. Reported summaries may average cells only after every constituent
cell's mandatory condition has been evaluated.

## Raw outcomes and derived endpoints

Raw trajectory fields are individual-observation-derived terminal squared
error, probe/test decision, selected \(q,b,\eta\), probe length, scheduled and
usable downstream updates, charged message/environment use, intended and
actual budget violations, commit/fallback, and regime decision. Hidden
\(\theta,\lambda\), latent common factors, and oracle actions are simulator
metadata and may enter only oracle/evaluation joins, never non-oracle policy
inputs.

Derived endpoints are:

- empirical terminal MSE: arithmetic mean of trajectory squared error;
- oracle regret: paired squared-error difference from the infeasible oracle
  under the same CRN block;
- baseline-relative safety deficit: positive part of the paired loss
  difference from always-all, divided by mean always-all loss;
- high-regime gain: paired always-all oracle regret minus controller oracle
  regret;
- wrong-commit loss: paired loss difference from the action chosen under the
  correct regime;
- directional identification error, fallback/commit probability, probe
  length, selected action, resource use, and violation probability;
- empirical break-even: the first registered scale with positive paired mean
  gain; no interpolation or post-hoc cell deletion;
- CVaR90: the empirical Rockafellar--Uryasev functional
  \(\min_z[z+(0.1n)^{-1}\sum_i(L_i-z)_+]\), including fractional mass at the
  empirical quantile when needed.

Analytic risks are forbidden as observed outcomes. They are used only in the
committed threshold manifest, the oracle benchmark, and public controller
qualification.

## Pairing and common random numbers

The resampling unit is the complete seed block within a scenario, regime,
and budget. All policy comparisons are paired by seed. Potential randomness
uses the committed key
`SHA256(seed|scenario|regime|physical_time|agent)`. Policy identity is absent
from the key. A policy sees only requested agents at reached physical times;
unrequested or future observations are not consumed, shifted, or reused.

## Simultaneous inference

The familywise level is `alpha=0.05`.

- Continuous paired endpoints use 20,000 seed-block bootstrap resamples with
  fixed analysis seed `20400101`. A centered studentized maximum statistic
  over all cells in the relevant gate family supplies simultaneous one-sided
  95% bounds. Zero empirical standard error is handled conservatively: the
  bound equals zero unless every paired difference is identical and strictly
  beyond the registered effect threshold.
- Directional error and event probabilities use exact one-sided
  Clopper--Pearson bounds with Holm's step-down allocation across all
  prespecified cells in that gate family. Low-to-high and high-to-low errors
  are separate families and both must pass.
- With zero events in \(n\) trials at adjusted tail level \(a\), the frozen
  upper bound is \(1-a^{1/n}\); zero observed events are never reported as
  zero risk.
- No unregistered subgroup, alternative confidence method, or aggregate
  “most gates pass” rule is allowed.

## Effect sizes and gates

The exact G1--G12 rules are machine-readable in
`exp016a_gate_table.json`. Particularly:

- G4 requires every above-\(B_S\) high-regime cell to have positive
  simultaneous lower confidence bound and at least 2% relative improvement
  versus always-all.
- G5 compares the simultaneous safety upper bound with the scenario's frozen
  `epsilon_safe` (`0.10` or `0.20`).
- G6 uses the directional target `delta=0.025`.
- G7 requires at least 75% of base scenarios to place their first positive
  registered budget in the closed analytic bracket \([B_N,B_S]\).
- G8 requires at least 3% relative improvement and a positive simultaneous
  lower bound versus the information-only controlled-sensing baseline in
  every prespecified delayed dual-budget cell.
- G9/G10 are per-cell mechanism gates; violations cannot be averaged away.

All twelve mandatory gates must pass before a separate formal-stage commit.

## Reproducibility and reporting

Core output order is frozen lexicographically by
`scenario_id, regime, budget_point, seed, policy`. Floats use IEEE-754
double precision and CSV `%.17g`; JSON uses sorted keys. The report must show
below-\(B_N\), gray, above-\(B_S\), negative-control, delay-sensitive,
message-limited, and environment-limited panels separately. A clean rerun
must reproduce core CSV/JSON byte-for-byte.

This plan contains no scientific outcomes.
