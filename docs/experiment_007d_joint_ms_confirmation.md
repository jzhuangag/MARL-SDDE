# EXP-007D preregistration: fresh-seed mean-square confirmation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp007d_preregistration_v1

## Purpose

EXP-007C formally failed because three gates treated catastrophic threshold
crossing as the primary proxy for mean-square stability.  Its labelled
post-hoc audit showed that finite but noncontracting trajectories made that
proxy insensitive.  EXP-007D performs a fresh-seed confirmation using
continuous squared error and a one-sided confidence bound.

The algorithm is not changed.  In particular, no coefficient is refitted:

\[
\eta_{\rm joint}(q,D,\rho)
=
\left[
\eta_{\rm mean}(q,D)^{-1}
+\frac{K(q,\rho)}{2\mu}
\right]^{-1},
\]

\[
K(q,\rho)
=
\lambda_{\max}\!\left(
\left[\rho+\frac{1-\rho}{q}\right]B+
\left[1-\rho-\frac{1-\rho}{q}\right]A^\top A
\right).
\]

The MRP, policy steps, five comparator policies, 13-point fixed-step grid,
4,000-update horizon, checkpoints, initialization, and paired-path design are
identical to EXP-007C.

## Fresh sample

- Formal seeds: 64 consecutive seeds beginning at 20270130.
- No seed from EXP-006A--C, EXP-007A--C, or an EXP-007D smoke test may enter
  the formal result.
- Expected formal rows:
  \(64\times2\times2\times2\times(5+13)=9{,}216\).

## Registered uncertainty calculation

Every confidence limit uses a deterministic 20,000-resample nonparametric
bootstrap with seed 20270730.

- A cell mean-square contraction upper limit is the 99th percentile of
  bootstrap sample means of final squared error.
- A paired policy-effect lower limit is the 1st percentile of bootstrap
  medians of the within-seed final-error ratio.
- A fixed grid step is `useful` only if its threshold-crossing rate is at most
  5% and its 99% bootstrap upper limit on mean final squared error is below
  the initial error one.

These are simultaneous all-cell gates: a gate passes only if every listed
cell passes its local condition.  No failed cell may be removed.

## Preregistered gates

All seven gates must pass.

1. **Analytic participation saturation.**  The exact \(K\) gate from
   EXP-007C repeats: independent \(q=16\) to 32 reduction at least 20%;
   \(\rho=0.9\) reduction at most 2%; high-correlation inflation at least
   fivefold for both \(q\).
2. **Joint mean-square contraction.**  `joint_aware` has no threshold
   crossings and the 99% bootstrap upper mean final error is below one in all
   eight cells.
3. **Correlation-awareness value.**  In each of the four \(\rho=0.9\)
   cells, the 99% bootstrap lower limit of the paired
   `correlation_blind / joint_aware` final-error ratio exceeds two.
4. **Delay-awareness value.**  For each \(\rho\in\{0,0.9\}\) at
   \(q=32,D=32\), the 99% bootstrap lower limit of the paired
   `delay_blind / joint_aware` final-error ratio exceeds 1.05.
5. **Nonvacuous tightness.**  In all eight cells,
   \(\eta_{\rm joint}\) is at least 40% of the largest useful registered
   grid step.
6. **Correlation adaptation retains speed.**  In every independent-agent
   cell, the median update at which `joint_aware` first reaches squared error
   0.5 is no later than `worstcase_correlation`, and its step size is at least
   four times the worst-correlation step.
7. **Accounting, numerics, and reproducibility.**  All 9,216 rows and
   checkpoints are valid.  A separate exact same-seed rerun must reproduce
   every saved artifact byte-for-byte.

## Interpretation rule

- If all gates pass, the scalar joint rule is promoted as a confirmed
  algorithmic prototype and the next theoretical target is a rigorous
  Lyapunov--Krasovskii inequality that upper-bounds its parallel-sum form.
- If any gate fails, the rule is not retuned.  The paper mainline falls back
  to correlation-limited effective participation and mean-square stability
  characterization, with no claim of an adaptive safe-step controller.

