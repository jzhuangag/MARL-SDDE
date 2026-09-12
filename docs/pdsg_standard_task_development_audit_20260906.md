# Standard-task policy-dependency development audit

Date: 2026-09-06

Status: development evidence only.  No result in this document is a
preregistered confirmation result, a controller comparison, or a training
return claim.

## Material Passport

- Artifact class: CPU code-experiment development audit.
- Environments: Gymnasium `HalfCheetah-v5`; PettingZoo `pistonball_v6`.
- Local environment: conda `ust2`, Python 3.11.13, Gymnasium 1.0.0, MuJoCo
  3.3.0, PettingZoo 1.26.1, pygame-ce 2.5.8, Pymunk 7.3.0, NumPy 2.2.2.
- Data origin: newly generated local CPU trajectories; no T-083A, EXP-017A,
  formal, or GPU result was reused.
- Purpose: decide whether a standard task has a state-dependent sparse
  policy-influence object before freezing a learning controller.
- Reproducibility status: recorded development outputs with SHA-256 hashes;
  no independent confirmation was attempted because the theorem-facing
  estimator gates failed.

## HalfCheetah `6x1`: stop the kinematic-chain hypothesis

The diagnostic used six distinct smooth policy blocks and four-point central
mixed finite differences of the common finite-horizon return.  One development
reset seed, horizon 40, donor step 0.08, and owner steps 0.01 and 0.02 produced
3,840 short rollouts.

- Median fraction of total cross-policy influence retained by the published
  six-joint chain: `0.3580323047`.
- Median advantage over degree-matched random supports: `0.0200362743`.
- Fraction of owners whose largest cross edge was a physical neighbor: `0.5`.
- Retained-fraction range across the two owner steps: `0.0205892552`.

The result rejects the inference that a kinematic chain is automatically a
sparse policy-gradient graph under shared rigid-body dynamics.  HalfCheetah is
retained as a dense control, not as the positive benchmark.

Output hashes:

- `influence_matrices.json`: `17631A06DF3135746777A9FB4DBA1984D36CECC54AF9457223D9192EF0EB64FF`.
- `summary_rows.csv`: `ED076C835AF225052BF846E7AB593ACCD597DEA73D945FF0EA8193AC75B1DEA9`.
- `summary.json`: `A6E5A14FF5B9B7D25D3D95CA385BBA3EB9080EF48025C89C352FE135D2AE4EF0`.

## Pistonball: state-dependent oracle locality exists

Pistonball has 20 distinct scalar policy blocks.  A public deterministic
burn-in generated three launch states with the ball near the right, middle,
and left portions of the piston array.  A launch-time ballistic interaction
tube was computed from ball position, horizontal velocity, rollout horizon,
and a declared spatial radius.  It does not use a post-rollout return.

At finite-difference step 0.08, the radius-zero predictive tubes retained
`0.875`, `0.871287`, and `1.0` of total measured cross influence while using
`0.031579`, `0.110526`, and `0.078947` of the complete directed edge set.
At step 0.04, a radius-one tube retained `1.0` at all three launch states while
using `0.078947`, `0.189474`, and `0.110526` of the complete edge set.  Thus
the high-influence support moves with the ball and a conservative predictive
tube has strong oracle compression headroom.

The mixed-derivative magnitude is not numerically stable.  Across launch
states, Spearman rank correlations between steps 0.04 and 0.08 were `0.523`,
`0.752`, and `0.945`; matrix-norm ratios were `0.696`, `5.284`, and `2.061`.
Pymunk contact switching therefore prevents interpreting this diagnostic as a
classical environment Hessian estimate.

Output hashes for the step-0.08 and step-0.04 development runs are stored in
their respective `SHA256SUMS.json` files.  The matrix hashes are
`D0DE0BB5D1FAE274FD866151B3656FE710187D667F8231DB34797DE2BADC4641`
and `FEBF0787971404C00A488DA01EC5B04D0B73313034EBC0B738D87FD7428D7292`.

## Return-only online graph discovery fails

Two lower-cost estimators were tested only as development diagnostics.

1. A paired transformed-Gaussian policy-score estimator used eight trajectories
   per donor sign.  Its radius-one median retained fraction was `0.154789`,
   median edge-cost fraction `0.110526`, random-support gap `0.098210`, and
   median signal-to-standard-error ratio `0.200389`.
2. A paired bilinear Rademacher smoothed-Hessian estimator used 64 probes per
   matrix.  Its corresponding values were `0.113098`, `0.110526`, `-0.000595`,
   and `0.713969`.

Neither estimator makes state-conditioned graph discovery reliable.  They are
stopped and will not be rescued by changing thresholds or adding formal seeds.
The positive object is instead a causal support derived from declared local
dynamics and launch state; a centralized critic may estimate signed value only
inside that support.

## Trajectory-cone width exposes a horizon phase

A subsequent reward-free development smoke used three fixed stochastic policy
profiles, the three launch states, horizons 4, 6, and 8, and 16 trajectory
noises per cell.  Sixteen calibration points cannot certify 90% split-conformal
coverage; the code correctly marked that requested certificate as unavailable.
The descriptive 80th-percentile path-residual median was 2.3741 piston widths
and its 90th percentile across cells was 5.4112.

After adding the fixed two-width contact pad, median edge-cost fractions were
0.236842, 0.347368, and 0.410526 at horizons 4, 6, and 8.  The corresponding
maximums were 0.347368, 0.9, and 1.0.  Longer rollouts can therefore erase the
sparse-graph advantage in specific launch states.  This observation motivates
joint horizon--graph control; it is not used to tune the separately frozen
PDSG-CONE-001 confirmation seeds.

Output hashes:

- `development_rows.csv`: `E78923E0444749E5B6A8ECF1AD5E64EBB22336DF0500232DC21D7A1B74A0CD3F`.
- `summary.json`: `601086966500E6F1271E9B13A904C44AEB25C2DE6BBC943222FA8D9431288155`.

## Fallacy scan (11/11)

1. No p-value was treated as an effect size; no p-value was computed.
2. No non-significant result was called evidence of equality.
3. Development seeds were not called an independent confirmation population.
4. Multiple launch states were all retained; no favorable state was selected
   post hoc as the reported population.
5. The deterministic finite-difference support was not called a smooth
   Hessian after the resolution check failed.
6. Common random numbers were used only for variance reduction, not as extra
   independent samples.
7. Return-only estimator failure was separated from oracle locality.
8. A local observation or physical edge was not assumed to imply policy-
   gradient locality; HalfCheetah provides the counterexample.
9. Edge count was reported separately from runtime and trajectory cost.
10. No causal controller-performance claim was made from a dependency audit.
11. No stopped experiment, threshold change, or unreported retry was used to
    manufacture a positive result.

## Decision

Do not preregister a HalfCheetah sparse-graph controller and do not use a
return-only sensor to infer Pistonball's graph.  Proceed to a theorem-first
causal-light-cone formulation.  Only after its tail bound and executable
Lyapunov action are closed may a fresh Pistonball controller pilot be
preregistered.
