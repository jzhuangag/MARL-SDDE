# EXP-007C preregistration: joint correlation--delay mean-square step size

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp007c_preregistration_v1

## Decision being tested

EXP-007B rejected the exact delayed mean-recursion boundary as a stochastic
TD controller.  EXP-007C tests whether one additional scalar second-moment
term is sufficient to recover a useful low-complexity safety rule.  No
preconditioner, actor--critic architecture, matrix inverse, or learned control
policy is allowed.

The homogeneous sample TD map is

\[
e_{k+1}=e_k-\eta\frac{1}{q}\sum_{i=1}^{q}
H_{i,k}e_{k-\tau_i},\qquad
H_{i,k}=\phi(S_{i,k})
[\phi(S_{i,k})-\gamma\phi(S'_{i,k})]^\top .
\]

Let \(A=\mathbb E[H]\),
\(\mu=\lambda_{\min}((A+A^\top)/2)\), and
\(B=\mathbb E[H^\top H]\).  Under the registered exchangeable pair-sharing
model,

\[
\alpha(q,\rho)=\rho+\frac{1-\rho}{q},
\]

\[
K(q,\rho)=\lambda_{\max}\!\left(
\alpha(q,\rho)B+
[1-\alpha(q,\rho)]A^\top A
\right).
\]

For zero delay and temporally independent matrices,
\(2\mu/K(q,\rho)\) is the elementary Euclidean second-moment step-size
threshold obtained from
\(\mathbb E\|(I-\eta\bar H)e\|^2\).
Let \(\eta_{\rm mean}(q,D)\) be the exact first stability boundary of the
delayed mean companion matrix from EXP-007B.  The registered joint rule is
the parallel sum

\[
\eta_{\rm joint}(q,D,\rho)=
\left[
\eta_{\rm mean}(q,D)^{-1}
+\frac{K(q,\rho)}{2\mu}
\right]^{-1}.
\]

This rule is a theorem-inspired conjecture for Markovian heterogeneous-delay
TD, not a theorem claimed in advance.  The experiment is allowed to falsify
it.  Its runtime computation is scalar once \(K\), \(\mu\), and the
mean-delay table are estimated or cached.

## Registered design

- MRP, features, reward, discount, pair-sharing mechanism, agent delay
  ordering, initialization, and divergence threshold are unchanged from
  EXP-007A/B.
- Agent counts: \(q\in\{16,32\}\).
- Registered maximum delays: \(D\in\{8,32\}\).
- Correlations: \(\rho\in\{0,0.9\}\).
- Horizon: 4,000 updates.
- Formal evaluation seeds: 32 consecutive seeds beginning at 20261230.
  These seeds have not been used in EXP-006A--C or EXP-007A--B.
- The analytic \(B\) and \(K\) values use the known registered finite MRP;
  no evaluation outcome is used to fit a constant.

The registered policies are:

1. `joint_aware`: \(\eta_{\rm joint}(q,D,\rho)\);
2. `correlation_blind`: replace \(K(q,\rho)\) by \(K(q,0)\);
3. `delay_blind`: replace \(\eta_{\rm mean}(q,D)\) by
   \(\eta_{\rm mean}(q,0)\);
4. `mean_only`: use \(\eta_{\rm mean}(q,D)\);
5. `worstcase_correlation`: replace \(K(q,\rho)\) by \(K(q,1)\);
6. a fixed logarithmic step-size grid
   \(\{0.005,\ldots,0.8\}\) with 13 points, used only as an offline
   tightness oracle.

All policies see identical transition paths within a seed/cell.  Checkpoint
errors are recorded at updates 250, 500, 1,000, 2,000, and 4,000.  A run that
crosses squared error \(10^{12}\) remains in every aggregate.

## Preregistered gates

All six gates must pass.

1. **Analytic correlation saturation.**  From \(q=16\) to \(q=32\),
   \(K(q,0)\) must decrease by at least 20%, whereas \(K(q,0.9)\) must
   decrease by at most 2%.  Also
   \(K(q,0.9)/K(q,0)\ge 5\) for both \(q\).
2. **Joint safety and contraction.**  `joint_aware` has zero threshold
   crossings and the median final squared error is below the initial value
   one in every one of the eight \((q,D,\rho)\) cells.
3. **Correlation awareness has operational value.**  In the four
   \(\rho=0.9\) cells, `correlation_blind` crosses the threshold in at least
   25% of runs overall, while `joint_aware` has zero crossings.
4. **Delay awareness has operational value.**  In the two \(q=32,D=32\)
   cells, `delay_blind` crosses the threshold in at least 25% of runs
   overall, while `joint_aware` has zero crossings.
5. **The rule is not vacuously conservative.**  In at least six of eight
   cells, \(\eta_{\rm joint}\) is at least one quarter of the largest
   registered grid step size whose threshold-crossing rate is at most 5%.
   In at least three of the four \(\rho=0\) cells, `joint_aware` reaches
   squared error 0.5 no later in median than `worstcase_correlation`.
6. **Accounting and numerical validity.**  All registered rows and
   checkpoints are present; every non-crossing output is finite; crossing
   times and flags agree; and all aggregate decisions are deterministic
   functions of the saved raw rows.

## Audit rules

- A failed gate remains failed; constants and thresholds will not be changed
  after the formal run.
- Smoke tests may check code shape and runtime only.  Their seeds are excluded
  from formal output.
- The exact same-seed run must be reproduced into a separate directory and
  all registered artifacts compared by SHA-256.
- Any post-hoc alternative is labelled exploratory and cannot repair the
  registered verdict.

