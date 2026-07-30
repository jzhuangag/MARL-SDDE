# EXP-001: Dependence-limited speedup under delayed Markov updates

This experiment is the first go/no-go test for the proposed research direction
on delay- and dependence-adaptive learning from multi-agent Markov data.

## Question

Does cross-agent dependence fundamentally limit parallel speedup, and can
heterogeneous staleness make accepting more agents suboptimal even when the
step size is tuned?

## Model

The scalar server iterate obeys

\[
x_{k+1}
=x_k-\eta a\sum_d w_d x_{k-d}
+\eta\sqrt{\rho}\sum_d w_d c_{k-d}
+\eta\sqrt{1-\rho}\,e_k .
\]

Here:

- \(w_d\) is the fraction of accepted agents with delay \(d\);
- \(c_k\) is a unit-variance AR(1) common factor;
- \(e_k\) is the average of \(q\) independent AR(1) factors, so
  \(\operatorname{Var}(e_k)=1/q\);
- \(\rho\) controls the common-noise fraction.

The registered baseline uses `sample_time` alignment: both the parameter and
the common Markov factor are evaluated at the worker's sample time. A separate
post-baseline sensitivity may use `server_time` alignment, in which parameter
copies are stale but agents share the same current global Markov factor. The
two mechanisms must not be pooled because sample-time staggering can itself
decorrelate a persistent common factor.

For deterministic delays, the recursion is represented as an augmented linear
system. Stability is determined by its spectral radius, while finite-horizon
and stationary mean-square errors are computed exactly using a discrete
Lyapunov equation.

## Pre-registered go/no-go checks

1. **Correlation saturation:** the improvement from \(q=1\) to \(q=32\) at
   \(\rho=0.9\) is less than half the improvement under independent noise.
2. **Interior parallelism optimum:** with heterogeneous delays and
   \(\rho=0.9\), the joint oracle selects fewer than all 32 agents.
3. **Delay-only gap:** a controller selected under the independence assumption
   has at least 20% larger finite-horizon MSE than the joint oracle at
   \(\rho=0.9\).
4. **Numerical verification:** Monte Carlo and exact finite-horizon MSE agree
   within 5% at the selected validation points.

These checks are exploratory feasibility gates, not paper-level hypothesis
tests. The thresholds were fixed before examining the generated results.

## Run

From this directory:

```powershell
python run_experiment.py --output-dir results/baseline
```

For the shared-current-environment sensitivity:

```powershell
python run_experiment.py `
  --common-noise-alignment server_time `
  --output-dir results/server_time_sensitivity
```

For the post-baseline transient-to-stationary crossover analysis:

```powershell
python run_crossover_analysis.py
```

For the predictable stagewise controller:

```powershell
python run_stagewise_controller.py `
  --output-dir results/stagewise `
  --num-seeds 64 `
  --bootstrap-replications 2000
```

For the pre-registered budget-matched participation surface:

```powershell
python run_budget_participation.py `
  --output-dir results/budget_participation `
  --mc-replications 10000
```

For the online probe-charging controller:

```powershell
python run_online_participation.py `
  --output-dir results/online_participation `
  --num-seeds 64 `
  --bootstrap-replications 2000
```

The sparse dynamic controller is registered as EXP-005C:

```powershell
python run_sparse_dynamic.py `
  --output-dir results/sparse_dynamic `
  --num-seeds 64 `
  --bootstrap-replications 2000
```

Its first primary execution timed out. The authorized v2 execution below uses
a semantics-checked Numba block kernel and completed within the original
timeout.

The EXP-006A oracle phase diagram is:

```powershell
python run_oracle_phase.py --output-dir results/oracle_phase
```

The EXP-006B raw observable-state controller and EXP-006C scalar
Lyapunov-surrogate controller are:

```powershell
python run_state_correlation.py --output-dir results/state_correlation `
  --num-seeds 64 --bootstrap-replications 2000
python run_lyapunov_state.py --output-dir results/lyapunov_state `
  --num-seeds 64 --bootstrap-replications 2000
python run_linear_td_correlation.py `
  --output-dir results/linear_td_correlation `
  --num-seeds 32 --bootstrap-replications 2000 --workers 4
python run_td_delay_stability.py `
  --output-dir results/td_delay_stability --num-seeds 16
python run_joint_mean_square_step.py `
  --output-dir results/joint_mean_square_step --num-seeds 32
python run_joint_ms_confirmation.py `
  --output-dir results/joint_ms_confirmation --num-seeds 64
python run_exact_lifted_boundary.py `
  --output-dir results/exact_lifted_boundary
python run_markov_jump_boundary.py
python run_expanding_markov_td.py
python run_decorrelated_theorem.py
python run_sharp_delay_bound.py
python run_predictable_mixing_controller.py
python run_finite_budget_controller.py
python run_joint_qbe_controller.py
python run_progressive_anytime_controller.py
```

Run deterministic implementation checks with:

```powershell
python -m unittest -v test_linear_model.py test_stagewise_controller.py `
  test_budget_participation.py test_online_participation.py `
  test_sparse_dynamic.py test_oracle_phase.py
python -m pytest -q test_markov_jump_ms.py `
  test_predictable_mixing_controller.py `
  test_progressive_mixing_controller.py
```

The default experiment uses 500 server iterations, \(q\in
\{1,2,4,8,16,32\}\), seven common-noise fractions, 55 step sizes, and 4,000
Monte Carlo replications for three exact-solution checks.

## Outputs

- `sweep.csv`: every exact parameter evaluation;
- `best_by_setting.csv`: best step size for each scenario, correlation, and
  agent count;
- `policy_comparison.csv`: joint oracle, delay-only selection, and fixed-agent
  baselines;
- `monte_carlo_validation.csv`: exact versus simulated finite-horizon MSE;
- `summary.json`: configuration, environment, and go/no-go outcomes;
- four PNG figures used for diagnosis.

The crossover analysis writes a separate `results/crossover/` directory. It is
exploratory and does not alter the registered EXP-001 verdict.

EXP-004 writes `results/stagewise/`. Its pre-registered primary gate failed:
joint step–participation adaptation strongly outperformed the independence-
based delay-only controller, but did not improve MSE over dependence-aware
step-size adaptation with all agents retained.

EXP-005A writes `results/budget_participation/`. All five pre-registered gates
passed. Under the primary matched message budget, independent-noise cells
selected \(q=32\), while the \(\rho=0.9\) cells selected \(q=1\) and attained
about 26% of the best all-agent MSE. This is an oracle mechanism result, not
yet an online-controller claim.

EXP-005B writes `results/online_participation/`. Five of six registered gates
passed. The controller selected many agents under independent noise and few
under clustered/global/mixed noise, and achieved 34.4% of the all-agent MSE in
correlated cells. It failed to beat fixed \(q=1\) after charging the 18%
full-probe cost, so the full-probe controller is rejected.

EXP-005C implements 2.4% sparse probing and within-run dependence shifts. Its
authorized execution v2 completed 64 paired seeds and reproduced all eight
artifacts byte-for-byte. The overall gate failed: adaptive/best-fixed was 2.619
(95% interval [1.628, 4.080]), adaptive/oracle was 8.226 ([6.215, 10.580]),
and switch response passed only 1/4 regimes. The piecewise oracle itself used
median \(q=32\) in every regime, exposing a mismatch between the registered
switch directions and the finite-budget proxy.

EXP-006A scans 9,720 deterministic oracle cells over dependence, budget,
current error, delay, and overhead. It passed five of six gates and reproduced
all artifacts exactly. Correlation responsiveness covered 37.22% of tracks and
ten contiguous actionable regions, but only 5.31% of groups changed optimal
agent count across delay. The combined gate therefore failed; the supported
target is state- and correlation-adaptive participation, with delay retained in
the stability/rate model rather than forced to control \(q\) directly.

EXP-006B rejects the raw gradient-magnitude state proxy. EXP-006C replaces it
with a scalar Lyapunov risk recursion and passes four of seven gates: it
improves on both raw-state and correlation-only controllers but remains 20.4%
worse than the strongest fixed \(q=4\) baseline and agrees with the
clairvoyant oracle in only 26.67% of post-warm-up actions. All nine outputs
reproduce byte-for-byte. The next work therefore moves to a theorem-first
correlation-limited speedup result and a linear TD benchmark rather than
further proxy tuning.

EXP-007A evaluates actual linear TD(0) on a seven-state Markov reward process.
All six formal gates pass and all 13 artifacts reproduce byte-for-byte.
Independent paths give median \(N_{\rm eff}(32)=30.996\); shared paths at
\(\rho=0.9\) give 1.111. At the long budget, the optimal count moves from
\(q=16\) to \(q=1\), with strong paired endpoint improvements. The audit also
finds that registered delays do not alter the selected count or step size, so
EXP-007A supports correlation-limited participation but not delay adaptivity.

EXP-007B activates the exact delayed mean-TD stability boundary. It passes four
of six gates and reproduces all seven artifacts exactly, but rejects the
mean-boundary step-size controller: stochastic TD can diverge below the mean
spectral boundary, especially under strong cross-agent correlation. This
motivates a correlation-aware mean-square Lyapunov bound for random delayed
Jacobians.

EXP-007C combines the exact mean boundary with the analytic aggregate-Jacobian
second moment through a scalar parallel-sum step. It contracts in all eight
cells, but formally fails three crossing-based gates because the blind rules
remain finite despite very large error. EXP-007D freezes the formula and uses
64 fresh seeds with 99% bootstrap mean-square endpoints. All seven gates pass,
all 9,216 rows are valid, and all ten artifacts reproduce byte-for-byte.
Doubling \(q\) reduces multiplicative curvature by 22.46% under independence
but only 0.32% at correlation 0.9. The correlation-blind/joint paired
final-error ratio has a 99% lower limit of at least 8.19 in every
high-correlation cell.

EXP-008A constructs the exact heterogeneous-delay covariance operator without
materializing its Kronecker matrix. The formal result passes four of seven
gates. The scalar joint rule is strictly stable in all 12 exact cells, and the
zero-delay agent-count effect saturates under high correlation as predicted.
The rule is not uniformly tight: it uses 13.3%--54.1% of the independent-time
exact boundary. The exact analysis also shows that delay dominates the
independent \(q=32,D=32\) cell, whereas multiplicative correlation dominates
the low-delay high-correlation cells.

EXP-008B validates the mode-conditioned Markov-jump covariance construction
but retains its weak temporal effect as a negative control. EXP-008C activates
a locally expanding conditional TD regime: at persistence 0.98, every exact
boundary is only 1.9%--4.0% of its i.i.d. counterpart, and the uninflated rule
is unstable in all 24 high-persistence cells.

EXP-008D validates a coarse proof-derived decorrelation rule and identifies a
delay-constant loss. EXP-008E replaces that loss with a sharp \(L_2\) bound;
all five gates pass in 72 exact cells. The online computation is a scalar
search using aggregate curvature, a total-variation certificate, and RMS
delay. It does not form a covariance matrix or preconditioner.

EXP-009A--C test predictable one-shot mixing certificates. Coverage and exact
safety pass, and participation falls sharply under high sharing. Finite-budget
and joint \((q,b,\eta)\) search beat the worst-mixing baseline in all
low/medium-persistence cells, but the worst high-persistence delayed
online/oracle expected-error ratio remains 10.46. These experiments reject a
uniform near-oracle claim for the static-pilot controller and motivate an
anytime progressive certificate.

EXP-009D implements that time-uniform progressive certificate. Simultaneous
coverage is 99.479%, all covered actions are exactly stable, and persistent
gaps shrink across blocks. The worst online/oracle ratio improves to 7.57 but
still fails the registered threshold five. This rules out a uniform
near-oracle claim under finite-budget high-confidence mixing estimation.

EXP-010A transfers the sharp homogeneous certificate to a seven-state,
four-feature TD family. All numerical gates and four of five scientific gates
pass, with zero divergences in 1,152 charged runs. Correlation reduces the
selected participation in every matched cell, temporal persistence increases
the median certified separation by \(18.36\times\), and all eight primary
artifacts reproduce byte-for-byte. The joint action is never worse than the
better endpoint in mean, but the frozen strict endpoint gate fails because the
joint optimizer selects an endpoint in five cells. The stationary-noise
finite-budget expression remains a controller surrogate until a generic
affine Markov-TD finite-time argument is proved.

EXP-010B supplies that finite-gap affine argument. It does not assume
conditional centering or orthogonality of the TD innovation, and it retains
innovation terms generated by stale-iterate telescoping. The 32-seed audit
passes all nine gates with zero divergences. Every proved bound beats no
update, all one-sided 99% empirical upper means lie below the theorem bound,
and the median bound/mean ratio is 20.77. The theorem-selected action remains
conservative in one fast-mixing endpoint comparison, so empirical near-oracle
participation is not claimed.

EXP-011A supplies the complementary impossibility result. In an exact Gaussian
one-step-mixing Markov subclass, the minimax speedup is
\(q/[1+(q-1)\rho]\), so any positive shared-noise fraction caps the benefit of
additional agents. Under a pathwise resource budget, even predictable adaptive
participation cannot beat the best information-per-cost action. The
deterministic audit passes all nine gates and reproduces all five core
artifacts byte-for-byte. At \(\rho=.9\), 32 agents provide only
\(1.1073\times\) speedup.

## Scope

This first experiment intentionally uses a linear fixed-policy-style model. It
tests whether the proposed mechanism exists before investing in a full
multi-agent temporal-difference or deep reinforcement-learning implementation.
No claim about nonlinear multi-agent reinforcement learning is made here.
The current CPU evidence establishes a controlled linear-TD mechanism and an
algorithmic prototype; nonlinear MARL remains a later GPU-backed validation.
