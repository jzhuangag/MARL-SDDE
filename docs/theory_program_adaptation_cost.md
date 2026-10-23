# Theory program: the statistical cost of adaptive participation

## Status and claim boundary

Working title: **Beyond Linear Speedup: The Statistical Cost of Adaptive
Participation under Correlated Markov Data**.

This document contains a proved fixed-design Gaussian lower-bound route, a
proved fixed-design testing upper bound, and a horizon-aware paid
explore-then-commit algorithm. T-016 additionally proves the adaptive
known-mixing change of measure and controlled-belief Pareto lower bound in
`adaptive_change_of_measure.md` and `adaptive_pareto_lower_bound.md`.
The theorem-derived fallback guarantees are in
`theorem_derived_fallback.md`. T-017 closes unrestricted unknown mixing
negatively, proves a finite-budget threshold sandwich on a compact separated
class, and shows unrestricted uniform matching impossible; see
`unknown_mixing_impossibility.md`, `adaptation_threshold_sandwich.md`, and
`ac9_uniform_matching_audit.md`. Matching the entire adaptive
controlled-belief optimum remains open.

EXP-014A and EXP-014B remain honest pilot failures. In particular, EXP-014B's
strict controller fell back in every cell because its time-uniform interval
remained `[0,1]`. The argument below does not reinterpret that outcome as a
numerical or task-heterogeneity failure.

## Observable-feedback model

There are at most \(Q\) agents and dual budgets
\((B_{\rm msg},B_{\rm env})\). At a decision time the controller chooses

\[
a=(q,b,\eta),\qquad
c_{\rm msg}(a)=h+q,\qquad c_{\rm env}(a)=b.
\]

An update issued at time \(t\) becomes usable after delay \(D\). Thus a
fixed action has at most

\[
N_D(a)=
\left[
\min\left\{
\left\lfloor\frac{B_{\rm msg}}{h+q}\right\rfloor,
\left\lfloor\frac{B_{\rm env}}b\right\rfloor
\right\}-D
\right]_+
\]

completed updates. The EXP-015A mechanism pilot treats \(\eta\) as fixed and
studies the information/participation decision \((q,b)\); an existing
discrete delayed-stability screen can be intersected with this action set.

At probe \(k\), separated by \(b\) environment transitions, the controller
sees **individual** observations

\[
X_{i,k}=C_k+\epsilon_{i,k},\quad i=1,\ldots,q,
\]

where

\[
C_{k+1}=\lambda^b C_k+
\sqrt{1-\lambda^{2b}}\,\xi_{k+1},\quad
\xi_k\sim N(0,\theta),\quad
\epsilon_{i,k}\stackrel{\rm iid}{\sim}N(0,1).
\]

The stationary common factor has variance \(\theta\), and the same-time
agent correlation is

\[
\rho=\frac{\theta}{1+\theta}.
\]

The simulator's \(\theta\), regime label, oracle action, and common factor are
hidden. Only the registered simple hypotheses \(\theta_0,\theta_1\), public
\(\lambda\) in the mechanism pilot, chosen actions, and individual samples
are visible. A `q=1` action is prohibited from issuing a cross-agent
correlation certificate. General unknown-\(\lambda\) adaptation remains an
explicit obligation.

The total-variance speedup is exactly

\[
S(q,\rho)=
\frac{q}{1+(q-1)\rho}.
\]

Under one scalarized round-cost budget, the stationary information risk is
proportional to

\[
(h+q)\left(\theta+\frac1q\right).
\]

Its continuous minimizer is

\[
q^\star_{\rm cont}=\sqrt{\frac h\theta}
=\sqrt{\frac{h(1-\rho)}{\rho}},
\]

which is precisely the existing Theorem-5 participation transition.

## Exact fixed-design information

Let \(n\) probes use fixed \((q,b)\), set \(a=\lambda^b\), and let
\(R_n(a)=(a^{|s-t|})_{s,t=1}^n\).

### Lemma 1: spatial reduction — proved

Rotate each \(q\)-agent observation into the normalized all-ones direction
and \(q-1\) orthogonal contrasts. The contrasts are iid \(N(0,1)\) under
both hypotheses. The sufficient common direction

\[
Y_k=q^{-1/2}\sum_{i=1}^q X_{i,k}
\]

has covariance

\[
A_\theta(q,b,n)=I_n+q\theta R_n(\lambda^b).
\]

Therefore all likelihood information about \(\theta\) lies in \(Y_{1:n}\).

### Theorem 1: exact probe KL — proved

For \(H_j:\theta=\theta_j\),

\[
\operatorname{KL}(P_{\theta_0}^{q,b,n}\Vert
P_{\theta_1}^{q,b,n})
=
\frac12\left[
\operatorname{tr}(A_{\theta_1}^{-1}A_{\theta_0})
-n-\log\det(A_{\theta_1}^{-1}A_{\theta_0})
\right].
\]

If \(r_1,\ldots,r_n\) are the eigenvalues of \(R_n(\lambda^b)\), this is

\[
\frac12\sum_{\ell=1}^n
\left[
\frac{1+q\theta_0 r_\ell}{1+q\theta_1 r_\ell}
-1-\log
\frac{1+q\theta_0 r_\ell}{1+q\theta_1 r_\ell}
\right].
\]

This formula is exact and numerically matches the dense covariance formula.
Unlike a generic two-arm mean test, \(q\) changes the covariance experiment,
\(b\) changes \(\lambda^b\), and both change the dual resource cost.

For small \(\Delta=\theta_1-\theta_0\), the local information is

\[
\operatorname{KL}
=
\frac{q^2\Delta^2}{4}
\operatorname{tr}
\left[
\{A_{\theta_0}^{-1}R_n(\lambda^b)\}^2
\right]+O(\Delta^3).
\]

When \(\lambda^b\) is close to one, the eigenvalue mass concentrates and
information fails to accumulate as \(n\) independent marginal probes.
Increasing \(b\) restores effective rank but consumes environment budget.

### Theorem 2: fixed-design identification threshold — proved

For any test whose two directional errors are at most \(\delta<1/2\), data
processing for the decision bit gives the necessary conditions

\[
\operatorname{KL}(P_0\Vert P_1)\ge
\operatorname{kl}(1-\delta,\delta),\qquad
\operatorname{KL}(P_1\Vert P_0)\ge
\operatorname{kl}(1-\delta,\delta).
\]

Define \(n_{\rm LB}(q,b)\) as the first \(n\) satisfying both inequalities.
Then a fixed design is impossible unless

\[
B_{\rm msg}\ge n_{\rm LB}(h+q),\qquad
B_{\rm env}\ge n_{\rm LB}b+D.
\]

The delay does not alter the likelihood of an already received fixed block,
but it reduces the number of decisions that can affect the remaining
optimization horizon.

## Paid exploration and safety tradeoff

The zero-mean Gaussian Bhattacharyya distance is exactly

\[
\mathcal B_n(q,b)
=
\frac12\log\det\frac{A_{\theta_0}+A_{\theta_1}}2
-\frac14\log\det A_{\theta_0}
-\frac14\log\det A_{\theta_1}.
\]

The equal-prior likelihood-ratio test has average error at most
\(\frac12e^{-\mathcal B_n}\). Let \(n_{\rm UB}(q,b)\) be the first \(n\)
with

\[
\mathcal B_n(q,b)\ge\log\frac1{2\delta}.
\]

This yields a finite fixed-design upper bound. EXP-015A selects the feasible
design minimizing the larger fraction of message and environment budgets
consumed by \(n_{\rm UB}\), while reserving a minimum commit horizon.

### Algorithm 1: horizon-aware paid ETC

1. Enumerate \(q\ge2\) and stability-feasible strides \(b\).
2. Compute \(n_{\rm LB}\), \(n_{\rm UB}\), and both exact costs.
3. If no design leaves the registered minimum commit horizon, execute the
   all-agent baseline without probing.
4. Otherwise execute the selected paid probe block.
5. Apply the exact Gaussian likelihood-ratio test using only observed
   individual samples.
6. Commit to the Theorem-5 action optimized for the selected hypothesis and
   the remaining dual budget.

For a high-correlation instance, the expected excess risk has the schematic
bound

\[
\mathcal R_{\rm ETC}
\le
\underbrace{\mathcal C_{\rm lost}
(n_{\rm UB},q,b,D)}_{\text{paid identification horizon}}
+
\underbrace{\delta\,\Delta_{\rm commit}
N_{\rm rem}}_{\text{wrong-commit term}},
\]

where every cost in \(\mathcal C_{\rm lost}\) is evaluated by the exact
finite-budget Gaussian mean risk. The bound is computable rather than
asymptotic.

### Proposition 1: what strict no-harm does and does not forbid — proved

With individual all-agent feedback, the all-agent baseline itself has
positive KL between \(H_0\) and \(H_1\). Therefore strict baseline play during
identification and later nontrivial adaptation are not logically
incompatible. A universal theorem claiming otherwise would be false in this
feedback model.

What is unavoidable is instance-dependent opportunity cost: until the
likelihood reaches the identification threshold, an algorithm that stays
safe on the low-correlation instance can spend that horizon using an action
that is suboptimal on the high-correlation instance. Cheaper small-\(q\)
probes may improve information per cost but can create positive low-regime
safety deficit. The relevant object is therefore a Pareto frontier among
oracle regret, probe cost, and safety deficit—not free strict improvement in
every cell.

## Relation to prior project theorems

- Theorem 5 supplies the exact action switch that defines the two regimes.
- Theorems 6--8 supply predictable time-uniform safety certificates but do
  not price the information needed to make them nonvacuous.
- EXP-014B demonstrates the extreme point of the Pareto frontier: zero
  deliberate safety deficit, zero successful identification, and zero gain.
- The exact KL above prices the probe phase needed to move away from that
  point.
- The existing discrete Lyapunov/finite-gap stability theorem can filter
  \((b,\eta,D)\); no nonlinear global-convergence claim is made.

## Autonomous theory decision

T-017's mandatory primary-literature confrontation rejects generic
controlled-sensing, change-of-measure, covariance-adaptive querying, and
multi-agent BAI machinery as novelty. The retained non-generic claim is the
coupled learning-value threshold: correlation, Markov persistence, stride,
participation-dependent information and terminal risk, both costs, delay,
wrong-oracle regret, and safety all enter. The unrestricted unknown-mixing
route is impossible. Current status is therefore **separated-class threshold
theory retained; full adaptive occupation matching remains open**.
