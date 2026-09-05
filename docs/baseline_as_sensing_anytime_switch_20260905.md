# Baseline-as-sensing anytime likelihood switch

Date: 2026-09-05

Status: **conditional matching upper policy proved for the binary
finite-commit Gaussian common-factor model.  It uses no extra probe, but it is
not a general asynchronous-MARL algorithm.**

## Algorithm

Let (a_0) be the normal all-agent training action.  It is optimal under the
low-correlation regime (H_0), while a different deployment (a_1) is better
under (H_1).  Every normally received (a_0) packet is used twice without
changing its resource charge:

1. update the two public Kalman filters and compute
   \[
   \ell_t=log
   \frac{p_1(Y_t\mid\mathcal F_{t-1},a_0)}
        {p_0(Y_t\mid\mathcal F_{t-1},a_0)};
   \]
2. apply exactly the same (a_0) learning update that the baseline would have
   applied.

Set (S_t=\sum_{k=1}^t\ell_k).  Let (T_B) be the last decision time at which
switching, after charging all messages, environment transitions, in-flight
packets and actuation delay (D), retains positive certified regime-one
deployment value.  For a threshold (h), define

\[
\tau_h=\inf\{t\le T_B:S_t\ge h\}.
\]

If the threshold is reached, commit to (a_1) at the next feasible execution
time; otherwise use (a_0) throughout.  Stale or in-flight (a_0) packets
after detection remain part of the switched branch risk and are not treated
as an instantaneous switch.

## Assumptions

1. (H_0,H_1) are simple hypotheses with known or separately certified
   mixing bounded away from one.  Their conditional Gaussian densities are
   exact and mutually absolutely continuous.  The (a_0) experiment has
   positive information rate.
2. Before switching, the actions and learning updates are samplewise identical
   to the all-agent baseline.  Likelihood computation neither discards nor
   changes a training update.
3. Public finite-budget branch risks satisfy, for every (t\le T_B),
   \[
   R_0^{\rm sw}(t,D)-R_0^{a_0}(B,D)\le W_{0,B},
   \tag{1}
   \]
   \[
   R_1^{\rm sw}(t,D)-R_1^*(B,D)\le d_B+\omega_Bt,
   \tag{2}
   \]
   and the no-switch loss is
   (W_{1,B}=R_1^{a_0}(B,D)-R_1^*(B,D)).  The term (d_B) includes delay and
   in-flight-update loss.
4. For some (s\in(0,1)), the deterministic-(a_0), length-(n) Chernoff
   exponent obeys
   \[
   \mathcal C_n(s)
   :=-\log\int(p_{1,n}^{a_0})^{1-s}(p_{0,n}^{a_0})^s
   \ge nc_s-C_{\rm init},
   \qquad c_s>0.
   \tag{3}
   \]
   In the Gaussian Markov model, (3) can be certified by the exact covariance
   formula or innovation recursion.  It does not assume bounded likelihood
   increments.

## Theorem 1: anytime expected safety

For (0<\epsilon<W_{0,B}), set

\[
h=\log(W_{0,B}/\epsilon).
\tag{4}
\]

Under (H_0), (e^{S_t}) is the exact nonnegative likelihood-ratio
martingale.  Ville's inequality and (1) give

\[
\mathbb P_0(\tau_h\le T_B)\le e^{-h}
=\epsilon/W_{0,B},
\]

\[
\mathbb E_0[R_0^{\rm switch}-R_0^{a_0}]
\le W_{0,B}\mathbb P_0(\tau_h\le T_B)
\le\epsilon.
\tag{5}
\]

This is an anytime statement.  It does not use a fixed-time tail bound in
place of Ville's inequality.  When \(\epsilon=0\), set (h=\infty), so no
nontrivial finite-budget switch is claimed.

## Theorem 2: high-regime opportunity upper bound

Before the threshold is crossed, the observed path is exactly the
deterministic (a_0) experiment.  For fixed (n\le T_B),

\[
\{\tau_h>n\}\subseteq\{S_n<h\}.
\]

Chernoff's method and (3) therefore imply

\[
\mathbb P_1(\tau_h>n)
\le\exp(sh+C_{\rm init}-nc_s).
\tag{6}
\]

Writing

\[
n_h=\left\lceil\frac{sh+C_{\rm init}}{c_s}\right\rceil,
\]

the tail-sum formula gives

\[
\mathbb E_1[\tau_h\wedge T_B]
\le n_h+\frac1{1-e^{-c_s}},
\tag{7}
\]

and

\[
\mathbb P_1(\tau_h>T_B)
\le\exp(sh+C_{\rm init}-T_Bc_s).
\tag{8}
\]

Combining (2), (7), and (8),

\[
\boxed{
\operatorname{Reg}_1
\le d_B+
\omega_B\left[
n_h+\frac1{1-e^{-c_s}}
\right]
+W_{1,B}\exp(sh+C_{\rm init}-T_Bc_s).}
\tag{9}
\]

If terminal risk is of order (B^{-1}),
\(\omega_B=O(B^{-2})\), (d_B=O(D/B^2)\), and the cutoff makes (8) of order
(B^{-2}), (9) is

\[
O\!\left(\frac{D+\log(W_{0,B}/\epsilon)}{B^2}\right).
\]

## Overshoot and block checking

The proof never invokes Wald's identity, so it does not silently assume a
uniform expected overshoot for unbounded Gaussian log-likelihood increments.
Ville's inequality is unaffected by crossing overshoot, and (7) is obtained by
a fixed-(n) Chernoff tail followed by summation.  If likelihood is checked
only once per block, the risk bound must additionally charge one complete
block of resources and opportunity loss.

## Conditional matching to the coupled lower bound

Let \(\mathcal C_B^{\rm coupled}(x,\epsilon)\) be the minimum regime-one
opportunity cost in the coupled occupation program required to accumulate
evidence (x).  The baseline policy is (C)-competitive if, over the relevant
evidence range,

\[
\omega_B
\left\lceil\frac{sx+C_{\rm init}}{c_s}\right\rceil
\le C\mathcal C_B^{\rm coupled}(x,\epsilon)+r_{0,B}.
\tag{10}
\]

A local sufficient condition for a finite action catalogue compares the
baseline's opportunity-to-information ratio with all admissible normal
training actions:

\[
\frac{\omega_B}{c_s}
\le C\inf_a\frac{\omega_{1,B}(a)}{c_s(a)},
\tag{11}
\]

with positively harmful actions evaluated using the Lagrangian ratio

\[
\frac{\omega_{1,B}(a)+\lambda_{\rm safe}s_0(a)}{c_s(a)}.
\]

Under (10), negligible cutoff tail, and the separated deployment assumptions,
the testing lower bound and (9) both scale as

\[
\Theta\!\left(
\frac{\text{opportunity}}{\text{information}}
\log\frac{g_0}{\epsilon}
\right),
\]

up to (C), initialization, delay, and rounding terms.

The competitiveness assumption is necessary.  If another zero-harm normal
training action has the same opportunity loss and arbitrarily larger
information rate, the coupled optimum uses it and baseline sensing can be
arbitrarily suboptimal.

## Complexity and scope

Normal (q_0)-agent aggregation remains (O(q_0d)).  The additional work per
packet is two scalar Kalman updates and two Gaussian log-density evaluations:
(O(1)) arithmetic and memory, plus fixed-delay bookkeeping.  There is no
additional message, environment transition, probe, sample split, QP, or
preconditioner.

The theorem-facing executable prototype is
`experiments/dependence_delay_linear/baseline_as_sensing_switch.py`; focused
tests are in
`experiments/dependence_delay_linear/test_baseline_as_sensing_switch.py`.
The prototype deliberately exposes the sequential likelihood state and bound
calculators only.  It does not mutate a frozen experiment runner or generate a
scientific outcome.

Together with the learning-to-testing reduction and coupled occupation
frontier, this establishes a coherent binary finite-commit
information--safety--learning theorem.  Its novelty is the combined frontier,
not sequential likelihood testing alone.  Exact binary regimes,
known/certified mixing, low-regime baseline optimality, public branch-risk
sensitivity, and information/opportunity competitiveness are substantial
restrictions.

It does not establish policy-gradient convergence, strategic interaction,
continuously interleaved asynchronous updates, or a standard nonlinear MARL
result.  A broader ICML claim remains unauthorized unless an outcome-free
standard Markov-learning audit verifies (10) and at least 10% equal-resource
headroom before any new benchmark experiment.
