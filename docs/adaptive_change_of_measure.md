# T-016: adaptive change of measure

## Scope

This note closes AC-7 for the registered stationary Gaussian common-factor
model with known \(\lambda\). It allows predictable randomized
\(A_t=(q_t,b_t,\eta_t)\), changing observation dimension, irregular gaps,
selection, stopping, dual budgets, and delayed usability. It does not claim
unknown-mixing coverage or an SDDE-to-discrete approximation.

Let \(S_t\) be physical observation time, with
\(S_t-S_{t-1}=b_t\), and under \(H_j\)

\[
C_{S_t}=\lambda^{b_t}C_{S_{t-1}}
 \sqrt{1-\lambda^{2b_t}}\,\xi_{j,t},\qquad
\xi_{j,t}\sim N(0,\theta_j).
\]

The controller observes
\(X_{i,t}=C_{S_t}+\epsilon_{i,t}\), \(i\le q_t\), with unit private
variance. Its random action kernel is the same under both hypotheses and is
measurable with respect to the pre-action history.

## Lemma 1: dimension-changing spatial reduction

Conditionally on the history and action, rotate \(X_t\) into

\[
Y_t=q_t^{-1/2}{\bf1}^{\mathsf T}X_t
=\sqrt{q_t}C_{S_t}+E_t
\]

and \(q_t-1\) orthogonal contrasts. The contrasts are standard normal under
both hypotheses, even when \(q_t\) changes. They contribute zero to the
likelihood ratio. Thus the scalar \(Y_t\), together with the realized action,
is sufficient for the hypothesis likelihood. This statement also covers
\(q_t=1\), where there are no contrasts.

Statistically, \(q=1\) contains variance information when private variance is
known. It is nevertheless not a cross-agent correlation certificate; project
algorithms keep the registered \(q\ge2\) rule for that interpretation.

## Kalman innovation recursion

Before observation \(t\), let the filter under \(H_j\) have
\(C_{S_t}\mid{\cal H}_{t-1},A_t\sim N(m^-_{j,t},P^-_{j,t})\). Initially,
\(m^-_{j,1}=0,P^-_{j,1}=\theta_j\). For \(t>1\),

\[
\begin{aligned}
m^-_{j,t}&=\lambda^{b_t}m^+_{j,t-1},\\
P^-_{j,t}&=\lambda^{2b_t}P^+_{j,t-1}
 (1-\lambda^{2b_t})\theta_j .
\end{aligned}
\]

With \(r_{j,t}=Y_t-\sqrt{q_t}m^-_{j,t}\) and
\(V_{j,t}=1+q_tP^-_{j,t}\),

\[
\ell_{j,t}=-\frac12\{\log(2\pi V_{j,t})+r_{j,t}^2/V_{j,t}\},
\]

\[
K_{j,t}=\frac{P^-_{j,t}\sqrt{q_t}}{V_{j,t}},\quad
m^+_{j,t}=m^-_{j,t}+K_{j,t}r_{j,t},\quad
P^+_{j,t}=P^-_{j,t}-K_{j,t}\sqrt{q_t}P^-_{j,t}.
\]

The two filters use the same realized data and actions but generally have
different means and variances. The exact stopped log likelihood ratio is

\[
\log L_\tau^{01}=\sum_{t=1}^{\tau}(\ell_{0,t}-\ell_{1,t}).
\tag{AC7-LR}
\]

## Theorem AC-7: stopped adaptive chain rule

Assume:

1. \(\theta_j\ge0\), \(0\le\lambda<1\), unit known private variance, and a
   stationary initial common factor;
2. \(A_t\) is selected by a common, possibly randomized, predictable kernel;
3. \(q_t,b_t\) are finite positive integers and \(\eta_t\) does not change
   the observation kernel;
4. \(\tau\) is a stopping time bounded by the pathwise dual budgets, or is a
   localizable stopping time for which the stopped log likelihood ratios are
   uniformly integrable.

Then

\[
\operatorname{KL}(P_0^\pi|{\cal H}_\tau\Vert
P_1^\pi|{\cal H}_\tau)
=
\mathbb E_0^\pi\sum_{t=1}^{\tau}{\cal I}_{01}(Z_t,A_t),
\tag{AC7}
\]

where \(Z_t=(m^-_{0,t},P^-_{0,t},m^-_{1,t},P^-_{1,t})\) and

\[
{\cal I}_{01}(Z_t,A_t)
=\frac12\left[
\log\frac{V_{1,t}}{V_{0,t}}
+\frac{V_{0,t}+q_t(m^-_{0,t}-m^-_{1,t})^2}{V_{1,t}}-1
\right].
\tag{1}
\]

The reverse identity holds after exchanging 0 and 1.

**Proof.** Factor the stopped path law into the initial law, common action
kernels, and conditional observation kernels. The action-kernel factors
cancel pointwise, including when the kernel selected \(q_t,b_t\) from earlier
observations. Lemma 1 reduces each observation ratio to the scalar innovation
ratio (AC7-LR). Conditional expectation under \(P_0\) gives the Gaussian KL
(1), including the posterior-mean difference. Tonelli applies to the
nonnegative conditional KL terms for bounded \(\tau\). Localization and
uniform integrability give the stated unbounded extension. The reverse
direction is identical. \(\square\)

Selection bias is therefore not ignored: it changes the distribution of
\(Z_t,A_t\) inside the expectation. It simply does not create a separate
hypothesis-dependent action likelihood.

## Dual budgets and delay

For overhead \(h\), a stopped probe path is feasible only if

\[
\sum_{t\le\tau}(h+q_t)\le B_{\rm msg},\qquad
\sum_{t\le\tau}b_t+D{\bf1}\{\tau>0\}\le B_{\rm env}.
\tag{2}
\]

If \(N\) optimization updates are scheduled after probing, only
\([N-D]_+\) can be usable under the registered delay convention. Delay does
not alter the likelihood of received observations; it restricts the stopped
experiment and the remaining commit horizon. Thus \(D\) enters AC-7 through
feasibility and opportunity cost, not through an invented covariance term.

The exact expected information is the innovation-information functional in
(AC7). There is generally no history-free number \(I(q,b)\) whose sum equals
it: \(P^\pm_{j,t}\) and the mean-separation term retain the entire adaptive
gap history. The controlled-belief occupation formulation in
`adaptive_pareto_lower_bound.md` preserves this dependence.

## Verification

`test_adaptive_change_of_measure.py` checks the conditional formula at
70-digit precision, innovation likelihood against a brute-force dense
Gaussian likelihood for irregular dimension-changing paths, a
selection-dependent action rule, both KL directions, and the likelihood-ratio
martingale identity. Boundary checks cover \(q=1,\lambda=0,\lambda=1\) as a
limit, exact dual-budget exhaustion, and \(D\) beyond the usable horizon.
