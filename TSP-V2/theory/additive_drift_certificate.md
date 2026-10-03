# Additive Lyapunov-Drift Certificate for Finite-Budget Participation

## 1. Certified design variable

Let `V_k` be an observable validation Lyapunov risk satisfying
`0 <= V_k <= 1`.
For participation action `q`, define the threshold hitting time

\[
\tau_q(v)=\inf\{k\ge 0:V_k\le v\}.
\]

Write `X_k=(V_k-v)_+` for the nonnegative excess risk.
Assume that, before hitting the registered target `v`, the fixed action has a
uniform additive drift in this stopped excess process,

\[
\mathbb E[X_k-X_{k+1}\mid\mathcal F_k,q]\ge d_q>0,
\qquad k<\tau_q(v).
\tag{1}
\]

After all calibration costs are charged, the exact dual-budget horizon is

\[
N_B(q)=\min\!\left\{
\left\lfloor B_m^{\rm rem}/c_m(q)\right\rfloor,
\left\lfloor B_e^{\rm rem}/c_e(q)\right\rfloor
\right\}.
\tag{2}
\]

The proposed controller selects the participation action; it does not alter
the policy architecture, optimizer, physical budgets, or validation target.

## 2. Simultaneous lower certificate

Calibration provides paired, disjoint validation observations
`(V_{q,s}^{before},V_{q,s}^{after})` in `[0,1]^2`.
Let `n_eff(q)` be a preregistered lower bound on the effective number of pairs
after temporal and cross-worker dependence, and let `b_mix(q)` be the
registered residual mixing-bias allowance.
For a catalogue of size `K` and family-wise confidence `1-delta`, define

\[
\underline d_q=
\frac{1}{n_q}\sum_{s=1}^{n_q}
\left(V_{q,s}^{before}-V_{q,s}^{after}\right)
-\sqrt{\frac{2\log(K/\delta)}{n_{\rm eff}(q)}}
-b_{\rm mix}(q).
\tag{3}
\]

Because each paired difference lies in `[-1,1]`, the square-root term is the
Hoeffding radius for range length two.
Under the registered blocking/mixing coupling, the final bias term converts
the independent-block bound to the Markov sample law.
A union bound makes (3) simultaneous over the catalogue.

The executable score is

\[
S_B(q)=N_B(q)[\underline d_q]_+,
\qquad
\widehat q\in\arg\max_{q\in\mathcal Q} S_B(q).
\tag{4}
\]

If every lower certificate is nonpositive, the controller uses a public
fallback rather than claiming unsupported adaptation.
Equation (4) is `O(|Q|)` and contains no Hessian, covariance matrix, or QP.

## 3. Finite-budget hitting-time guarantee

**Theorem 1 (Lyapunov-Drift Participation Rule).**
Suppose (a) `V_k` is adapted and lies in `[0,1]`, (b) condition (1) holds
uniformly for each fixed catalogue action until its threshold hitting time,
(c) all calibration and validation costs are charged before (2), and (d) the
event `d_q >= underline d_q` holds simultaneously.
Then every action with `underline d_q>0` satisfies

\[
\mathbb E[\tau_q(v)]
\le \frac{V_0-v}{\underline d_q},
\qquad
\Pr\{\tau_q(v)>N_B(q)\}
\le
\frac{V_0-v}{N_B(q)\underline d_q}.
\tag{5}
\]

Consequently, (4) minimizes the upper bound in (5) over all positively
certified catalogue actions.

**Proof.**
Apply the additive-drift theorem to the stopped nonnegative excess process
`X_{k wedge tau_q(v)}` and replace `d_q` by its simultaneous lower bound.
This gives the expectation inequality in (5).
Markov's inequality applied to the nonnegative hitting time gives the second
inequality.
The numerator is action independent, so minimizing the probability bound is
equivalent to maximizing `N_B(q) underline d_q`.

The theorem explains the design role of the Lyapunov function: it turns
observed learning progress and exact resource horizons into the online action
score, rather than serving only as a post-hoc stability proof.

## 4. MARL proof interface

The generic theorem is complete only after two MARL-specific statements are
established for the chosen validation risk.
First, the paired validation construction must justify the effective-sample
and mixing-bias inputs in (3).
Second, the calibrated drift must transfer uniformly to the post-calibration
horizon above the target threshold.

The next development experiment is therefore not a return benchmark.
It must use fresh seeds to test whether a bounded, disjoint validation risk can
produce nontrivial lower drift certificates under full cost accounting.
Only after that interface passes may a return-based controller experiment be
preregistered.
