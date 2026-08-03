# T-050 stationary participation limit and break-even theorem

## Decision boundary

T-050 explains the T-049A failure without altering its population or gates.
It separates two questions that the earlier controller line had conflated:

1. can a time-varying participation schedule improve a stationary problem;
2. can a paid correlation certificate select a better *fixed* participation
   level across instances and still amortize its cost?

The first question has a sharp negative answer in the independent Gaussian
subclass and only a lower-order packing benefit at a long horizon. The second
has a positive, explicit break-even condition. This preserves a possible
ICML-level fixed-participation adaptation thesis, but it stops the
within-trajectory schedule thesis in a stationary homogeneous regime.

## Theorem 1: stationary independent schedules have only packing value

For an independent Gaussian round, the server observes the average of (q)
agents with variance

\[
 v(q)=\sigma_c^2+\frac{\sigma_e^2}{q},\qquad
 c(q)=h+q.
\]

Its Fisher information for the shared mean is (i(q)=1/v(q)). For a finite
catalogue \({\cal Q}\), let

\[
 e^\star=\max_{q\in{\cal Q}}\frac{i(q)}{c(q)},
 \qquad q_e\in\arg\max_q i(q)/c(q).
\]

Every deterministic schedule satisfying \(\sum_t c(q_t)\le B\) obeys

\[
 I(q_{1:T})=\sum_t i(q_t)\le B e^\star.                 \tag{1}
\]

The predictable-policy argument of T-038 makes the same lower bound valid
for data-dependent randomized schedules: conditional posterior variance is a
function of the realized design, so random selection cannot beat the best
feasible deterministic sequence in minimax risk.

Let (R_{\rm fix}) be the risk of the best repeated fixed action and
(R_{\rm seq}) the risk of the best arbitrary feasible sequence. Repeating
(q_e) gives

\[
 \frac{R_{\rm fix}-R_{\rm seq}}{R_{\rm fix}}
 \le 1-\frac{\lfloor B/c(q_e)\rfloor c(q_e)}{B}
 <\frac{c(q_e)}B.                                      \tag{2}
\]

Thus any scheduled improvement is a finite-budget remainder effect. If the
budget contains at least 20 efficiency-optimal rounds, even the most
favourable arbitrary schedule has less than a 5% improvement ceiling over
the best fixed action. If (B) is divisible by (c(q_e)), the bound is zero
and repeating (q_e) is exactly optimal.

For continuous (q), minimizing (c(q)v(q)) yields

\[
 q_e^\star=\sqrt{\frac{h\sigma_e^2}{\sigma_c^2}},       \tag{3}
\]

clipped to the permitted interval. Equations (1)--(3) are exact, not an
asymptotic effective-sample-size argument.

## Theorem 2: fixed-q delayed PR phase at stationarity

Let the centered Markov innovation have absolutely summable matrix lag
covariances (K_k), with long-run covariance

\[
 \Gamma=\sum_{k=-\infty}^{\infty}K_k.
\]

Consider a stable delayed linear recursion with drift (A), fixed bounded
delay (D), and a Polyak--Ruppert readout. Under fixed prefix participation
(q), the trajectory-switch construction of T-049 multiplies every lag by

\[
 g(q,\rho)=\rho+\frac{1-\rho}{q}.
\]

The zero-frequency transfer of the stable delayed recursion is (A^{-1}):
the delay factor equals one at frequency zero. Hence, for risk matrix (Q),

\[
 N\,\mathbb E\|\bar e_N\|_Q^2
 \longrightarrow
 g(q,\rho)\,C_{\rm task},\qquad
 C_{\rm task}=\operatorname{tr}
 (Q A^{-1}\Gamma A^{-\mathsf T}).                       \tag{4}
\]

With message budget (B), (N=B/(h+q)+O(1)), so

\[
 B\,\mathbb E\|\bar e_N\|_Q^2
 \longrightarrow C_{\rm task}(h+q)
 \left(\rho+\frac{1-\rho}{q}\right).                  \tag{5}
\]

The task and any fixed stable delay scale the risk but do not change the
leading participation phase. Its continuous optimum is

\[
 q_{\rm PR}^\star(\rho)
 =\sqrt{\frac{h(1-\rho)}{\rho}},                        \tag{6}
\]

again clipped to the action interval. Equation (6) is decreasing in
correlation and provides the mechanism for adapting a fixed learning action
across correlation instances.

The finite-state implementation computes \(\Gamma\) exactly. If (F_{su})
is the edge-gradient first moment and (m_s=\sum_uF_{su}), then with
(Z=(I-P+\mathbf1\pi^\top)^{-1}),

\[
 \sum_{k\ge1}K_k
 =\left[\sum_{s,u}\pi_sF_{su}(Zm)_u^\top\right]^\top.  \tag{7}
\]

This is checked against direct lag summation in the CPU tests.

Equation (4) is a fixed-(q) asymptotic statement. It does **not** claim
that a fixed action dominates every finite Markov schedule. T-038 correctly
leaves open nonconstant deterministic designs when a stride or covariance
state changes.

## Theorem 3: exact leading-order probe break-even threshold

Fix an instance and let (K_0) be the leading coefficient of the public
strong fixed baseline, (K_\star) that of the oracle action, and (K_w) a
valid coefficient bound after a wrong decision. Suppose a fully charged
independent probe costs (C_p) message units and the decision error is at
most \(\alpha\). Define

\[
 K_{\rm ad}=(1-\alpha)K_\star+\alpha K_w.
\]

Within the leading law, paid classify-then-commit beats the baseline exactly
when

\[
 \frac{K_{\rm ad}}{B-C_p}<\frac{K_0}{B}.
\]

If (K_{\rm ad}<K_0), this is equivalent to

\[
 B>B_{\rm BE}:=\frac{C_pK_0}{K_0-K_{\rm ad}}.           \tag{8}
\]

If (K_{\rm ad}\ge K_0), the certified threshold is infinite: more horizon
cannot rescue a classifier whose error-weighted action coefficient is not
better than the baseline. Dual budgets, integer horizons, delay, and exact
finite-time remainders are handled by the T-048 finite-table certificate;
(8) is the interpretable leading-order threshold and a prospective design
rule.

## Consequence for T-049A

T-049A used only 96 or 192 reference updates. Its stable lifted recursions
have spectral radii close to 0.9965. A half-horizon PR burn-in therefore does
not place all tasks deep in the noise-dominated regime. The exact scan found:

- the correlation direction in all adjacent comparisons;
- only 2.0754% no-probe fixed-action oracle value;
- no nonconstant-schedule improvement in any of 252 cells;
- a 0.9880% loss after the registered probe was charged.

This pattern is consistent with (2), (5), and (8): short-horizon transient
risk hides the stationary participation phase, and the small remaining gap
cannot pay for the probe. It is not evidence that the exact covariance bridge
or correlation phase has the wrong sign.

## Authorized continuation

The next experiment, if any, must have a new identifier and select its
horizon before outcomes using a public contraction target and (8). It may
test an independent-probe, commit-to-fixed-(q) controller on the same
unchanged Gymnasium task marginals. It may not revive the stopped
time-varying EXP-021A design. A CPU static oracle gate must first demonstrate
at least 5% full-cost value and broad directional support; GPU remains
unauthorized until that gate and a sampled CPU pilot both pass.
