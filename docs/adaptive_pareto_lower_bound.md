# T-016: instance-dependent adaptive Pareto lower bound

## Instances and opportunity loss

Take \(H_0:\theta=\theta_0\) and \(H_1:\theta=\theta_1\), with public
\(\lambda\), such that their finite-budget oracle actions differ:
\(a_0^\star\ne a_1^\star\). For action \(a=(q,b,\eta)\), let
\(\Delta_j(z,a)\ge0\) be its one-step opportunity loss relative to the
instance-\(j\) oracle at controlled Kalman belief \(z\). It may include the
loss of an optimization update, message consumption, stride consumption, and
the reduction in usable decisions caused by delay. It is not assumed that
all-agent play is uninformative.

Indeed, in the registered individual-feedback model all-agent observations
have positive KL when \(\theta_0\ne\theta_1\). The lower bound prices how long
safe or otherwise chosen actions take to identify the regime; it does not
assert a false no-harm impossibility.

## Theorem 1: adaptive information constraint

If a stopped policy returns \(\widehat J\in\{0,1\}\) and both directional
errors are at most \(\delta<1/2\), then for \(j\ne k\)

\[
\mathbb E_j^\pi\sum_{t\le\tau}{\cal I}_{jk}(Z_t,A_t)
\ge d_\delta:=\operatorname{kl}(1-\delta,\delta).
\tag{3}
\]

This follows from AC-7 and data processing through the decision bit. Both
directions are required.

Let the expected stopped occupation measure under instance \(j\) be

\[
\nu_j(G,a)=\mathbb E_j^\pi
\sum_{t\le\tau}{\bf1}\{Z_t\in G,A_t=a\}.
\]

Every \(\delta\)-correct policy induces a controlled occupation measure
satisfying the Kalman flow constraints and

\[
\begin{aligned}
&\int {\cal I}_{jk}(z,a)\,\nu_j(dz,da)\ge d_\delta,\\
&\int(h+q(a))\,\nu_j(dz,da)\le B_{\rm msg},\\
&\int b(a)\,\nu_j(dz,da)+D\,P_j(\tau>0)\le B_{\rm env}.
\end{aligned}
\tag{4}
\]

Consequently its identification opportunity cost obeys

\[
\mathbb E_j^\pi\sum_{t\le\tau}\Delta_j(Z_t,A_t)
\ge
\inf_{\nu\in{\cal O}_j}
\left\{\int\Delta_j(z,a)\nu(dz,da):
\int{\cal I}_{jk}(z,a)\nu(dz,da)\ge d_\delta,\ (4)\right\}.
\tag{5}
\]

Here \({\cal O}_j\) is the set of nonnegative occupation measures satisfying
the controlled Kalman flow and stopping constraints. Equation (5) is the
Markov, participation-, stride-, and cost-dependent analogue of the familiar
allocation lower bound. It is an exact relaxation of policy performance, not
an iid action-count fiction.

## Corollary: high-regime identification price

Suppose the policy is restricted to actions in
\({\cal A}_{\rm safe}\) during identification, and on \(H_1\)

\[
\Delta_1(z,a)\ge\underline\Delta_1>0
\quad\text{for }a\in{\cal A}_{\rm safe}
\]

until the decision threshold is crossed. If their conditional information
is bounded by
\(\overline I_{10}^{\rm safe}
=\sup_{z,a\in{\cal A}_{\rm safe}}{\cal I}_{10}(z,a)<\infty\), then

\[
\mathbb E_1[\text{identification opportunity cost}]
\ge
\underline\Delta_1\,
\frac{d_\delta}{\overline I_{10}^{\rm safe}},
\tag{6}
\]

subject also to the two resource feasibility inequalities in (4). A sharper
value is the occupation program (5). Formula (6) explicitly quantifies the
high-correlation price of remaining in low-instance-safe actions while still
allowing the all-agent baseline to provide information.

For a fixed block or a reset construction, the belief flow is deterministic
and (5) reduces to an action-count program
\(\inf_x\sum_a x_a\Delta_j(a)\) with exact block KL. For general irregular
adaptive gaps it does not: the mean-separation term in AC-7 is
history-dependent. Dropping the flow constraint would produce only a
computable relaxation, not an exact Markov theorem.

## Pareto interpretation

Varying the permitted low-instance safety slack changes
\({\cal A}_{\rm safe}\) and therefore the information/opportunity optimum in
(5). The resulting frontier has three genuine coordinates:

1. identification opportunity cost and oracle regret;
2. safety deficit relative to all-agent;
3. message/environment expenditure, with \(D\) reducing usable horizon.

This is the defensible object left by the EXP-015A result. The observed
fallback `0.777778` is not changed, rounded, or used as a theorem target.
