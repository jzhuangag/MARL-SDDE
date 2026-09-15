# T-064 online joint Lyapunov optimizer audit

## Scope

T-064 is an outcome-free algebraic audit. It produces no RL trajectory,
scientific endpoint, pilot result, seed claim, or experimental authorization.
Its purpose is to determine whether online joint `(q,eta)` selection requires a
heuristic finite catalogue or a general mixed-integer solver.

## Meaning of fully charged observable sensing

An observable sensor is a statistic computed from information available to the
algorithm before the selected action produces its outcome. It may use states,
actions, rewards, TD increments, current parameters, and registered budget
state. It may not use true correlation, the unknown fixed point, future
learning loss, or an outcome-aware oracle.

Fully charged means that sensing has no free side channel. For every sensor
block, the ledger includes:

- every actor transition, including burn-in or separation gaps required by the
  Markov proof;
- every actor-to-server summary and server overhead message;
- elapsed decision blocks and their effect on the delay pipeline;
- the opportunity cost of transitions that cannot also be counted as
  independent learning data.

Only the residual budgets enter the subsequent learning horizon and comparator
calculation. A no-probe strong comparator receives the corresponding full
budget.

## Meaning of two short TD residual trajectories

At a predictable decision time, freeze the current parameter `theta_t` and
collect two disjoint short Markov blocks under the same fixed behavior policy.
For block `j`, compute the average TD update vector

\[
\widehat F_t^{(j)}
=\frac1L\sum_{\ell=1}^{L}
\phi(S_\ell^{(j)})
\{R_\ell^{(j)}+\gamma\phi(S_{\ell+1}^{(j)})^\top\theta_t
-\phi(S_\ell^{(j)})^\top\theta_t\}.
\]

After accounting for regularization and Markov blocking, both blocks have the
same conditional mean `F(theta_t)`. Conditional independence gives

\[
\mathbb E\langle\widehat F_t^{(1)},\widehat F_t^{(2)}\rangle
=\|F(\theta_t)\|^2,
\]

while

\[
\frac12\mathbb E
\|\widehat F_t^{(1)}-\widehat F_t^{(2)}\|^2
=\operatorname{tr}\operatorname{Var}(\widehat F_t^{(1)}).
\]

They are called trajectories because each average is formed from a contiguous
Markov path, not iid replay samples. They are short because sensing is
performed only at separated decision epochs and amortized over many learning
updates. They are fully charged because their transitions and messages are
deducted even though they are not credited as independent learning gradients.

The exact identities require conditional independence or an explicit
cross-covariance correction. Unknown mixing bias and finite-block confidence
bounds remain proof obligations.

## Composite Lyapunov function

The delayed learning state is lifted to

\[
z_t=(e_t,e_{t-1},\ldots,e_{t-D}),
\qquad V_t=z_t^\top Pz_t.
\]

`P` is a positive-definite offline certificate for the entire allowed gain
interval. `V_t` penalizes current error and the stale states that will enter
future updates.

For planned per-block expenditures `bar_c_m,bar_c_e`, resource queues obey

\[
Q_{t+1}^i=[Q_t^i+c_t^i-\bar c_i]_+,
\qquad i\in\{m,e\}.
\]

The composite function

\[
\Phi_t=V_t+\frac{\gamma_m}{2}(Q_t^m)^2
+\frac{\gamma_e}{2}(Q_t^e)^2
\]

has two roles. Its first term certifies learning contraction against delayed
noise. Its queue terms turn scarce messages and actor transitions into online
shadow prices. Minimizing a conservative upper bound on `Delta Phi_t` therefore
balances statistical progress with resource consumption in one decision.

Virtual-queue stability only controls average expenditure. T-064 retains a
separate residual-budget shield for exact finite-budget feasibility. The shield
rejects any action that cannot pay for its current cost, in-flight delay, and
registered fallback.

## Convex joint action

After substituting confidence bounds, the q-dependent robust drift score is

\[
J(q,\eta)=
\left(u+\frac vq\right)\eta^2-r\eta+\lambda q+C,
\]

where `r,u,v,lambda` are nonnegative observable or certified quantities. The
term `eta^2/q` is jointly convex for `q>0`, so the continuous problem is convex.
It is exactly representable as an SOCP through

\[
\eta^2\le zq.
\]

A plain QP is not exact, and making q integer produces a mixed-integer conic
problem. Neither is necessary in the scalar homogeneous-agent model.

For fixed q,

\[
\eta^\star(q)=
\Pi_{[\eta_{\min},\eta_{\max}]}
\frac r{2(u+v/q)}.
\]

This is exact partial minimization. The profiled objective is convex, so after
finding its continuous minimum, checking the neighboring integers recovers the
global integer solution over every feasible q. This removes the heuristic
powers-of-two catalogue without increasing online complexity.

## CPU verification

The implementation is
`experiments/dependence_delay_linear/t064_joint_clf_optimizer.py`. Its targeted
tests verify:

1. the closed-form gain matches a dense gain grid;
2. continuous optimization plus at most two integer evaluations matches full
   enumeration for 1,000 randomized coefficient settings and up to 128 agents;
3. perfect common correlation removes the statistical value of additional
   participation under a positive resource price;
4. increasing the certified noise scale reduces the drift-minimizing gain;
5. invalid confidence and action inputs are rejected.

These are algebraic tests, not evidence that the assumed drift coefficients
are valid for Markov TD. The next proof must derive those coefficients and the
observable confidence bounds from the delayed Markov model.

## Decision

Replace the sparse finite participation catalogue in the proposed ICML
mainline with a convex Lyapunov index controller over all currently feasible
integer participation levels. Retain catalogue scanning only as an
implementation ablation or as a fallback when heterogeneous per-agent costs
destroy the scalar convex structure.
