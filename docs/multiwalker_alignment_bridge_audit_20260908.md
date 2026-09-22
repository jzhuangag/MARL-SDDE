## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory--experiment alignment and algorithm-interface audit
- Origin Date: 2026-09-08
- Verification Status: CAUSAL OBSERVATION/DRIFT INTERFACE CLOSED; MULTIWALKER ALIGNMENT HEADROOM OPEN
- Version Label: multiwalker_alignment_bridge_v1

# From policy-profile value to learnable Lyapunov drift in Multiwalker

## Audit decision

The independently confirmed exact oracle answers a necessary question: a
dynamic policy-cache schedule has reproducible finite-horizon return value
above a strong online envelope under the declared prefix communication budget.
It does **not** answer the distinct question used by the core convergence
theorem: whether refreshing an edge improves the alignment of the returned
owner-gradient packet with the current cooperative objective gradient.

These objects are related but not equal. For launch `p` and edge action `a`,

\[
 R_p^H(a)=\mathbb E[\text{discounted profile return}\mid\mathcal F_{b(p)},a]
\]

is the object used by the exact Multiwalker qualification, whereas

\[
 A_p(a)=\mathbb E[\langle\nabla_iF(\theta_{b(p)}),g_p(a)\rangle
 \mid\mathcal F_{b(p)}]
\tag{1}
\]

is the signed packet value in the topology-robust Lyapunov drift. A positive
gap in `R_p^H` does not mathematically imply a positive gap in (1). The oracle
result is therefore retained as a benchmark-value certificate, while an
alignment-oracle gate is required before fitting an online controller.

## Observable delayed feedback

At launch, owner `i` snapshots a reference direction `v_(i,p)` computed only
from packets completed before launch. For the selected cache action, the fully
charged rollout later returns gradient packet `g_p(a_p)`. The scalar target

\[
 Y_p=\langle v_{i,p},g_p(a_p)\rangle
\tag{2}
\]

is then observable without an extra environment trajectory. Only the selected
action produces a target; unselected counterfactual gradients are unavailable
to the online method. The launch snapshot, selected feature, packet identity,
and declared receipt event are enforced by
`alignment_feedback_ledger.py`. A future reference direction cannot be
substituted retroactively.

Suppose `||nabla_i F(theta_b)-v_(i,p)||<=e_(i,p)` and
`||g_p(a)||<=G_p(a)`. Cauchy--Schwarz gives

\[
 \langle\nabla_iF(\theta_b),g_p(a)\rangle
 \ge Y_p-e_{i,p}G_p(a).
\tag{3}
\]

If the owner block moves by at most `M_p` before receipt and is `L_i` smooth,
then the receipt update `theta_i^+=theta_i-alpha g_p(a)` obeys

\[
 \Delta F
 \le-\alpha\{Y_p-G_p(a)[e_{i,p}+L_iM_p]\}
 +\frac{L_iG_p(a)^2}{2}\alpha^2.
\tag{4}
\]

Equation (4) is the exact bridge from an observable delayed packet target to
the core Lyapunov index. It shows separately what must be learned (`Y_p`),
what must be bounded (reference error, packet norm, receipt motion), and what
is optimized (edge and packet weight).

## Delayed optimistic instantiation

Let fixed launch-measurable features satisfy

\[
 \mathbb E[Y_p\mid\mathcal F_{b(p)},a]
 =x_p(a)^\top w_*+b_p(a),\qquad |b_p(a)|\le\epsilon_p^{app}(a).
\tag{5}
\]

The existing delayed ridge state uses only completed selected packets. Under
conditionally sub-Gaussian packet innovations and non-informative service
delay, its self-normalized radius is valid for all current candidates. Random
receipt order does not change the completed-sample Gram matrix; validity still
requires independent per-packet trajectory innovations conditional on launch
state. Shared or overlapping rollout noise cannot be silently treated as a
martingale difference.

The topology-robust online action inserts the optimistic value

\[
 A_p^U(a)=\min\{A_{max},x_p(a)^\top\widehat w_p+r_p(a)\}
\]

into

\[
 \min_{a,\,0\le\alpha\le\bar\alpha}
 -V[A_p^U(a)-G_p(a)(e_{i,p}+L_iM_p)]\alpha
 +\frac{VL_iG_p(a)^2}{2}\alpha^2+Q_pc_p(a).
\tag{6}
\]

For each edge, `alpha` has a scalar projected closed form; the finite local
edge scan then selects `a`. Thus the same Lyapunov drift chooses the directed
training-time cache refresh and delayed packet weight. Optimism intentionally
permits exploration and yields cumulative selected-action regret. It is not a
per-launch no-harm certificate. Replacing it by a strict lower-confidence
shield would recreate the previously observed no-information absorbing state.

The main implementation now has a dedicated topology-robust optimistic core
wrapper. The older joint wrapper remains only for the optional fixed-universe
cache-energy extension.

## Frozen next kill question

Before any selected-only ridge fit or CTDE efficacy pilot, a CPU development
gate must directly measure (1), not `R_p^H`:

1. use five distinct stochastic tanh-Gaussian actors and the registered
   recipient-specific chain caches;
2. use randomized owner selection with positive probability for every owner,
   matching the full-gradient theorem corollary;
3. build a privileged high-sample all-current reference gradient on disjoint
   rollout randomness at each audited launch;
4. estimate candidate conditional mean packet alignments with common random
   numbers, charging every simulated transition in the audit ledger;
5. optimize the resulting alignment values under the same null-or-one-edge
   action class and every prefix communication constraint;
6. compare with no refresh, complete refresh, age, parameter/action mismatch,
   round-robin, random, fixed edges, and a one-step alignment oracle;
7. require material aggregate headroom and broad cellwise direction before
   inspecting any learned critic.

This privileged calculation is a diagnostic upper bound, not the algorithm.
If it fails, the return-oracle result remains true but does not support the
learning-drift mainline, and Multiwalker controller development stops. If it
passes, an online selected-only ridge/feature development split may be frozen,
followed by untouched-seed confirmation and only then causal cached-profile
learning curves.

## Complexity and role of SDDE

The online ledger stores one selected `d`-vector and one reference direction
per outstanding packet. A ridge receipt costs `O(d^2)` via a rank-one inverse;
candidate confidence evaluation costs `O(Delta d^2)` and the Lyapunov roots
cost `O(Delta)`. With fixed local `d` and degree `Delta`, cost is independent
of total team size. No future replay, MILP, Hessian, or global agent covariance
is part of the online controller.

The exact event-time recursion remains the main theorem. A controlled hybrid
SDDE may be added only after a discrete-to-continuous coupling theorem; its
role would be to expose delay/queue/interaction phase boundaries, not to
replace (4)--(6) or to manufacture a convergence claim.

## Novelty boundary

Self-normalized linear confidence and delayed linear-bandit analysis are
inherited tools. Existing asynchronous cooperative linear-MDP work exchanges
data for homogeneous learners, while AsynCoMARL learns execution-time message
graphs. The proposed object instead controls which distinct teammate policy
version a rollout worker receives during CTDE, then couples that causal cache
choice to the returned owner's update weight and an explicit byte queue. This
combination remains a novelty hypothesis until a fresh systematic review at
submission time; the alignment-headroom and learned-controller gates must
demonstrate that the distinction has empirical value.
