# Causal redesign after the A1 CPU decision

## Material Passport

Date: 2026-10-15. Status: corrected transport implementation and conditional
theory interface; no successor scientific preregistration or performance claim.
Input: the complete, failed A1 feasibility study and its deterministic replay.

## The research question remains explicit

For independently parameterized interacting policies, can a server use
completed policy-gradient work and observed policy-version motion to allocate
future work and choose accepted update steps more effectively under actual
communication and optimizer budgets? A result must compare with strong
sequential and adaptive references and include the estimation cost.

The A1 data reject the tested noisy menu as a qualified solution. They do not
establish that all schedules have negligible opportunity. Reference's aliased
delay schedule and the failure of independent search transfer make that stronger
interpretation especially unsupported.

## Executable event model

At launch tick `s`, actor `i` receives the joint version vector `v(s)` and a
unique job identifier. A public service sequence indexed by `(seed,i,ordinal)`
sets its completion tick. Launch consumes reserved computation and message
capacity; actual transmitted bytes are also recorded at completion. The
server receives a raw gradient once and owns its current optimizer state.
Accepting or rejecting a ready packet does not undo its launch cost.

`causal_packet_ledger.py` implements these semantics without depending on
returns or actor-selection indices for the service stream. A successor
benchmark must additionally audit nonzero stale exposure, distinct method
trajectories where expected, and total budget usage before any scientific run.

## What Lyapunov can and cannot optimize in this model

Write `J(theta)` for the actual expected finite-horizon team return, `L` for
a gradient-Lipschitz constant, `g_i` for a received ascent gradient in actor
block `i`, and `theta^s` for its launch policy. On a simultaneous confidence
event, suppose

\[
\|g_i-\nabla_i J(\theta^s)\|\le r_i,
\qquad
\|\nabla_iJ(\theta)-\nabla_iJ(\theta^s)\|
\le L_i\|\theta-\theta^s\|.
\]

The second inequality requires a justified uniform constant, including
cross-actor policy motion. Define `d_i = r_i + L_i ||theta-theta^s||`. Then
Cauchy--Schwarz gives

\[
\langle\nabla_iJ(\theta),g_i\rangle
\ge \|g_i\|^2-d_i\|g_i\|.
\]

If `H` is a valid block curvature majorant along every proposed update
segment, let `C_ij=||g_i|| H_ij ||g_j||`. The smoothness inequality yields

\[
J(\theta+\operatorname{blockdiag}(g)\alpha)-J(\theta)
\ge B(\alpha)
=\sum_i\alpha_i(\|g_i\|^2-d_i\|g_i\|)
-\tfrac12\alpha^\top C\alpha.
\]

For a positive semidefinite `C`, maximizing this lower bound over bounded
nonnegative steps is a concave quadratic problem. Eligibility removes packets
that have not arrived; zero step rejects a packet. This is a conditional return
bound only if the gradients and constants above concern `J` itself. A critic
loss gradient cannot be silently substituted for a team-return gradient.

A confidence event must be uniform over all eligible packets and all inspected
actions before adaptive selection. Marginal per-packet intervals and noise
assumptions at launch alone do not automatically survive receipt-time selection.
Markov trajectory bias must be included in `r_i`; it is not removed by naming
the data a stochastic gradient.

Now let resource queues obey `Q_r(t+1)=[Q_r(t)+c_r(x_t)-b_r]^+`, where `x_t`
is the **launch** action. With `V=-v J+sum_r Q_r^2/2`, the one-step bound is

\[
\Delta V\le -v B(\alpha_t)
+\sum_r Q_r(t)(c_r(x_t)-b_r)
+\tfrac12\sum_r(c_r(x_t)-b_r)^2.
\]

The update part depends on ready packets and `alpha_t`; the resource term
depends on new launches `x_t`. Charging ready-packet acceptance again would
misprice sunk work. Conversely, minimizing this one-step expression does not
value a new launch's future completion and can choose perpetual no-launch.
Thus **a valid joint launch/step algorithm still needs a multi-step drift or
an explicitly calibrated potential for in-flight work**. The displayed
inequality does not establish queue stability, finite-time convergence, or a
competitive return bound by itself.

No SDDE approximation is required for these exact discrete accounting and
smoothness steps. A diffusion limit can only be added after the discrete
algorithm and its scaling assumptions have been specified.

## Concrete next experiment design boundary

The successor screen must change the failed mechanism before spending more
rollout search:

1. Use the unique-job event model with exogenous service streams and server
   optimizer ownership; verify actual stale exposure per method.
2. Freeze common training prefixes and an eligible reference action at each
   comparison point. Evaluate candidate-minus-reference increments on the
   same independent episode clusters. The reference must be an admissible
   action in the candidate comparison.
3. Measure paired episode-level variance on **new qualification seeds** and
   calculate whether the registered minimum effect can be resolved within
   the sensing budget. Do not simply increase the 32-episode search count.
4. Freeze a score based on already-paid, previously completed work; qualify
   its independent ranking and simultaneous confidence before it controls a
   scientific run. Any fresh rollout or gradient computation is charged.
5. Derive a multi-step drift statement that accounts for outstanding work and
   ties the scored quantity to actual team return. A generic queue inequality
   or a QP solver alone is insufficient.

If the precision calculation is infeasible or the score cannot be computed
causally with a useful bound, stop before a new return benchmark. This is the
necessary design decision, not a new narrative or a relaxed gate.

## Clarification to the earlier deterministic counterexample

For `J(a1,a2)=a1+a2-kappa*a1*a2`, replacing deterministic action zero by
deterministic action one demonstrates the exact sequential-surrogate identity
`2-kappa` and the missing interaction term. Its old/new KL divergence is
infinite. Therefore that example is not an exact finite HAPPO certificate
including a KL penalty. The algebraic interaction conclusion remains valid;
the finite trust-region certificate requires supported stochastic policies.
