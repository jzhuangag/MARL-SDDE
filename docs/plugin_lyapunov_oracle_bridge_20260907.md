## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theorem derivation and algorithm-design audit
- Origin Date: 2026-09-07
- Verification Status: DISCRETE PLUG-IN REDUCTION PROVED; CONTROLLED NEURAL MARKOV RATE OPEN
- Version Label: plugin_lyapunov_oracle_bridge_v1

# From edge estimates to a dynamic-oracle Lyapunov guarantee

## Decision

The main algorithm should not require every selected packet to have a positive
lower-confidence alignment.  That rule is a valid optional safety mode, but the
development simulation shows that it can leave a finite-sample error floor and
erase the value of dynamic collaboration.  The primary controller instead
minimizes the same joint graph--weight Lyapunov index using a predictable
plug-in edge estimate and pays an explicit estimation term in its finite-time
guarantee.

This is not an unprotected heuristic.  The action still comes from exact
minimization of the proved one-step Lyapunov surrogate; uncertainty appears as
a measurable oracle-regret term rather than as a hard veto on collaboration.

## Setup

For launch event `p`, candidate cache action `a`, and packet weight
`alpha in [0, alpha_max]`, write the action-dependent part of the true drift
bound as

\[
J_p(a,\alpha;A_p)
=-m_p(a;A_p)\alpha+\frac{C_p(a)}2\alpha^2
-B_p(a)+Q_pc_p(a),
\]

where

\[
m_p(a;A_p)=V[A_p(a)-L_iG_p(a)M_p]-G_p(a)S_p.
\]

All terms other than the conditional alignment `A_p(a)` are launch
measurable.  Let `Ahat_p(a)` be a launch-measurable estimate, and suppose an
event `E_p` supplies the simultaneous bound

\[
\max_a |Ahat_p(a)-A_p(a)|\leq epsilon_p.
\tag{1}
\]

The plug-in controller computes the scalar closed-form minimizer for every
local candidate and selects

\[
(ahat_p,alphahat_p)\in\arg\min_{a,\alpha\in[0,\bar\alpha]}
J_p(a,\alpha;Ahat_p).
\tag{2}
\]

For a sparse factor graph this costs `O(Delta_i)` scalar operations after the
edge statistics are available.  It is not a graph QP and requires no Hessian
inverse.

## Theorem 1: deterministic dynamic-oracle reduction

On `E_p`, for every feasible launch-measurable comparator
`(a_p^o,alpha_p^o)`,

\[
J_p(ahat_p,alphahat_p;A_p)
-J_p(a_p^o,alpha_p^o;A_p)
\leq 2V\bar\alpha\,epsilon_p.
\tag{3}
\]

### Proof

The alignment is the only estimated term and enters the index linearly, hence

\[
|J_p(a,\alpha;Ahat_p)-J_p(a,\alpha;A_p)|
\leq V\bar\alpha\,epsilon_p
\]

simultaneously for all candidates and weights.  Apply this inequality to the
selected pair, use the exact minimization in (2), and apply it once more to the
comparator.  This gives (3).  No independence between the selected action and
the confidence event is needed because (1) is simultaneous over the candidate
set and `ahat_p` is predictable.

## Corollary 1: finite-time stationarity with estimation cost

Under the assumptions of the exact discrete theorem in
`joint_factor_lyapunov_theorem_20260907.md`, replace its certified action by
(2).  On the intersection of the simultaneous estimation events,

\[
\begin{aligned}
\sum_{p<N}\kappa_p\,\mathbb E\|\nabla_iF(\theta_{b(p)})\|^2
\leq{}&F(\theta_0)-F_\star
+\frac{H_0+Q_0^2/(2\nu)}{V}\\
&+\sum_{p<N}\mathbb E(R_p^o+\Xi_p^+)
+2\bar\alpha\sum_{p<N}\mathbb E\epsilon_p\\
&+\frac{N\nu(c_{\max}+\bar c)^2}{2V}.
\end{aligned}
\tag{4}
\]

Thus the controller remains competitive with a dynamic, launch-measurable
graph--weight oracle whenever the average estimation radius vanishes.  The
hard-LCB controller corresponds to demanding a per-packet certificate; (4)
instead controls the cumulative price of estimation, which is the quantity
needed for finite-time learning.

## Tabular Markov instantiation

For the finite-state interface already proved in
`markov_alignment_certificate_20260907.md`, compute both a lower and an upper
robust value from the same count-uniform confidence sets.  A plug-in value
obtained from the empirical kernel and empirical state scores obeys (1) with

\[
epsilon_p=
\max_{a,s}\max\{Ahat_p(a,s)-L_p(a,s),
                 U_p(a,s)-Ahat_p(a,s)\}.
\tag{5}
\]

The summable event allocation already used by MCERT-001 makes (5)
simultaneous over all refresh events.  In a fixed uniformly mixing finite
model, the radii shrink at the usual count-dependent square-root rate, so the
extra term in (4) is sublinear when every relevant state is visited linearly
often.

This statement does **not** yet prove the neural Pursuit case.  There, the
candidate score is computed by a learned locally factored critic and the
policy update changes the trajectory kernel.  A deliverable theorem still
needs an explicit sum of (i) Markov/Poisson sampling error, (ii) critic and
factor approximation error, and (iii) within-window controlled-kernel drift.

## Development evidence and interpretation

The end-to-end two-state development model contains a common local update on
every event and an optional state-dependent teammate factor.  It identified
two distinct facts:

1. the exact dynamic Lyapunov controller has substantial cumulative-risk
   headroom over fixed-edge schedules;
2. the hard-LCB controller is too conservative at the tested finite horizon,
   whereas the sequential plug-in controller reaches the strong fixed
   envelope with much lower communication and small positive gains in some
   development seeds.

These are design observations, not formal paper evidence.  No seed or threshold
has been frozen for this new controller.  The next experiment must test the
phase predicted by (4): horizon versus estimation cost, Markov persistence,
delay, factor switching, degree, and communication budget, all against a
strong fixed-edge envelope and an exact dynamic oracle.

## Stop conditions before a standard MARL run

Do not preregister a Pursuit GPU run unless a fresh CPU design audit shows all
of the following without changing its gates after seeing confirmation data:

1. positive equal-resource headroom relative to the strong fixed and myopic
   online envelopes, not merely to a weak fixed edge;
2. a nontrivial fraction of the exact dynamic-oracle gain is recovered after
   charging estimation and local-factor computation;
3. the queue respects the communication budget and the measured computation
   scales with local degree;
4. dense graphs honestly lose the sparse-complexity advantage;
5. the controlled-Markov/critic approximation term needed by (4) is stated
   and measured rather than hidden inside a generic confidence radius.

