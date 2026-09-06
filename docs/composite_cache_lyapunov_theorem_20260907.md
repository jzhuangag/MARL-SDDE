## Material Passport

- Origin Skills: academic-research-suite; publishable-academic-writing
- Origin Mode: theorem derivation and implementation alignment
- Origin Date: 2026-09-07
- Verification Status: VERIFIED FINITE-EVENT ALGEBRA; CONDITIONAL LEARNING THEOREM
- Version Label: composite_cache_lyapunov_v1

# Composite Lyapunov control of asynchronous policy freshness

## The single controlled object

The training state contains distinct current policies `theta_j`, recipient
worker caches `chi_(j->i)`, delayed owner-gradient packets, and a communication
queue.  The method controls one object: the directed policy-cache refresh graph
used to launch asynchronous rollouts.  The graph is neither a value-
factorization graph nor an execution-time communication graph.

The composite Lyapunov state is

\[
 \mathcal L_t
 =V F(\theta_t)+H_t+\frac{1}{2\nu} Q_t^2,
 \qquad
 H_t=\frac12\sum_{i\ne j}\beta_{ji}
 \|\theta_{j,t}-\chi_{j\to i,t}\|^2 .
\tag{1}
\]

where `nu>0` is the dual-queue step.  Thus Lyapunov theory is both the design
and analysis principle.  `F` values learning progress, `H` prices strategically
stale teammate policies, and `Q` prices communication debt.  The same one-step
upper bound selects the graph edge and telescopes in the convergence proof.

## Exact launch reset

At launch `p`, refreshing `e=(j->i)` copies the current `theta_j` into the
recipient cache.  Holding all current policies fixed, the exact decrease is

\[
 B_p(e)=\frac{\beta_{ji}}2
 \|\theta_{j,b(p)}-\chi_{j\to i,b(p)}\|^2 .
\tag{2}
\]

The null action has `B_p(empty)=0`.  Equation (2) is computed from the version
cache already present at the centralized learner; it requires no environment
probe, reward, or additional sensor.

## Exact receipt increment

Suppose a delayed packet owned by `j` is applied as
`theta_j^+=theta_j-w_p G_p`.  Its exact increment to every outgoing cache debt
is

\[
 \Gamma_p=
 -w_p\left\langle G_p,
   \sum_{i\ne j}\beta_{ji}(\theta_j-\chi_{j\to i})
 \right\rangle
 +\frac{w_p^2}{2}\|G_p\|^2
   \sum_{i\ne j}\beta_{ji}.
\tag{3}
\]

This identity is evaluated at receipt, so it remains exact under arbitrary
intervening launches and receipts.  If `||G_p||<=Gbar_p`, its realized upper
bound is

\[
 \bar\Gamma_p=
 w_p\bar G_p
 \left\|\sum_{i\ne j}\beta_{ji}(\theta_j-\chi_{j\to i})\right\|
 +\frac{w_p^2\bar G_p^2}{2}\sum_{i\ne j}\beta_{ji}.
\tag{4}
\]

Gradient clipping makes `Gbar_p` explicit.  The weighted cache displacement in
(4), however, is generally a receipt-time quantity because intervening events
can change both current policies and caches.  It is not launch-predictable
without a separate in-flight/path bound.  The primary controller therefore
does not pretend to know (4): its positive part is retained explicitly in the
finite-time remainder.  A valid launch-predictable common envelope may replace
that realized remainder in a corollary.

## Edge decision

Let `D_p^F(a)` be the paired launch-to-receipt learning-drift bound from the
signed packet theorem and let `c_p(a)` be refresh units.  Define

\[
 \widehat J_p(a)=
 V\widehat D_p^F(a)-B_p(a)+Q_pc_p(a).
\tag{5}
\]

The controller executes

\[
 a_p\in\arg\min_{a\in\{\varnothing\}\cup\mathcal C_p}
 \widehat J_p(a),
 \qquad
 Q_{p+1}=[Q_p+\nu(c_p(a_p)-\bar c)]^+ .
\tag{6}
\]

The scaling in (1) makes the queue drift exactly
`Q_p(c_p-bar c)` plus a remainder at most
`nu(c_p-bar c)^2/2`; consequently `nu` changes the dual response speed without
silently rescaling the queue price in (5).  Iteration also gives the pathwise
finite-horizon budget relation

\[
 \frac1N\sum_{p<N}c_p(a_p)
 \le \bar c+\frac{Q_N}{\nu N}.
\tag{7}
\]

The nonlinear implementation evaluates the signed change in `D_p^F` for every
donor with one cross-policy VJP, evaluates (2) while forming the same parameter
displacement, and scans `1+Delta_p` scalar indices.  It does not insert a
receipt-time quantity into the launch score.
Arithmetic after differentiation is `O(Delta_p)`; parameter contraction is
`O(Delta_p d)` and the actual runtime must be reported.

A hard causal prefix cap may remove positive actions from (6) so that every
finite experiment satisfies its policy-byte budget exactly.  This does not
change the minimizer over the remaining feasible set.

## Conditional finite-time theorem

Assume the paired smoothness/receipt conditions of the signed packet theorem,
eventual terminal drain, bounded costs, and a simultaneous predictable score
event

\[
 \max_{a\in\mathcal A_p}
 |\widehat D_p^F(a)-D_p^F(a)|\le\epsilon_p^F.
\tag{8}
\]

Assume also a predictable randomized comparator distribution supported on the
same currently feasible action set.  If `a_p^circ` is drawn from that
distribution, require

\[
 \mathbb E[c_p(a_p^\circ)-\bar c\mid\mathcal F_{b(p)}]\le 0
\tag{9}
\]

and

\[
 \mathbb E\!\left[D_p^F(a_p^\circ)
 -\frac{B_p(a_p^\circ)}{V}\,\middle|\,\mathcal F_{b(p)}\right]
 \le -\kappa_p\|\nabla_{i_p}F(\theta_{b(p)})\|^2+R_p.
\tag{10}
\]

Then

\[
 \sum_{p<N}\kappa_p\mathbb E\|\nabla_{i_p}F(\theta_{b(p)})\|^2
 \le F(\theta_0)-F_\star
 +\frac{H_0+Q_0^2/2}{V}
 +\sum_{p<N}\mathbb E R_p
 +\frac1V\sum_{p<N}\mathbb E[\Gamma_p^+]
 +2\sum_{p<N}\mathbb E\epsilon_p^F
 +\frac{NB_Q}{V}.
\tag{11}
\]

### Proof

At every launch, (2) accounts for the entire cache-state jump.  At every
receipt, the block-smoothness bound accounts for the objective jump and (3)
accounts for the entire outgoing-cache jump.  Pair each launch with its
eventual receipt.  The queue half-square inequality contributes
`Q_p(c_p-bar c)+B_Q`, where one may take
`B_Q=nu sup_p(c_p-bar c)^2/2`.  Exact minimization of (5), followed by
averaging over the comparator's launch-measurable randomization, incurs at most
`2 V epsilon_p^F`.  The conditional cost constraint (9) removes its expected
queue term because `Q_p` is launch-measurable.  Summing every launch reset, every receipt increment, and every queue
increment telescopes `V F+H+Q^2/2` in chronological event order.  Because the
launch rule does not know the future receipt term, upper-bound it by
`Gamma_p^+=max(Gamma_p,0)` after it is realized.  Use
`F>=F_star`, `H_N>=0`, `Q_N^2>=0`, insert (10), and divide by `V`.

Equation (11) is conditional because the neural critic/VJP score event (8)
still needs an estimator-specific Markov generalization argument.  The cache
identities and telescoping step are exact and do not depend on that future
argument.  If a valid launch-time common upper bound for (4) is available, it
can be included in every candidate and comparator drift; the primary theorem
does not require or fabricate one.

## Why this removes the null absorbing state

For one feasible edge `e`, let

\[
 \delta_p(e)=
 [\widehat D_p^F(e)-\widehat D_p^F(\varnothing)]_+
\]

and use a common receipt envelope.  Edge `e` strictly beats null whenever

\[
 B_p(e)>V\delta_p(e)+Q_pc_p(e).
\tag{12}
\]

Therefore an untrained or conservative critic does not permanently force the
null action: if a repeatedly eligible donor continues to change, its exact
quadratic cache debt can eventually dominate a bounded score disadvantage and
queue price whenever a prefix-budget token is available.  If the donor stops
changing, no forced refresh is needed.  This is a conditional non-absorption
statement, not a promise that every edge will be used.

The actual four-agent Pistonball plumbing smoke illustrates the mechanism.
The learning-only score used zero of four available refresh units; adding the
cache term selected three causal edges while preserving the exact prefix
budget and draining all eight gradient packets.  These are implementation
diagnostics, not return evidence.

## Scope and remaining obligation

The theorem controls a training-time graph under CTDE; decentralized execution
uses the final local actors and no cache messages.  Primary comparisons remain
at matched actor transitions and optional policy bytes.  Wall-clock and VJP
overhead are secondary but mandatory measurements.

Before the standard benchmark can support the paper, the GPU pilot must show
that the signed term improves over strong age/mismatch and complete-burst
controllers.  A result that merely matches mismatch scheduling would validate
cache stabilization but not the claimed strategic value.  The pilot must also
measure sensitivity to `beta/V`; this coefficient cannot be selected after
looking at formal seeds.
