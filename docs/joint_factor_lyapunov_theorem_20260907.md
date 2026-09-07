## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theorem derivation and executable algebra
- Origin Date: 2026-09-07
- Verification Status: EXACT ONE-STEP ALGEBRA; CONDITIONAL STOCHASTIC THEOREM
- Version Label: joint_factor_lyapunov_v1

# Joint Lyapunov control for factored asynchronous MARL

## One problem and one decision

Consider CTDE in a cooperative Markov game with distinct decentralized actors
`theta_i`. A rollout worker for owner `i` holds cached teammate policies
`chi_(j|i)` on a state-dependent local factor graph. It may refresh at most one
eligible cache edge before launching a trajectory. The owner gradient packet
arrives after random computation and communication delay.

At launch event `p`, the controlled action is

\[
  u_p=(a_p,\alpha_p),\qquad
  a_p\in\{\varnothing\}\cup\mathcal E_p,
  \quad 0\leq\alpha_p\leq\bar\alpha .
\tag{1}
\]

Here `a_p` is the policy-cache refresh and `alpha_p` is carried by the packet
and used as its receipt-time update weight. Both are online decisions. This is
not a heuristic combination of a graph rule and a learning-rate schedule: the
pair exactly minimizes one predictable upper bound on the same composite
Lyapunov drift.

## Composite state

For minimization of the cooperative loss `F`, define

\[
 \mathcal L_t
 =V F(\theta_t)
 +H_t+\frac{Q_t^2}{2\nu},\qquad
 H_t=\frac12\sum_{(j,i)\in E_t}\beta_{ji,t}
 \|\theta_{j,t}-\chi_{j\mid i,t}\|^2 .
\tag{2}
\]

`H` is strategic cache mismatch and `Q` is communication debt. Because the
factor support is state dependent, changing `E_t` creates an additional exact
graph-switch increment. Its positive part must be retained as a remainder (or
bounded using factor degree and parameter diameter); it cannot be silently
treated as a cache refresh. The queue obeys

\[
 Q_{p+1}=[Q_p+\nu(c_p(a_p)-\bar c)]^+ .
\tag{3}
\]

The graph is a training-time state variable. Final execution uses only local
actors and has no cache communication.

## Predictable one-step bound

Let `b(p)` and `r(p)` be launch and receipt times. Assume block smoothness
`L_i`, candidate-specific gradient clipping `||g_p(a)||<=G_p(a)`, and a predictable receipt-motion
bound

\[
 \|\theta_{r(p)}-\theta_{b(p)}\|\le M_p.
\tag{4}
\]

Let `A_p^L(a)` be a simultaneous launch-measurable lower bound on conditional
alignment:

\[
 A_p^L(a)\le
 \mathbb E[\langle\nabla_iF(\theta_{b(p)}),g_p(a)\rangle
 \mid\mathcal F_{b(p)}].
\tag{5}
\]

It must subtract statistical, Markov-mixing and factor-approximation errors;
those terms are not hidden in a fitted coefficient. Let `S_p` simultaneously
upper-bound, for every candidate, the norm of the weighted outgoing cache
displacement at receipt, and let `W_p` be a common upper bound on total outgoing
cache weight. Let `Xi_p^+` upper-bound the positive change of `H` caused only by
factor-support changes between decision epochs while holding the corresponding
policy/cache values fixed. Then smoothness, Cauchy--Schwarz, and the exact
quadratic cache identity give

\[
 \begin{aligned}
 \mathbb E[\Delta(VF+H)\mid\mathcal F_{b(p)}]
 \le{}&-m_p(a)\alpha
       +\frac{C_p(a)}{2}\alpha^2-B_p(a)+R_p+\Xi_p^+,\\
 m_p(a)={}&V[A_p^L(a)-L_iG_p(a)M_p]-G_p(a)S_p,\\
 C_p(a)={}&G_p(a)^2(VL_i+W_p),
 \end{aligned}
\tag{6}
\]

where `B_p(a)` is the exact launch-time cache reset and `R_p` contains only
other action-independent terms. A crude explicit graph-switch bound follows
from bounded local degree, bounded cache weights and a bounded policy diameter;
the performance theorem reports it rather than assuming it vanishes. Equation
(6) is conservative but causal: it never inserts a receipt-time observation
into a launch decision.

The half-square queue inequality adds `Q_p(c_p(a)-bar c)` plus
`nu(c_p(a)-bar c)^2/2`. Before action selection, the latter is bounded by one
common constant over the finite cost set. Dropping that common bound and the
remaining action-independent terms yields the index

\[
 J_p(a,\alpha)
 =-m_p(a)\alpha+\frac{C_p(a)}{2}\alpha^2
  -B_p(a)+Q_pc_p(a).
\tag{7}
\]

## Exact local minimizer

For every candidate edge,

\[
 \alpha_p^*(a)
 =\Pi_{[0,\bar\alpha]}\!\left(\frac{m_p(a)}{C_p(a)}\right),
 \qquad
 a_p\in\arg\min_a J_p(a,\alpha_p^*(a)).
\tag{8}
\]

Thus delay, Markov/factor uncertainty and receipt cache motion automatically
shrink the packet weight through `m_p`; they are not mapped to an ad-hoc stale
gradient decay. The edge decision simultaneously trades predicted descent,
exact cache reset and queue price. For local degree `Delta`, (8) requires at
most `Delta+2` gradient evaluations and `O(Delta)` scalar work. The independently
audited Pursuit interface has maximum `Delta=4` after a rarely active public
architectural cap.

## Conditional performance theorem

Suppose (4)--(6) hold simultaneously for `p<N`, costs are bounded, the null
action with zero communication and zero packet weight is always feasible, and
every launched packet is eventually drained. Compare against any
launch-measurable randomized policy supported on the same feasible action set
whose conditional expected cost is at most `bar c`. If that comparator obeys

\[
 \mathbb E[-m_p(a_p^\circ)\alpha_p^\circ
 +C_p(a_p^\circ)(\alpha_p^\circ)^2/2-B_p(a_p^\circ)
 \mid\mathcal F_{b(p)}]
 \le -V\kappa_p\|\nabla_iF(\theta_{b(p)})\|^2+VR_p^\circ,
\tag{9}
\]

then the policy in (8) satisfies

\[
 \begin{aligned}
 \sum_{p<N}\kappa_p\,
 \mathbb E\|\nabla_iF(\theta_{b(p)})\|^2
 \le{}& F(\theta_0)-F_\star
 +\frac{H_0+Q_0^2/(2\nu)}{V}\\
 &+\sum_{p<N}\mathbb E (R_p^\circ+\Xi_p^+)
 +\frac{N\nu(c_{\max}+\bar c)^2}{2V}.
 \end{aligned}
\tag{10}
\]

If an estimated lower alignment differs from a valid one by at most
`epsilon_p`, add `2 bar(alpha) sum_p epsilon_p` to the unscaled right-hand side.

### Proof

At launch, cache replacement decreases `H` by exactly `B_p(a)`. At receipt,
block smoothness gives the objective terms in (6), while expansion of the
squared outgoing cache differences gives its linear and quadratic cache terms.
The gradient change between launch and receipt is bounded by `L_i M_p`, giving
`V alpha L_i G_p(a) M_p`. Apply the queue half-square inequality to (3). The
choice (8) exactly minimizes the remaining action-dependent expression, so it
is no larger than the conditional expectation of the comparator expression;
the comparator queue term is nonpositive because `Q_p` is launch measurable.
Sum launches, receipts and queue events in chronological order. `VF+H+Q^2/(2nu)`
telescopes after inserting the explicit graph-support increments; upper-bound
their positive parts by `Xi_p^+`. Use `F>=F_star`, `H>=0`, `Q^2>=0`, then
insert (9) and divide by `V`. A uniform alignment error perturbs the chosen and comparator indices by
at most `bar(alpha) epsilon_p` each.

This proof is exact conditional on (4)--(6). It is not yet a complete deep-RL
theorem: an estimator-specific simultaneous bound for `A_p^L`, including
Markov mixing and critic/factor approximation, remains required.

## Queue stability

If `m_p(a)<=m_max`, `C_p(a)>=C_min>0`, `B_p(a)<=B_max`, and every positive action
costs at least `c_min`, its advantage over the zero action is no more than
`m_max^2/(2C_min)+B_max`. Hence no positive-cost action is selected once

\[
 Q_p>\frac{m_{\max}^2/(2C_{\min})+B_{\max}}{c_{\min}}.
\tag{11}
\]

Adding the largest final queue increment yields a deterministic queue cap and

\[
 \frac1N\sum_{p<N}c_p(a_p)
 \le \bar c+\frac{Q_N-Q_0}{\nu N}.
\tag{12}
\]

The closed-form minimizer and queue cap are implemented and checked against a
dense numerical grid in `joint_factor_lyapunov.py`.

## Remaining ICML kill gates

1. The finite-state Poisson holdout and exact known-model forms of (5) are
   proved in `markov_alignment_certificate_20260907.md`; the count-uniform
   tabular state-conditional form passed independent MCERT-001 confirmation.
   Close the remaining neural learned-model/critic radius; a stationary bridge
   alone can erase the switching signal.
2. Construct an exact small factored Markov game where (8) has a strict
   equal-resource advantage over strong fixed, age and myopic schedulers.
3. Only then freeze a Pursuit learner and test return, sample progress,
   communication Pareto curves, delay, degree and dense-task degeneration.
4. Treat an SDDE as an optional corollary only after proving generator or weak
   convergence from the event-time process. The discrete theorem is the main
   guarantee.
