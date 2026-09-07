## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theorem derivation and executable algebra
- Origin Date: 2026-09-07
- Verification Status: CORE THEOREM TOPOLOGY-ROBUST; CACHE EXTENSION EXACT WITH DECLARED REMAINDER
- Version Label: joint_factor_lyapunov_v2

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

## Layer I: topology-robust core Lyapunov controller

The principal theorem uses only the cooperative learning potential and the
communication debt,

\[
 \mathcal L_t^{\rm core}
 =V F(\theta_t)+\frac{Q_t^2}{2\nu},
 \qquad
 Q_{p+1}=[Q_p+\nu(c_p(a_p)-\bar c)]^+ .
\tag{2}
\]

It does not sum cache energy over the currently active graph. Therefore the
theorem remains valid when Pursuit changes which local factors are present.
The graph is a training-time action set; final execution uses local actors and
no cache communication.

Let `b(p)` and `r(p)` be packet launch and receipt times. Assume block
smoothness `L_i`, action-specific clipping `||g_p(a)||<=G_p(a)`, and the
predictable receipt-motion bound

\[
 \|\theta_{r(p)}-\theta_{b(p)}\|\le M_p.
\tag{3}
\]

Let `A_p^L(a)` be a simultaneous launch-measurable lower bound on the
conditional alignment

\[
 A_p^L(a)\le
 \mathbb E[\langle\nabla_iF(\theta_{b(p)}),g_p(a)\rangle
 \mid\mathcal F_{b(p)}].
\tag{4}
\]

Statistical, Markov-mixing, controlled-kernel, and factor-approximation errors
must be subtracted explicitly in this bound. Smoothness and (3)--(4) give

\[
 \mathbb E[\Delta(VF)\mid\mathcal F_{b(p)}]
 \le -m_p^{\rm core}(a)\alpha
 +\frac{C_p^{\rm core}(a)}2\alpha^2+R_p,
\tag{5}
\]

where

\[
 m_p^{\rm core}(a)=V[A_p^L(a)-L_iG_p(a)M_p],
 \qquad
 C_p^{\rm core}(a)=VL_iG_p(a)^2,
\tag{6}
\]

and `R_p` contains only action-independent declared remainders. The queue
half-square inequality adds `Q_p(c_p(a)-bar c)` and an action-independent
upper bound on `nu(c_p(a)-bar c)^2/2`. Hence the executable core index is

\[
 J_p^{\rm core}(a,\alpha)
 =-m_p^{\rm core}(a)\alpha
 +\frac{C_p^{\rm core}(a)}2\alpha^2+Q_pc_p(a).
\tag{7}
\]

For every current candidate,

\[
 \alpha_p^{\rm core}(a)=
 \Pi_{[0,\bar\alpha]}\!\left(
 \frac{m_p^{\rm core}(a)}{C_p^{\rm core}(a)}\right),
 \qquad
 a_p\in\arg\min_a
 J_p^{\rm core}(a,\alpha_p^{\rm core}(a)).
\tag{8}
\]

The edge and packet weight are therefore selected by the same Lyapunov bound.
The edge enters through its action-specific future trajectory alignment; delay
shrinks its weight through `M_p`, and the queue prices its actual policy bytes.
For local degree `Delta`, (8) costs `O(Delta)` scalar work after the signed
statistics are formed. No cache topology appears in (2) or (7).

## Core conditional performance theorem

Suppose (3)--(5) hold simultaneously for `p<N`, costs are bounded, a zero-cost
null action with zero packet weight is feasible, and all launched packets are
eventually drained. Compare with any launch-measurable randomized policy on the
same current action sets whose conditional expected cost is at most `bar c`.
If that comparator satisfies

\[
 \mathbb E[-m_p^{\rm core}(a_p^\circ)\alpha_p^\circ
 +C_p^{\rm core}(a_p^\circ)(\alpha_p^\circ)^2/2
 \mid\mathcal F_{b(p)}]
 \le -V\kappa_p\|\nabla_iF(\theta_{b(p)})\|^2+VR_p^\circ,
\tag{9}
\]

then (8) obeys

\[
 \begin{aligned}
 \sum_{p<N}\kappa_p\,
 \mathbb E\|\nabla_iF(\theta_{b(p)})\|^2
 \le{}& F(\theta_0)-F_\star
 +\frac{Q_0^2}{2\nu V}
 +\sum_{p<N}\mathbb E(R_p+R_p^\circ)\\
 &+\frac{N\nu(c_{\max}+\bar c)^2}{2V}.
 \end{aligned}
\tag{10}
\]

A uniform action-wise alignment error `epsilon_p` adds at most
`2 bar(alpha) sum_p epsilon_p` to the right-hand side. This theorem has no
topology-turnover term.

### Proof

At receipt, apply block smoothness and replace the receipt gradient by its
launch value using (3). This proves (5). The scaled queue recursion yields

\[
 \Delta[Q_p^2/(2\nu)]
 \le Q_p(c_p-\bar c)+\nu(c_p-\bar c)^2/2.
\]

The rule (8) exactly minimizes all remaining action-dependent terms. Its index
is therefore no larger than the conditional comparator index; the comparator
queue cross-term is nonpositive because `Q_p` is launch measurable and its
conditional expected cost is at most `bar c`. Sum events chronologically,
telescope `VF+Q^2/(2nu)`, use `F>=F_star` and `Q^2>=0`, insert (9), and divide
by `V`. Uniform alignment error perturbs the selected and comparator indices by
at most `V bar(alpha) epsilon_p` each.

The proof is exact conditional on the declared alignment and motion bounds. It
is not yet a distribution-free neural-Pursuit theorem: the learned-critic
radius remains an experimental/theoretical interface obligation.

## Layer II: optional fixed-universe cache-energy extension

Persistent caches may be regularized without invalidating the core theorem.
Fix one edge universe `U` and define

\[
 H_t=\frac12\sum_{e=(j,i)\in U}\beta_e
 \|\theta_{j,t}-\chi_{e,t}\|^2,
 \qquad \beta_e\ge0.
\tag{11}
\]

The state-dependent candidate set may change arbitrarily inside `U`; inactive
edges remain represented in (11). With fixed weights, candidate-set turnover
does not change `H`. Exact launch reset `B_p(a)`, a simultaneous outgoing-cache
displacement bound `S_p`, and total outgoing weight `W_p` produce

\[
 \begin{aligned}
 m_p^{H}(a)&=m_p^{\rm core}(a)-G_p(a)S_p,\\
 C_p^{H}(a)&=C_p^{\rm core}(a)+G_p(a)^2W_p,\\
 J_p^{H}(a,\alpha)&=-m_p^{H}(a)\alpha
 +C_p^{H}(a)\alpha^2/2-B_p(a)+Q_pc_p(a).
 \end{aligned}
\tag{12}
\]

The same scalar projection and finite candidate scan exactly minimize (12).
This extension gives a persistent freshness pressure, but it is not needed for
the topology-robust core result.

If the weights are instead allowed to change, all weights must still be
represented on one fixed `U`. Holding policies and caches fixed during the
reweighting event, the exact topology-motion jump is

\[
 \Xi_p=\frac12\sum_{e=(j,i)\in U}
 (\beta_{e,p+1}-\beta_{e,p})
 \|\theta_{j,p}-\chi_{e,p}\|^2.
\tag{13}
\]

For `||theta_j-chi_e||<=R`,

\[
 \Xi_p\le\Xi_p^+
 \le\frac{R^2}{2}\sum_{e\in U}
 [\beta_{e,p+1}-\beta_{e,p}]_+.
\tag{14}
\]

The cache-augmented analogue of (10) adds `H_0/V` and
`sum_p E[Xi_p^+]/V`, and uses the comparator expression in (12). If one writes
`H_t` only over the active `E_t`, this is exactly the indicator-weight special
case of (13); omitting `Xi_p` is incorrect. High graph turnover is therefore a
measured source of cache-theorem cost, not a free selling point.

### Event partition and no-double-counting contract

At every decision epoch, cache-energy change is partitioned in this order:

\[
 \Delta H_p=
 \underbrace{\Xi_p}_{\text{weight/support motion}}
 -\underbrace{B_p(a_p)}_{\text{launch cache copy}}
 +\underbrace{D_p^{\rm receipt}}_{\text{donor parameter update}}.
\tag{15}
\]

Each term is evaluated against the state immediately preceding its event. The
controlled-kernel discrepancy is not a fourth cache term: it enters exactly
once when constructing the alignment bound `A_p^L(a)` for `Delta F`. Likewise,
`B_p`, `D_p^receipt`, and `Xi_p` never enter the controlled-kernel radius. This
separates distribution shift in the learning potential from bookkeeping change
in cache energy, even though both are causally triggered by the same refresh.
The exact three-event identity is covered by an executable regression test.

## Queue stability

For the core controller, a positive action can improve the zero-action index
by at most `(m_max^core)^2/(2 C_min^core)`. It is not selected once

\[
 Q_p>\frac{(m_{\max}^{\rm core})^2}
 {2C_{\min}^{\rm core}c_{\min}}.
\tag{16}
\]

For the cache extension, add `B_max` to the numerator. Adding the largest final
queue increment yields a deterministic cap, and queue iteration gives

\[
 \frac1N\sum_{p<N}c_p(a_p)
 \le \bar c+\frac{Q_N-Q_0}{\nu N}.
\tag{17}
\]

The core minimizer is implemented in `core_factor_lyapunov.py`; the
cache-augmented minimizer remains in `joint_factor_lyapunov.py`, and the exact
topology-motion identity is in `composite_cache_lyapunov.py`. All are tested
against direct energy differences or dense grids.

## Remaining ICML kill gates

1. The finite-state Poisson holdout and exact known-model forms of (5) are
   proved in `markov_alignment_certificate_20260907.md`; the count-uniform
   tabular state-conditional form passed independent MCERT-001 confirmation.
   Close the remaining neural learned-model/critic radius; a stationary bridge
   alone can erase the switching signal.
2. PDSG-FR-001 tests only the core controller: it contains no persistent
   cache-energy state and must not be cited as evidence for Layer II.
3. Before Pursuit efficacy, implement and test real launch-time cache copying,
   message charging independent of receipt weight, and cache persistence.
4. Only then freeze a Pursuit learner and test return, sample progress,
   communication Pareto curves, delay, degree and dense-task degeneration.
5. Treat an SDDE as an optional corollary only after proving generator or weak
   convergence from the event-time process. The discrete theorem is the main
   guarantee.
