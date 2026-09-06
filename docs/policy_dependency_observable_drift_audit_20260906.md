# Observable signed drift for policy-dependency synchronization

Date: 2026-09-06

Status: theory-interface audit.  This document does not register or authorize a
new outcome experiment, pilot, formal seed set, GPU job, or HPC4 operation.

## 1. Decision

The controlled strategic-freshness problem remains viable, but the PDSG-001
oracle is not directly executable.  A causal controller can instead minimize a
confidence-valid upper bound on the same one-step Lyapunov drift.  The bound has
a closed-form step for each candidate edge and an `O(deg(i))` edge comparison
once its predictable statistics are available.

The interface is conditionally feasible.  It is not theorem-complete because
the Markov confidence radii, delayed launch/receipt history term, and deep-MARL
autodiff cost have not yet been established.

## 2. Information available when an edge is chosen

Let `F_k` be the sigma-field immediately before dispatching the next rollout
owned by agent `i`.  The action must be `F_k`-measurable.

Exactly observed at dispatch are:

- current server policy blocks and every worker cache version;
- the version and timestamp of every deliverable policy snapshot;
- the current communication queue, edge costs, and declared budget;
- all completed trajectories, critic states, and optimizer states;
- the dependency sparsity mask fixed before the next trajectory.

Predictable but estimated quantities are:

- the current owner-block objective gradient `a_i = grad_i F(theta_k)`;
- the candidate mixed-cache gradient `g_ij`;
- their alignment and candidate-gradient norm;
- block smoothness and Markov-bias envelopes.

Forbidden quantities are the next Markov state, the noise or return of the
trajectory being scheduled, the future policy path, an unknown optimum, and
outcome-selected constants.  In particular, the registered quadratic target
and expected Hessian used by PDSG-001 are oracle-only.

## 3. Confidence-valid drift score

For a candidate refresh `j -> i`, suppose the dispatch-time estimators satisfy
on one simultaneous event

\[
\|a_i-\widehat a_i\|\le r_{a,i},
\qquad
\|g_{ij}-\widehat g_{ij}\|\le r_{g,ij}.
\tag{1}
\]

Cauchy--Schwarz gives the observable alignment lower bound

\[
\ell_{ij}=
\langle\widehat a_i,\widehat g_{ij}\rangle
-r_{a,i}\|\widehat g_{ij}\|
-r_{g,ij}\|\widehat a_i\|
-r_{a,i}r_{g,ij}
\le \langle a_i,g_{ij}\rangle,
\tag{2}
\]

and the norm upper bound

\[
U_{ij}=\|\widehat g_{ij}\|+r_{g,ij}
\ge \|g_{ij}\|.
\tag{3}
\]

If `F` is block-`L_i` smooth, the learning term obeys

\[
F(\theta_k-\alpha U_i g_{ij})-F(\theta_k)
\le -\alpha\ell_{ij}
+\frac{L_i\alpha^2}{2}U_{ij}^2.
\tag{4}
\]

Let `v_(j,k)` be the snapshot that will actually be delivered.  The exact
dispatch-time cache-debt reduction is

\[
R_{ij,k}=\frac{p_{ij}}2\left(
\|\theta_j^k-\chi_{j\to i}^k\|^2
-\|\theta_j^k-v_{j,k}\|^2
\right).
\tag{5}
\]

The second norm prevents a delayed snapshot from receiving the same credit as
a fresh snapshot.  `R_(ij,k)` may be negative.

For the virtual queue

\[
Q_{k+1}=[Q_k+c_{ij}-\bar c]^+,
\tag{6}
\]

the standard square expansion gives

\[
\frac12(Q_{k+1}^2-Q_k^2)
\le Q_k(c_{ij}-\bar c)
+\frac12(c_{ij}-\bar c)^2.
\tag{7}
\]

An action-dependent certified score is therefore

\[
\overline\Psi_{ij}(\alpha)=
-\alpha\ell_{ij}
+\frac{L_i\alpha^2}{2}U_{ij}^2
-R_{ij,k}
+Q_kc_{ij}
+\frac12(c_{ij}-\bar c)^2
+\overline R^{\rm hist}_{ij,k},
\tag{8}
\]

after dropping the common term `-Q_k bar(c)`.  The final history term must be a
predictable upper bound for in-flight source motion and Markov packet bias.  It
cannot be silently treated as an action-independent constant unless that is
proved.

For fixed `j`, the exact minimizer of the first two terms on
`[0, alpha_max]` is

\[
\alpha_{ij}^{\rm safe}
=\Pi_{[0,\alpha_{\max}]}
\frac{[\ell_{ij}]_+}{L_iU_{ij}^2},
\tag{9}
\]

with zero step when `U_(ij)=0`.  Comparing (8) over no refresh and the eligible
incoming edges jointly selects the graph edge and update mass.  Thus the
Lyapunov function is both the analysis certificate and the online design
objective.

## 4. Low-complexity estimator interface

A prospective deep implementation may use a completed replay block available
before dispatch:

1. estimate the current owner gradient with the centralized critic;
2. form the stale-cache candidate gradient already required by the actor
   update;
3. define the scalar alignment surrogate
   `s_i(chi_i) = <stopgrad(hat(a_i)), hat(g_i(chi_i))>`;
4. take one reverse-mode derivative of `s_i` with respect to all teammate
   cache blocks and partition it by donor;
5. contract each block with the known version displacement and pay a certified
   Taylor remainder;
6. compute (8)--(9) for the declared sparse dependency neighbors.

This avoids one trajectory per edge and avoids a Hessian inverse.  It does not
yet prove a one-backward-pass overhead claim: framework tracing, critic
sharing, vector dimension, and the remainder certificate must be measured.

The replay estimator may reuse ordinary training data and therefore need not
consume extra environment transitions.  Its computation, memory, and any
additional critic evaluation must nevertheless be reported and included in
the experimental cost comparison.

## 5. Timing discrepancy exposed by the audit

The frozen PDSG-001 runner supplies `theta^(k-d)` immediately at event `k`,
refreshes the cache, and forms the same-event gradient.  It is an exact
delayed-snapshot model, not a complete launch/in-flight/receipt simulator.
Accordingly, PDSG-001 establishes dynamic headroom in that model but does not
verify the history term in (8).

The paper-level discrete process must separately represent:

- dispatch and source-version capture;
- communication receipt and cache replacement;
- rollout birth under the resulting mixed policy versions;
- trajectory receipt and owner-block parameter update.

Only after proving an event-time drift bound for this process may a
functional Itô--Dynkin calculation for the controlled hybrid SDDE be used as a
continuous approximation and phase interpretation.

## 6. Proof obligations before a successor CPU gate

1. Construct `F_k`-measurable estimators in (1) from completed Markov data.
2. Derive time-uniform radii under an explicit mixing or regeneration
   assumption, including critic and importance-weight errors.
3. Bound the linearization remainder used by the one-reverse-pass interface.
4. Prove the delayed event-time recursion including (5), (7), and the packet
   pipeline.
5. Show that minimizing (8) yields communication-queue mean-rate stability and
   a finite-time stationarity bound relative to the declared comparator.
6. Establish that the confidence event is nonvacuous on an outcome-free CPU
   design grid.
7. Measure estimator runtime and memory before registering any learning
   efficacy experiment.

Failure of obligations 1, 3, or 6 rejects the proposed low-complexity
controller.  It does not erase the PDSG-001 problem-level headroom and does not
authorize tuning on its frozen outcomes.
