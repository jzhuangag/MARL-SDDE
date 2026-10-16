# Four-event Lyapunov accounting for policy-cache synchronization

Date: 2026-09-06

Status: timing and algebra audit.  It corrects the causal placement of graph
and step decisions but does not complete the stochastic finite-time theorem.

## 1. Four distinct events

The asynchronous process must distinguish:

1. **dispatch:** the server selects a refresh edge and captures a policy
   snapshot;
2. **delivery:** the selected snapshot replaces one worker cache entry;
3. **rollout birth:** the worker begins a Markov trajectory under the resulting
   mixed policy-version vector;
4. **trajectory return:** the server receives the block-gradient packet and
   chooses its application step.

The refresh graph is measurable at dispatch.  The scalar update mass is
measurable at return.  They cannot be represented as one simultaneous physical
action unless communication and rollout time are both zero.

## 2. Exact delivery drift

Let a dispatch for edge `j -> i` capture `v_j` while the recipient cache is
`chi_(j->i)`.  When delivery occurs at event `r`, the donor has reached
`theta_j^r`.  The cache-energy reduction is exactly

\[
R_{ij,r}=\frac{p_{ij}}2\left(
\|\theta_j^r-\chi_{j\to i}\|^2
-\|\theta_j^r-v_j\|^2
\right).
\tag{1}
\]

Writing `delta=v_j-chi_(j->i)` gives

\[
R_{ij,r}=\frac{p_{ij}}2\|\delta\|^2
+p_{ij}\langle\theta_j^r-v_j,\delta\rangle.
\tag{2}
\]

If the donor path from dispatch to delivery is bounded by `E_(j,r)`, the
dispatch-time predictable lower bound is

\[
R_{ij,r}\ge
\frac{p_{ij}}2\|\delta\|^2
-p_{ij}E_{j,r}\|\delta\|.
\tag{3}
\]

This is the correct refresh credit.  A delay can make it negative.

## 3. Exact return-time outgoing-cache drift

When owner block `i` applies a returned packet `g_i` with step `alpha`, every
cache in which `i` is a donor becomes more or less stale.  Define

\[
d_i^{\rm out}
=\sum_{\ell\ne i}p_{\ell i}
(\theta_i-\chi_{i\to\ell}),
\qquad
P_i^{\rm out}=\sum_{\ell\ne i}p_{\ell i}.
\tag{4}
\]

The exact outgoing-cache energy increment is

\[
-\alpha\langle d_i^{\rm out},g_i\rangle
+\frac{\alpha^2}{2}P_i^{\rm out}\|g_i\|^2.
\tag{5}
\]

Combining (5) with block smoothness of the learning objective gives

\[
\Delta(F+C)
\le
-\alpha\langle\nabla_iF+d_i^{\rm out},g_i\rangle
+\frac{\alpha^2}{2}(L_i+P_i^{\rm out})\|g_i\|^2.
\tag{6}
\]

Thus an update that descends `F` can still be poor for the composite Lyapunov
function because it makes many other workers strategically stale.  This is a
MARL-specific coupling term absent from shared-parameter worker-count tuning.

## 4. Pending-rollout history envelope

An update of block `i` also changes the staleness of every rollout already in
flight.  Suppose the nonnegative pending history term has the certified
increment

\[
\Delta\mathcal H_k(\alpha)
\le h_{i,k}\alpha\|g_i\|
+\frac{\kappa_{i,k}}2\alpha^2\|g_i\|^2,
\tag{7}
\]

where `h_(i,k)` and `kappa_(i,k)` are computed from packet birth versions,
public cross-block smoothness bounds, and the accepted policy path.  Then the
return-time scalar score is convex with certified alignment

\[
\mathcal A_{i,k}
=\langle\nabla_iF+d_i^{\rm out},g_i\rangle
-h_{i,k}\|g_i\|
\tag{8}
\]

and curvature

\[
\mathcal C_{i,k}=L_i+P_i^{\rm out}+\kappa_{i,k}.
\tag{9}
\]

Its closed-form minimizer is

\[
\alpha_k^*
=\Pi_{[0,\bar\alpha]}
\frac{[\mathcal A_{i,k}]_+}
{\mathcal C_{i,k}\|g_i\|^2}.
\tag{10}
\]

Gradient-estimation error replaces the alignment in (8) by a lower bound or
adds an expectation-level selection-error term.  Equation (10) itself does not
require the graph action to occur at return.

## 5. Causally coupled graph and step decisions

At dispatch, a candidate edge is valued by:

- its predictable delivery credit from (3);
- the estimated minimized future return score obtained from (6)--(10);
- its communication queue price;
- an explicit error allowance for the state changes before delivery and
  return.

At return, the server recomputes (10) using the received packet and current
debts.  Reoptimization is no worse than executing the provisional step under
the return-time certified score.  A theorem still must connect the
dispatch-time predicted score to that return-time score; the difference is the
pending-history/estimation term, not zero.

## 6. Implications for existing evidence

PDSG-001 combines delayed-snapshot cache replacement and the owner update in
one event.  Its oracle minimizes only the expected quadratic potential, not
the full composite Lyapunov function in (6).  Therefore:

- its 13.9878% median active gain remains valid as a problem-headroom result;
- it does not validate (3), (7), or the two-event controller;
- the failed unsigned controller remains failed;
- a successor experiment must use a new four-event runner frozen before its
  outcomes are observed.

## 7. Remaining theorem step

The next proof artifact must define an explicit pending history energy whose
event shifts imply (7) and telescope across all four event types.  It must
avoid charging the same source motion both in a gradient-bias radius and in
the history energy.  Only then can the stochastic expected-drift lemma be
summed into a finite-time stationarity and communication-budget theorem.
