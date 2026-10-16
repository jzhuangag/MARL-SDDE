# Reusable long-horizon transfer statistics: closure and decision

2026-08-31. Internal AI-assisted derivation and deterministic qualification.
Parent: cefa5d5ff99c0ed212d3a43a71988d2a5dd2fe43.
This is not a new efficacy experiment, manuscript, or preregistration.

## Decision

**The parameter/direction/remaining-horizon reuse interface closes under a
fixed tabular learning law with independent conditional simulator access.**
Reuse does not require that TD dynamics preserve the donor subspace. The
implementation propagates the full anchor directions during one collection
phase and pays an explicit residual penalty at every later query.

**Reject caching alone as the final low-cost paper mechanism.** It is a
correct reference, not yet an affordable controller with a distinctive
learning benefit. The comparison to repeatedly rebuilding a full-horizon
estimator saves work; comparison to a cached short-unroll reference does not
establish an efficiency advantage. No new efficacy pilot is authorized.

The coherent problem remains long-horizon credit assignment for collaborative
training. This result resolves a specific implementation obstacle; it does not
turn caching, confidence intervals and a QP into three separate contributions.

## 1. Access, timing and fixed-law contract

Use the finite tabular fixed-policy model and bounded TD update in
observable_transfer_feasibility.md. Write B=R/(1-gamma), 0<eta<=1, and

\[
v^+=B_{sj}v+\eta r e_s,\qquad
B_{sj}=I-\eta e_s(e_s-\gamma e_j)^\top.
\]

Fix before collection: an anchor v0, an r-column matrix E, maximum horizon H,
return length L, n replicas per registered starting state, failure probability
delta and the registered state set S. Anchor v0 and v0+E[:,a] lie in [-B,B]^d.
The transition/reward law, fixed policy, representation, evaluation weighting
(uniform over d states), discount and TD step remain unchanged.

Collect independent reset-access replicas, separately for each s in S, as in
the preceding reference. Random K is uniform on 0,...,H-1; train the anchor
for K transitions and propagate all columns of E in the full d-dimensional
parameter space. Obtain an independent L-step return from an independently
uniform evaluation state. Every transition and reset is charged. All n and
all batches are fixed in advance; finalize collection before controlled
training. A changed parameter vector is allowed, but a changed sampling law
is not. A law-tag string guards configuration; it cannot prove stationarity.

For the sequential guarantee below, conditional on the completed cache and
the full observed training history, the recipient's next transition/reward
must have the registered law given its current state. Independent exogenous
recipient MRPs satisfy this assumption. Marginally Markov observations with
unobserved common noise need not. Enlarging the conditioning state can repair
that issue only when it is observed and its full cache cost is counted.
Cross-agent correlation is not silently covered by a marginal single-agent
Markov assumption. No nonlinear TD or joint-action policy-control claim follows.

## 2. One sample serves every remaining horizon

Let U_K be the fully propagated E. The original encoder emits

\[
\widehat G=H U_K[U,:]^\top U_K[U,:],\qquad
\widehat g=H U_K[U,:]^\top(v_K[U]-Y_L).
\]

Place these quantities in bin K. For every h<=H define

\[
\bar G_{s,h}=\frac1n\sum_{a=1}^n1\{K_a<h\}\widehat G_a,
\qquad
\bar g_{s,h}=\frac1n\sum_{a=1}^n1\{K_a<h\}\widehat g_a. \tag{1}
\]

The denominator is **all n replicas**, not the number with K<h. Since
P(K=t)=1/H, the H multiplier cancels this probability at each t<h. Thus (1)
is unbiased for the h-step anchor coefficients G_h,g_h defined by the finite
return target V_L. A cumulative sum over bins obtains all h without any new
transitions. Empty prefixes do not mean zero uncertainty.

Let Delta=max|E_ia| and m=|S| H [r(r+1)/2+r]. With

\[
c_n=\sqrt{2\log(2m/\delta)/n},\qquad
r_G=H\Delta^2c_n,\quad r_g=2H\Delta Bc_n,\quad
b_{g,h}=h\Delta B\gamma^L, \tag{2}
\]

the bounded-replicate argument from the preceding note gives one event of
probability at least 1-delta on which all state/horizon coefficients obey
entrywise errors at most r_G and r_g+b_g,h relative to the infinite-value
target. The H, not h, in the sampling radii is essential. The return-bias
bound uses h because it bounds the expectation summed over h times.

No independence between horizons is assumed: a union bound suffices. No
replica is added twice when queried again. This is a fixed-batch simultaneous
event, not optional-stopping or continuous-chain inference. Repeated queries
need no further union bound over parameter vectors because the next algebra
is uniform. Coverage conditional on an adaptively selected *query history*
must not be claimed anew as 1-delta; use the original joint event instead.

## 3. Exact reuse and an explicit out-of-anchor penalty

For a later bounded local value v and bounded donor endpoints v+D[:,j],
choose any coordinates c and A and compute the actual residuals

\[
v=v_0+Ec+e,\qquad D=EA+F.
\]

Coordinates may be signed and may depend on the completed cache. They need
not be orthogonal projections. For beta>=0, 1^T beta<=1 set u=A beta.
When e=F=0, affinity of the executed TD rule gives exactly

\[
A_h(v,D\beta)=u^\top G_hu+2u^\top(g_h+G_hc). \tag{3}
\]

This identity concerns full future local training, not a projection of its
dynamics. In particular U_t may leave span(E) during collection.

For nonzero residuals let

\[
\epsilon=\|e\|_\infty,\quad
\nu_j=\|F[:,j]\|_\infty,\quad b_j=\|EA[:,j]\|_\infty,
\quad a_j=\|A[:,j]\|_1.
\]

Each B_sj is nonnegative with row sums at most one. Its products therefore
do not increase infinity norms. At any future time write a0 for the predicted
current error, z for the propagated e, b0 for the predicted transfer and f
for the propagated F beta. The actual current error a=a0+z is bounded by 2B.
The difference between actual and projected squared-error contrasts is

\[
2zb_0+2af+2b_0f+f^2.
\]

Consequently its absolute h-step, uniform-state-averaged expectation is at most

\[
h\{2\epsilon b^\top\beta+4B\nu^\top\beta
       +2(b^\top\beta)(\nu^\top\beta)+(\nu^\top\beta)^2\}. \tag{4}
\]

The mixed quadratic in (4) is not in general convex. Do not call it a convex
QP. For an implementable convex upper bound use s_beta=1^T beta and

\[
2(b^\top\beta)(\nu^\top\beta)+(\nu^\top\beta)^2
\le (2b_{max}\nu_{max}+\nu_{max}^2)s_\beta^2. \tag{5}
\]

This term vanishes when F=0. The statistical part follows from
||u||_1<=a^T beta:

\[
r_G(a^\top\beta)^2+
2(r_G\|c\|_1+r_g+b_{g,h})a^\top\beta. \tag{6}
\]

The cross term r_G||c||_1 cannot be omitted when v changes. Combining (3)-(6)
gives the executed upper quadratic U_h(beta)=beta^T M beta+2 l^T beta with

\[
M=A^\top\bar G A+r_Gaa^\top
 +h(2b_{max}\nu_{max}+\nu_{max}^2)11^\top,
\]
\[
l=A^\top(\bar g+\bar Gc)
 +(r_G\|c\|_1+r_g+b_{g,h})a+h(\epsilon b+2B\nu). \tag{7}
\]

M is positive semidefinite. Uniformly over all valid queries and beta, on
the single coverage event A_h<=U_h. Zero transfer has exactly zero upper
advantage; execute a feasible candidate only if U_h<0. The existing bounded-
iteration projected-gradient QP solves (7); its stated real-arithmetic gap
bound is 2 lambda_max(M)/(2I). Numerical negativity is an execution check,
not a floating-point formal verification.

## 4. What sequential training guarantee follows

Fix T<=H controlled learning updates after cache collection and let the
registered set contain every reachable recipient state. The actual beta at
time t uses h=T-t, not a fixed receding lookahead falsely substituted for the
remaining horizon. Condition on a particular completed cache. On its good
event, (7) certifies every selected transfer at every reachable parameter/state.
The counterfactual baseline-value telescoping identity in
rl_collaboration_integration_decision.md then gives

\[
J_T(\pi\mid\mathrm{cache})\le J_T(\mathrm{local})
\quad\text{on the good cache event}.
\]

Here J_T is expected cumulative prediction risk over T learning updates;
the local baseline follows its own parameter history. The conditional Markov
law assumption in Section 1 justifies using the registered baseline value in
that telescoping step even though parameters and donors depend on the cache.
On the bad cache event every cumulative risk lies between 0 and 4TB^2.
Integrating over the once-built cache yields

\[
E[J_T(\pi)]-J_T(\mathrm{local})\le4TB^2\delta. \tag{8}
\]

This is not a claim of conditional coverage after each decision, pathwise
no-harm, terminal-error safety, general nonstationary learning or best-static
graph matching. Nor is (8) same-total-budget safety: both trajectories receive
T learning updates after the paid data-collection phase. Giving those probing
resources to an unprobed baseline changes the comparator. For multiple
recipients, their law-conditioning and failure-probability allocation must be
specified, not inferred from an aggregate queue. Stale donor endpoints are
allowed as the actual received D; delayed local TD updates need another law.

This is a sufficient fixed-law controller guarantee, derived here. The
bounded-variable, affine-sensitivity, convex-optimization and telescoping
ingredients are inherited; no novelty priority is asserted.

## 5. Cost and matched short-unroll comparison

For S=|S| cached initial states the expected collection cost is
S n [(H-1)/2+L] transitions, S n (H-1+L) at worst, and 2 S n resets.
The prefix coefficients need S H (r^2+r) scalars. Holding the anchor needs
d(r+1) additional scalars; collection workspace, counters and object overhead
are additional. Tabular build arithmetic is
O(S n [dr+Hr+L+r^2]+S H r^2). Each query, with supplied c,A, costs
O(dr(k+1)+r^2 k+r k^2) before the k-dimensional QP. Donor messages cost dk
parameter scalars per query; computing alignment/coordinates is additional.
This is neither constant memory in H nor an O(kd) total algorithm.

For H=64,S=2,n=128,L=32,r=2 and one query at every remaining h=64,...,1:

- cache: 16,256 expected transitions, 512 resets, 768 prefix coefficients;
- rebuilding a reference at each h with the same n: 391,168 expected
  transitions, 16,384 resets; ratio 24.063;
- cached H=8 reference: 9,088 expected transitions; the long cache costs
  1.7887 times as many transitions;
- the long-cache cost alone is 254 transitions per controlled learning update.

These are arithmetic counts, not measured runtimes, actual generated
trajectories, sample-complexity equivalence, or a lower bound on necessary cost.
At small h, the all-prefix estimator retains H in its confidence width;
rebuilding at h can have tighter bounds with the same n. Caching can also be
applied to the short-unroll competitor, so comparing only with an uncached
long-unroll method would overstate the advantage.

For a matched short m-step quadratic and h>m, the missing future advantage
obeys, with d_j=||D[:,j]||_infinity,

\[
|A_h(\beta)-A_m(\beta)|
\le(h-m)\{(d^\top\beta)^2+4B d^\top\beta\}. \tag{9}
\]

This follows by nonexpansivity at each omitted time. The implemented short-
unroll reference adds that convex tail penalty to obtain a valid h-step upper
bound. Running a short meta-gradient without a tail certificate remains a
legitimate empirical competitor, but it is not the same guarantee. A sharper
tail bound needs a proved contraction/integrated-sensitivity property; merely
discarding the tail is not a solution. Meta-Gradient RL already propagates
learning sensitivities and uses subsequent validation data [2]. Its original
meta-parameters are return parameters, not the transfer weights here.

## 6. Completed qualification and what it does not show

Exact enumeration on the declared two-state MRP verifies (1) and (3) across
both initial states, every h=1,2,3, changed parameters and signed coordinates.
The maximum affine-identity discrepancy is 1.11e-16. A query with nonzero e,F
has true finite-return advantage -0.17590 and conservative upper +0.24716:
correctness of the shield does not imply that it detects this beneficial query.

An independently stated algebraic non-vacuity witness uses H=2, gamma=.2,
eta=.5, reward bound R=1, actual zero rewards, v0=(1,-1), E=(-1,1), L=6 and
the **hypothetical** n=4096 per-state radii. At exact population coefficients,
the QP selects beta=.76388 with upper -1.03868 and true advantage -1.56745.
No 4096-replica batch was collected: this demonstrates algebraic possibility,
not empirical confidence coverage, affordable activation or experiment success.

Tests verify normalization, no duplicate counting, incomplete-batch rejection,
law/state/update guards, immutable finalized coefficients, zero-action safety,
parameter uncertainty, residuals and omitted-tail accounting. Source code and
result hashes, regression and replay are recorded separately. No frozen data,
controller, gate, seed or T-083A result has been changed.

## 7. Next decision, not another performance run

Do not promote this reference or launch a cache-versus-weak-baseline pilot.
Its algebraic interface is resolved; its empirical affordability is not.
Do not conclude universal impossibility from these conservative bounds or
from the public n=128 arithmetic example.

The next bounded question is whether an **observable integrated-sensitivity
bound** permits a genuinely cheaper long-horizon certificate than the cached
short-unroll reference. Specify a contraction/data-access class, derive the
omitted-tail bound and its own certification cost, and compare the resulting
minimum stated resource requirements before any efficacy protocol. Reusing
the same affordable certificate should support both long and short baselines.
If it supplies no useful distinction, stop this mechanism rather than
rebranding generic meta-gradient caching as an ICML contribution. Do not add
nonstationarity, SDDE or new experiment numbers to avoid this test.

### A concrete conditional route for the next implementation

The following derivation specifies the next testable interface; it is not an
empirical contraction certificate. Fix a block length m0 before independent
collection. Because products T_m0=B_(m0-1)...B_0 are nonnegative,

\[
\|T_{m0}\|_\infty=\|T_{m0}1\|_\infty\in[0,1].
\]

Propagating the single vector initially equal to 1 therefore observes this
operator norm without constructing a d-by-d matrix. In tabular TD each row
update is constant arithmetic; the vector uses O(d) space. The transitions
and state resets still cost resources. With n_c independent m0-step replicas
per state, a one-sided bounded-variable union bound certifies, on one event,

\[
\max_s E[\|T_{m0}\|_\infty\mid S_0=s]
\le\widehat\kappa
:=\min\{1,\max_s\bar z_s+
\sqrt{\log(|S|/\delta_c)/(2n_c)}\}.
\]

All reachable conditioning states must be covered, not just the currently
visited ones. If kappa_hat<1 and the fixed conditional Markov law holds,
submultiplicativity and conditioning at each block boundary imply
E||T_(j m0)||_infinity<=kappa_hat^j. The argument is not independence between
successive blocks: the next block's conditional mean bound holds uniformly
at its observed starting state. Any partial last block has norm at most one.
Thus the tail coefficient in (9), for a lookahead m0, can be replaced by

\[
\Phi(h,m_0,\widehat\kappa)
=\sum_{t=m_0}^{h-1}\widehat\kappa^{\lfloor t/m_0\rfloor}
\le\frac{m_0\widehat\kappa}{1-\widehat\kappa}. \tag{10}
\]

Multiply Phi by [(d^T beta)^2+4B d^T beta] and add it to the m0-step upper
quadratic. If kappa_hat=1, retain the finite h-m0 tail; never divide by zero
or declare contraction. For m0<d, each trajectory leaves some state row
unupdated, so the row-sum norm is exactly one: this particular certificate
cannot then show strict contraction. More generally gamma<1 alone does not
guarantee that a finite block visits every state.

This route costs |S| n_c m0 additional transitions and |S| n_c resets unless
a valid shared-batch construction is separately established. Predetermining
delta_c and the coefficient-cache probability allocation yields a joint
event by a union bound; dependence between the two events is not a problem.
Do not choose m0,n_c by repeatedly peeking at these same samples. This exact
conditional derivation avoids full-horizon propagation but does not establish
an affordable n_c, useful tail width, payoff after costs, or novelty. Those
are the next qualification, using the same certificate for short-unroll
comparators, not assumed positive outcomes.

## Sources and verification boundary

[1] Hoeffding, Probability Inequalities for Sums of Bounded Random Variables,
JASA 58(301), 13-30, 1963, DOI
[10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).
Used only for the bounded-replicate concentration ingredient; the applicable
bound and union construction are explicit above and in the preceding note.

[2] Xu, van Hasselt and Silver, Meta-Gradient Reinforcement Learning, arXiv
[1805.09801v1](https://arxiv.org/html/1805.09801v1), 2018, Section 1, equations
(1)-(5). The reuse comparison is our algebraic analysis, not a claim that the
source already proves (7)-(8) or an assertion that all meta-gradient methods
lack long-horizon theory. This is not an exhaustive novelty review.

Fresh metadata checks and unavailable resolvers are recorded in
reusable_transfer_cache_source_verification.json. This note is not a final
paper/bibliography delivery.
