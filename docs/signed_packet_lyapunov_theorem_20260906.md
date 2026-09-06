# Paired Lyapunov theorem for signed policy-version packets

Date: 2026-09-06

Status: conditional discrete theorem and low-complexity design interface.
This document does not assert that a neural critic already supplies the
required score certificate, and it does not authorize a formal or GPU run.

## 1. Why the graph must be signed

An asynchronous owner-`i` rollout is launched with the current owner policy
and a cache of teammate policy versions.
A refresh edge `j -> i` changes the conditional mean of the future packet,
not merely its norm.
The useful quantity is therefore its signed alignment with the current
learning direction.
Reducing an unsigned staleness radius can waste communication on a large but
helpful stale component or remove a small but harmful one.

The unified problem is to choose one causal policy-version refresh at packet
birth so as to minimize certified future learning drift while a virtual queue
enforces the long-run message budget.
The packet may return after other policy blocks have changed, so the proof
must pair its launch decision with its own receipt rather than pretending that
both occur in one synchronous iteration.

## 2. Launch and receipt model

Index launches by `p=0,...,N-1` and let `r(p)` be the receipt event of packet
`p`.
At launch, owner `i_p` has an eligible causal-cone action set

\[
\mathcal A_p=\{\varnothing\}\cup
\{\{j\mathbin\to i_p\}:j\in\mathcal C_p\}.
\tag{1}
\]

Thus a launch refreshes no edge or one edge.
Repeated launches produce a dynamic directed graph, while the exact action
scan remains linear in the local interaction degree `Delta_p=|C_p|`.
The rollout horizon and receipt step cap `w_p` are public and fixed for the
primary algorithm.

For action `a in A_p`, let the packet gradient have launch-conditional mean
`mu_p(a)` and second moment `m_p(a)`.
Let

\[
g_p^0=\nabla_{i_p}F(\theta^{b(p)})
\tag{2}
\]

be the joint-potential block gradient at birth.
Assume block `i_p` is `L_(i_p)`-smooth and the update sample is conditionally
unrevealed to all intervening learner events.
This condition holds for independent worker random-number streams because the
learner does not use packet `p` before its receipt.
Continuing-chain implementations require an explicit martingale or
regenerative replacement.

Let `d_p(a)` be a predictable bound satisfying

\[
\left\|
\mathbb E[\nabla_{i_p}F(\theta^{r(p)})\mid\mathcal F_{b(p)}]
-g_p^0
\right\|\le d_p(a).
\tag{3}
\]

Policy-version paths, an in-flight cap, and block cross-smoothness can produce
this bound.
The launch score is

\[
D_p(a)=
-w_p\langle g_p^0,\mu_p(a)\rangle
+\frac{L_{i_p}w_p^2}{2}m_p(a)
+w_p d_p(a)\|\mu_p(a)\|.
\tag{4}
\]

By block smoothness, conditional independence, and Cauchy--Schwarz,

\[
\mathbb E[
F(\theta^{r(p)+})-F(\theta^{r(p)})
\mid\mathcal F_{b(p)}]
\le D_p(a).
\tag{5}
\]

Equation (5) is the exact launch-to-receipt bridge.
It prices learning progress, packet noise, and all policy motion that occurs
before the packet is consumed.

## 3. Lyapunov design rule

Let messages have cost `c_p(a)` and average budget `bar c`.
Use the virtual queue

\[
Q_{p+1}=[Q_p+c_p(a_p)-\bar c]^+.
\tag{6}
\]

Suppose a centralized critic and sparse Jacobian-vector products provide a
predictable estimate `Dhat_p(a)` with simultaneous error

\[
\max_{a\in\mathcal A_p}|\widehat D_p(a)-D_p(a)|\le\epsilon_p.
\tag{7}
\]

The executed graph action is

\[
a_p\in\arg\min_{a\in\mathcal A_p}
\{V\widehat D_p(a)+Q_pc_p(a)\}.
\tag{8}
\]

This is a Lyapunov design action, not a post-hoc stability check.
It minimizes the certified paired drift of the learning potential and resource
queue.
The null-plus-one-edge action set makes (8) an exact `O(Delta_p)` scan.
No dense Hessian, matrix inverse, generic quadratic program, or environment
probe per edge is required.

For the selected action and the true minimizer of the same index,

\[
V D_p(a_p)+Q_pc_p(a_p)
\le
\min_a\{V D_p(a)+Q_pc_p(a)\}+2V\epsilon_p.
\tag{9}
\]

The factor two follows from the uniform error at the selected and comparator
actions.

## 4. Main paired-drift theorem

Assume (a) every launched packet is eventually received during a terminal
drain phase; (b) (5) and (7) hold; (c) costs lie in `[0,c_max]`; and (d) there
is a predictable comparator `a_p^circ` with

\[
\mathbb E[c_p(a_p^\circ)-\bar c\mid\mathcal F_{b(p)}]\le0
\tag{10}
\]

and

\[
D_p(a_p^\circ)
\le-\kappa_p\|g_p^0\|^2+R_p,
\qquad \kappa_p>0.
\tag{11}
\]

Then the signed policy-version scheduler (8) satisfies

\[
\sum_{p<N}\kappa_p\mathbb E\|g_p^0\|^2
\le
F(\theta^0)-F_\star
+\frac{Q_0^2}{2V}
+\sum_{p<N}\mathbb E[R_p+2\epsilon_p]
+\frac{NB_Q}{V},
\tag{12}
\]

where `B_Q` may be any uniform upper bound on
`(c_p(a)-bar c)^2/2`.

### Proof

The half-square queue inequality gives

\[
\frac{Q_{p+1}^2-Q_p^2}{2}
\le Q_p(c_p(a_p)-\bar c)+B_Q.
\tag{13}
\]

Pair the queue change at launch `p` with the objective change at its own
receipt `r(p)`.
Equation (5), followed by (9), bounds their conditional sum by the same pair
under `a_p^circ`, plus `2V epsilon_p+B_Q`.
After expectation, (10) removes the comparator queue term.
Summing in launch order is valid even though receipts occur in another order:
the objective changes telescope in chronological receipt order and every
packet appears exactly once.
Insert (11), telescope `Q_p^2/2`, use `F(theta)>=F_star` and `Q_N^2>=0`, and
divide by `V`.
This proves (12).

If packets remain in flight, (12) carries an explicit terminal boundary equal
to a valid upper bound on their unmatched paired drift.
They cannot be silently counted as completed.

Queue iteration also yields

\[
\frac1N\sum_{p<N}c_p(a_p)
\le\bar c+\frac{Q_N-Q_0}{N}.
\tag{14}
\]

When the null action is free, every positive action costs at least `c_min`,
and estimated drift scores are uniformly bounded, the standard DPP threshold
gives a deterministic `O(V)` queue cap.
Thus `V=sqrt(N)` gives an `O(N^(-1/2))` average-budget remainder.
The stationarity rate is governed by the comparator remainder and score error;
the theorem does not call a fixed error neighborhood exact convergence.

## 5. Favorable and unfavorable phases

Suppose one comparator packet obeys
`||mu_p-g_p^0||<=B_p`, has conditional variance at most `sigma_p^2`, and has
receipt-gradient motion at most `d_p`.
Writing `A_p=||g_p^0||`, equation (4) gives

\[
D_p(a_p^\circ)
\le-w_p(A_p^2-A_pB_p)
+\frac{L_{i_p}w_p^2}{2}
\{(A_p+B_p)^2+\sigma_p^2\}
+w_pd_p(A_p+B_p).
\tag{15}
\]

Hence (11) holds with `kappa_p=kappa w_p`, `0<kappa<1`, whenever

\[
(1-\kappa)A_p^2
\ge A_pB_p+
\frac{L_{i_p}w_p}{2}\{(A_p+B_p)^2+\sigma_p^2\}
+d_p(A_p+B_p).
\tag{16}
\]

This is the coherent phase statement.
Sparse policy-version refresh is valuable when it reduces `B_p` enough to
cross (16); large random delay increases `d_p`, and weak signal or large
Markov noise makes every safe graph action indistinguishable.
The result does not claim universal gain in the latter phase.

## 6. What remains before preregistration

1. Replace the exact quadratic launch score by an observable centralized-
   critic/JVP estimate and establish (7) under a stated Markov sampling model.
2. Derive `d_p(a)` from recorded policy-version paths and a bounded in-flight
   condition without using future information.
3. Verify that one-edge signed scheduling retains at least 10% equal-resource
   terminal-risk headroom against every fixed one- and two-edge graph and
   standard online freshness heuristic on independent CPU development seeds.
4. Freeze the estimator, seeds, comparator-selection rule, terminal and AUC
   gates, cost accounting, and reproduction hashes in a separate commit.
5. Only after that CPU pilot passes should a standard MARL GPU experiment be
   designed.

The controlled SDDE is optional.
If retained, its role is to approximate how `d_p`, noise, and queue pressure
move the phase boundary in (16).
The discrete paired theorem (12) remains the primary correctness guarantee
unless a finite-horizon weak-approximation error is proved.
