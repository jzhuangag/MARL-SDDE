# T-031 theory program: dependence-adjusted delayed Markov learning

## Model and proof target

Let `m` agents observe one common mean operator `h(x)` through a jointly
stationary Markov process.  At server update `t`, a predictable subset `S_t`
returns gradients computed at agent-specific stale iterates.  For fixed
weights `a_t` supported on `S_t`, the update is

\[
x_{t+1}=\Pi\!\left[x_t-\alpha_t
 \sum_{i\in S_t}a_{t,i}H_i(x_{t-\tau_{t,i}},Z_{t,i})\right].
\]

Every marginal agent targets the same root.  Cross-agent dependence affects
only the innovation law, not the target operator.

At the root, define the long-run covariance blocks

\[
\Omega_{ij}=\sum_{\ell=-\infty}^{\infty}
 \operatorname{Cov}(\xi_{i,0},\xi_{j,\ell}).
\]

The selector must trade the quadratic noise term
`a_S^T Omega_S a_S` against contraction loss from stale parameters and the
three remaining budgets.  Delay may not appear as an unrelated final
constant.

## Theorem chain

### T1: dependence-adjusted finite-time upper bound

For strongly monotone affine Markov SA with certified geometric mixing,
bounded predictable delays, and predictable subset/weight choices, prove a
last-iterate and averaged-risk bound of the form

\[
\mathbb E\|e_T\|_W^2
\leq \mathcal C_T(S_{0:T-1},\tau_{0:T-1})\|e_0\|_W^2
+\sum_{t<T}\mathcal K_{t,T}
 \alpha_t^2 a_t^\top\Omega_{S_t}a_t
+\mathcal R_{\rm mix,delay}.
\]

The proof obligation is to retain identity-specific stale drift and
cross-stream long-run covariance after conditioning.  The existing Theorem 4
may be reused only after its cross-agent independence specialization is
removed rigorously.

### T2: matching Gaussian Markov lower bound

On a linear Gaussian AR Markov subclass with the same dependency graph and
delay/cost process, derive a local minimax or two-point lower bound whose
information term contains the same selected long-run covariance and usable
horizon.  This prevents the covariance term from being only an artifact of an
upper-bound proof.

### T3: count-only impossibility

Construct two instances with the same number of agents, marginal laws,
individual costs, and delay multiset, but different assignments of dependency
edges.  Show that any policy that uses only `q=|S|` incurs a nonvanishing or
growing excess finite-time risk relative to an identity-aware policy.  The
target statement is an `Omega(g)` effective-sample-size gap for `g`
dependency groups, subject to a matching upper construction.

### T4: low-complexity selector

For block-compound-symmetric dependence with certified group labels, fix a
participation count `m` and write the group allocation cost as

\[
\nu(k_1,\ldots,k_G)
=\frac{1}{m^2}\sum_g \sigma_g^2
 \{k_g+\rho_g k_g(k_g-1)\}
+D_T(k_g,\tau_g).
\]

When the theorem-derived delay term preserves discrete convexity, prove that
the marginal-cost heap/water-filling rule is exactly optimal.  Its target
runtime is `O(N+m log G)` and memory `O(N+G+Dd)`.  A small public grid of `m`
values can then be enumerated to handle overhead and budgets.

For a sparse nonnegative dependency graph, use cached marginal gains to
obtain a constant-factor approximation for a theorem-derived information
lower bound.  Runtime may depend on `|E|`; no linear-time claim is allowed for
a dense graph.

### T5: certified graph extension

If graph labels are learned, use a disjoint calibration tape and a
time-uniform, mixing-aware edge certificate.  Prove recovery or safe
coarsening only on a separated class.  AC-7 supplies the predictable adaptive
change-of-measure skeleton; unrestricted unknown mixing remains impossible.

### T6: SDDE bridge

Only after T1--T4 close, consider the scaling `alpha -> 0` with physical
delays held on the rescaled time axis.  Establish weak convergence of the
interpolated error process to a stochastic delay equation and control the
discretization remainder.  A Lyapunov--Krasovskii generator must recover the
same subset-dependent covariance/staleness score.  Until this is proved,
SDDE remains an interpretation layer and is excluded from the title.

## Algorithmic object

The working algorithm name is **Fresh-Diverse Greedy (FDG)**.  Each agent
profile contains a certified dependency group or sparse-edge sketch, a
marginal innovation bound, its observed age, and its resource costs.  In the
block theorem, the marginal cost of taking the next agent from group `g` is
the group's discrete-convex covariance increment plus the finite-time
freshness penalty.  The implementation uses one heap and enumerates at most a
small public grid of participation counts.

The deployed update uses only scalar weights and ordinary gradient/TD
aggregation.  It does not multiply by a proof metric and does not perform a
Hessian or covariance inversion.

## Proof order and kill gates

1. Prove T3 first; without a sharp count-only separation the new problem lacks
   a compelling reason to exist.
2. Close T1 and T2 on the same Gaussian/affine subclass and verify matching
   dependence and delay scaling.
3. Prove the exact block selector in T4 and implement its full-risk analytic
   oracle.
4. Run the CPU static gate and actual-learning pilot.
5. Add the sparse-graph and learned-certificate extensions only if the block
   theory and CPU learning effects survive.
6. Attempt T6 last.  Failure of T6 removes SDDE from the main claims but does
   not invalidate a strong discrete-time paper; failure of T1--T4 stops the
   ICML line.
