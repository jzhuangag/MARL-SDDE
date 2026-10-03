# Conditional performance bound for Lyapunov-clocked optimism

## Certificate interface

Let \(z_k\) be the lifted delayed state and let \(c_k\) include the predictable
local game-geometry certificate and the identity of the agent whose packet has
arrived.  Assume a common metric \(P\succ0\) and positive multipliers
\(q_{c,u}\) give a pathwise or simultaneously high-probability certificate

\[
z_{k+1}^{\top}Pz_{k+1}
\le q_{c_k,u_k}z_k^{\top}Pz_k+\nu_k,
\qquad u_k\in\{0,1\},
\]

where zero denotes a cheap stale coordinate update and one denotes a fully
charged arrival-fresh optimistic anchor.  For a finite lifted linear game,

\[
q_{c,u}=\lambda_{\max}\!\left(
P^{-1/2}M_{c,u}^{\top}PM_{c,u}P^{-1/2}
\right)
\]

is directly checkable after the arrival is known.  An expected multiplier over
random arrivals is also useful for an iid mean-square theorem, but it must not
be substituted for this pathwise certificate in an arbitrary-context claim.
A metric can be constructed from a stable reference
anchor kernel through the discrete Lyapunov equation

\[
P=I+\mathbb E[M_{c,1}^{\top}PM_{c,1}].
\]

The current code solves this equation, verifies positive definiteness, and
checks the generalized eigenvalue inequality.  This is a theorem tool, not an
assumption that the nonlinear Markov-game certificate is already available.

## Drift-plus-log-contraction rule

Define \(\ell_{c,u}=\log q_{c,u}\) and resource debt

\[
Z_{k+1}=[Z_k+u_k-\bar u]_+.
\]

The causal action minimizes

\[
V\ell_{c_k,u}+Z_ku
\]

over the hard-budget-feasible actions.  Equivalently, it buys the anchor iff

\[
V(\log q_{c_k,0}-\log q_{c_k,1})>Z_k.
\]

Thus the same decision has an exact learning interpretation: it spends one
resource unit only when the certified reduction in log energy exceeds the
current Lyapunov price.

## Finite-horizon constrained bound and its exact scope

Because

\[
\frac12(Z_{k+1}^2-Z_k^2)
\le \frac12+Z_k(u_k-\bar u),
\]

the standard drift-plus-penalty comparison yields, **under iid certificate
contexts and a stationary randomized comparator whose average anchor use is at
most \(\bar u\)**,

\[
\frac1K\sum_{k<K}\ell_{c_k,u_k}
\le
\frac1K\sum_{k<K}\ell_{c_k,u_k^{\pi}}
+\frac{1}{2V}+\frac{Z_0^2}{2VK},
\]

in expectation, up to the usual terminal-debt term.  The independence is used
to make
\(\mathbb E[Z_k(u_k^{\pi}-\bar u)]\le0\).  For an arbitrary Markov context,
the queue and context are correlated and this cancellation is not automatic;
a mixing/blocking or online primal--dual regret term is required.  The
corresponding average resource excess is bounded by \(Z_K/K\); a hard
remaining-budget guard makes pathwise overshoot zero.

### Exact finite-horizon budget without a hard-guard proof mismatch

A hard guard changes the feasible action set after the call budget is
exhausted, so the unconstrained per-step comparison used above no longer holds
for a comparator that calls the oracle later.  The following reserve removes
that mismatch for the executable controller.

Assume the certified log benefit is uniformly bounded,

\[
0\vee(\ell_{c,0}-\ell_{c,1})\le \Delta_{\max}.
\]

Under the strict decision rule
\(u_k=1\{V(\ell_{c_k,0}-\ell_{c_k,1})>Z_k\}\), induction gives

\[
0\le Z_k<V\Delta_{\max}+1=:Z_{\max}.
\]

For an integer finite-horizon allowance \(B\), run the same controller without
a post-hoc hard guard using the nominal queue arrival

\[
\bar u'=\frac{B-Z_{\max}}{K},
\qquad B\ge Z_{\max}.
\]

Because the queue update always satisfies
\(Z_{k+1}\ge Z_k+u_k-\bar u'\), telescoping yields the deterministic bound

\[
\sum_{k<K}u_k
\le K\bar u'+Z_K
<B.
\]

Thus an integer number of calls is at most \(B\) for every context sequence;
no iid, Markov, or concentration assumption is needed for the accounting
claim.  The reserve is \(O(V\Delta_{\max})\), so it is useful only when the
horizon budget exceeds this transient.  The code exposes this construction as
`finite_horizon_budget_reserve` and exhaustively checks all length-eight
strings over four signed log-gain levels.

In the noiseless linear case, the exact deterministic implication is instead:
if the **realized** certified cumulative log multiplier obeys
\(\sum_{k<K}\log q_{c_k,u_k}\le-\kappa K\), iteration of the pathwise
quadratic certificate gives

\[
\mathbb E\|z_K\|_P^2
\le e^{-\kappa K}\|z_0\|_P^2.
\]

With bounded additive term \(\nu_k\le\bar\nu\), the same realized product
recursion gives an explicit geometrically weighted noise floor.  An expected
negative average log multiplier alone is insufficient to claim an expected
energy contraction via Jensen; bounded-increment martingale concentration or a
uniform conditional multiplier bound is additionally required.  For Markov
certificates, mixing affects both the validity of \(q_{c,u},\nu_k\) and the
queue-comparator analysis.

## What is proved and what remains open

The following pieces are exact for the lifted linear subclass:

- heterogeneous-arrival phase boundary at zero delay;
- failure of stale-only extra-gradient in registered delayed examples;
- lifted second-moment spectral criterion;
- positive-definite metric construction from a stable fresh-anchor kernel;
- generalized multiplier audit and drift-plus-log scheduling bound.
- bounded virtual debt and an exact finite-horizon call budget obtained by
  reserving its worst-case terminal value.

The ICML-level theorem is **not closed** until all of the following hold for
the executable Markov policy-gradient algorithm:

1. an independently sampled or time-uniform confidence construction for local
   symmetric/skew geometry and every multiplier used before the action;
2. a common or slowly varying metric under nonlinear policy changes, with the
   metric-variation debt explicitly charged;
3. a bound for delayed Markov policy-gradient bias and critic error;
4. a mixing/martingale argument converting the stochastic DPP comparison into
   a high-probability cumulative log contraction, followed by conversion of
   lifted energy to a last-iterate VI or Nash gap;
5. a resource comparator that includes fresh-rollout, backward-pass,
   communication, and commit-barrier wall-clock costs.

The first CPU phase/headroom scan is restricted to the current-oracle
\(D=0\) contract, where the clock-balanced metric and phase boundary are
exact.  The delayed lifted machinery remains a falsification/extension audit.
A positive linear scan alone cannot authorize a GPU experiment; Markov-noise
certificates and a nonlinear actor--critic interface must still close.
