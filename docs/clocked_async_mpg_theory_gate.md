# Clocked asynchronous MPG: first theory gate

Status: algebra and filtration audit only.  No convergence theorem, efficacy
experiment, seed registry or GPU work is authorized.

## 1. The filtration issue comes first

Let a policy-gradient packet for actor `i` be born at event `b` and computed
from a fixed-length Markov segment `Xi_b`.  At arrival event `k`, the exact
identity is

\[
\widehat g_{i,k}-\nabla_i\Phi(\theta^k)
=M_{i,b}+B_{i,b}+S_{i,b,k},
\]

with

\[
\begin{aligned}
M_{i,b}&=\widehat g_{i,k}
-\mathbb E[\widehat g_{i,k}\mid\mathcal F_b],\\
B_{i,b}&=\mathbb E[\widehat g_{i,k}\mid\mathcal F_b]
-\nabla_i\Phi(\theta^b),\\
S_{i,b,k}&=\nabla_i\Phi(\theta^b)-\nabla_i\Phi(\theta^k).
\end{aligned}
\]

`M_(i,b)` is centered at the **birth filtration**, not automatically at the
arrival filtration.  If completion time depends on the realized trajectory or
gradient noise, arrival order is informative.  The executable counterexample
uses two independent Rademacher noises.  Positive noise completes at time zero
and negative noise at time one.  Although each worker's noise has mean zero,
the first-arrival noise has mean `0.5`; under exogenous completion it has mean
zero.  Thus an asynchronous proof may not silently condition on arrival and
retain unbiasedness.

The prospective theorem must therefore do one of the following:

1. use fixed-length rollouts and assume completion is conditionally independent
   of trajectory innovations given birth information;
2. analyze the marked completion process without claiming an arrival-time
   martingale difference; or
3. introduce and fully analyze a valid correction for informative completion.

The first option is the bounded initial scope.  Variable episode length or
state-dependent simulation time is outside the first theorem unless option 2
or 3 is completed.

## 2. Markov and teammate-drift terms

Under a declared geometric-mixing/burn-in condition, the Markov term must have
an explicit conditional bound such as

\[
\|B_{i,b}\|\le \varepsilon_{\rm mix}(m,b),
\]

where `m` is the charged trajectory length.  No independence between
overlapping packets follows from this inequality.

For coordinate-wise cross smoothness,

\[
\|\nabla_i\Phi(x)-\nabla_i\Phi(y)\|
\le \sum_j L_{ij}\|x_j-y_j\|,
\]

the stale teammate term obeys

\[
\|S_{i,b,k}\|
\le \sum_j L_{ij}\|\theta_j^k-\theta_j^b\|.
\]

The tests verify this inequality on coupled concave quadratic potentials.  They
also give two histories with identical scalar age but mismatch `1.5` and `0`,
respectively, because only one history changes a coupled teammate block.  This
supports using an interaction-weighted drift in the theorem; it does not make
the matrix `L` observable in neural MARL.

## 3. One-event potential inequality

For an `L_i`-smooth block and an update using the stale direction
`g_hat_(i,k)`, the deterministic lower bound is

\[
\Phi(\theta^{k+1})-\Phi(\theta^k)
\ge
\alpha_k\langle\nabla_i\Phi(\theta^k),\widehat g_{i,k}\rangle
-\frac{L_i\alpha_k^2}{2}\|\widehat g_{i,k}\|^2.
\]

The implementation verifies equality for every tested quadratic coordinate
update.  Combining it with the decomposition above is valid, but a finite-time
rate still requires a careful conditional expectation and control of the
history-dependent stale term.

## 4. Lyapunov--Krasovskii history identity

For maximum event delay `D`, let `s_r=theta^(r+1)-theta^r` and

\[
H_k=\sum_{r=k-D}^{k-1}(r-(k-D)+1)\|s_r\|^2.
\]

The exact shift identity is

\[
H_{k+1}-H_k
=D\|s_k\|^2-\sum_{r=k-D}^{k-1}\|s_r\|^2.
\]

The negative history sum is the term that can absorb delayed cross-block
energy; the positive new-step term modifies the allowable step size.  The
identity is verified exactly by the tests.  It does not yet prove that one
single coefficient closes all block, critic and Markov terms.

## 5. Gate result

Closed at this stage:

- exact birth-filtration decomposition;
- explicit informative-completion counterexample;
- interaction-weighted quadratic mismatch bound;
- exact quadratic one-event smoothness certificate;
- exact delay-history drift identity.

Still open and mandatory:

- a trajectory-level mixing bound for the actual policy-gradient/critic
  estimator;
- a valid event filtration under overlapping packets;
- coefficients that close the composite Lyapunov drift;
- conversion to a Markov-game Nash-gap criterion;
- renewal-time conversion and a nonempty fully charged speedup region;
- proof that the result is not a direct corollary of existing delayed Markov
  stochastic approximation.

The gate therefore records **continue theory only**, not pass to experiments.

Verification on the final source state:

- targeted theory-gate tests: `7 passed in 0.16s`;
- complete experiment regression: `931 passed, 7 skipped in 123.00s`.
