# Exact-gradient event-time Lyapunov lemma

Status: proved intermediate lemma for smooth finite-dimensional potentials.
It is not yet a Markov-game, actor--critic or wall-clock theorem.

## Setting

For clarity write the problem as minimizing `f=-Phi`.  Partition `x` into
scalar blocks and suppose

\[
|\nabla_i f(x)-\nabla_i f(y)|
\le \sum_j L_{ij}|x_j-y_j|,
\qquad L_{ij}\ge0.
\]

At event `k`, block `I_k=i` is drawn independently with probability `p_i>0`.
The exact gradient is evaluated at a predictable stale iterate
`x^(b_(k,i))`, where `0 <= k-b_(k,i) <= D`, and

\[
x^{k+1}=x^k-\alpha U_i\nabla_i f(x^{b_{k,i}}).
\]

This lemma assumes exact gradients, iid activation independent of the current
iterate, a deterministic maximum event delay and unconstrained Euclidean
updates.  None may be silently imported into the final MARL theorem.

Define

\[
\ell_i=\sum_jL_{ij},
\qquad
w_j=\sum_i p_i\ell_iL_{ij},
\]

and, with `s_r=x^(r+1)-x^r`, the interaction-weighted history

\[
H_k=\sum_{r=k-D}^{k-1}(r-(k-D)+1)
    \sum_jw_j|s_{r,j}|^2.
\]

## Proposition

Assume `f` is bounded below and choose a constant step satisfying, for every
block `i`,

\[
L_{ii}\alpha+D^2w_i\alpha^2\le1. \tag{1}
\]

Then the Lyapunov--Krasovskii functional

\[
\mathcal V_k=f(x^k)+\frac{\alpha D}{2}H_k
\]

obeys

\[
\mathbb E[\mathcal V_{k+1}\mid\mathcal F_k]
\le \mathcal V_k
-\frac{\alpha}{2}\sum_i p_i|\nabla_i f(x^k)|^2. \tag{2}
\]

Consequently, for `p_min=min_i p_i`,

\[
\frac1K\sum_{k=0}^{K-1}\mathbb E\|\nabla f(x^k)\|^2
\le
\frac{2(\mathcal V_0-f_{\inf})}{\alpha p_{\min}K}. \tag{3}
\]

The largest step allowed by (1) is the minimum blockwise positive root

\[
\alpha_{\max,i}=
\begin{cases}
\dfrac{-L_{ii}+\sqrt{L_{ii}^2+4D^2w_i}}{2D^2w_i},&D^2w_i>0,\\
1/L_{ii},&D^2w_i=0,\ L_{ii}>0,\\
+\infty,&D^2w_i=L_{ii}=0.
\end{cases}
\]

For `D=0`, this recovers the standard block-smooth step restriction.

## Proof

Let `g_i^k=nabla_i f(x^k)` and `g_i^b=nabla_i f(x^b)`.  Block smoothness and
the polarization identity give

\[
\begin{aligned}
f(x^{k+1})-f(x^k)
&\le-\alpha\langle g_i^k,g_i^b\rangle
  +\frac{L_{ii}\alpha^2}{2}|g_i^b|^2\\
&=-\frac\alpha2|g_i^k|^2
  -\frac\alpha2(1-L_{ii}\alpha)|g_i^b|^2
  +\frac\alpha2|g_i^k-g_i^b|^2. \tag{4}
\end{aligned}
\]

Weighted Cauchy--Schwarz, followed by the delay path bound, yields

\[
|g_i^k-g_i^b|^2
\le \ell_i\sum_jL_{ij}|x_j^k-x_j^b|^2
\le D\ell_i\sum_{r=k-D}^{k-1}\sum_jL_{ij}|s_{r,j}|^2. \tag{5}
\]

Averaging (5) over `I_k` produces

\[
\mathbb E[|g_{I_k}^k-g_{I_k}^b|^2\mid\mathcal F_k]
\le D\sum_{r=k-D}^{k-1}\sum_jw_j|s_{r,j}|^2. \tag{6}
\]

The exact history shift is

\[
H_{k+1}-H_k
=D\sum_jw_j|s_{k,j}|^2
-\sum_{r=k-D}^{k-1}\sum_jw_j|s_{r,j}|^2. \tag{7}
\]

The negative term in (7), multiplied by `alpha D/2`, cancels (6).  Since only
block `i` moves and `s_(k,i)=-alpha g_i^b`, the remaining new-history cost is

\[
\frac{\alpha^3D^2}{2}\sum_i p_iw_i|g_i^b|^2. \tag{8}
\]

Combining (4)--(8), condition (1) makes every coefficient of `|g_i^b|^2`
nonpositive, leaving (2).  Summing (2), using
`sum_i p_i|g_i|^2 >= p_min||grad f||^2`, and lower-bounding the functional by
`f_inf` proves (3).

## What this closes and what it does not

The proposition rigorously connects asynchronous block interaction, delay and
a Lyapunov history energy.  It also exposes a conservative `D^2` stability
price and activation penalty `p_min`.  Both are targets for a sharper
instance-sensitive analysis, not results to hide.

It is currently a generic delayed block-coordinate lemma.  It does **not** yet
pass the paper novelty gate.  The next proof must add the actual Markov policy-
gradient/critic decomposition, retain distinct-agent game semantics, and
convert stationarity to a standard Nash-gap criterion.  If those steps merely
substitute constants into generic delayed stochastic approximation, the
candidate must stop.

The implementation verifies the history weights, maximum-step roots and (2)
on random coupled positive-semidefinite quadratic systems.  These deterministic
checks validate algebra, not stochastic coverage or experimental efficacy.

## Conditional-noise corollary

Suppose the stale block estimator is

\[
\widehat g_i^b=\nabla_i f(x^b)+\xi_{i,k},
\quad
\mathbb E[\xi_{i,k}\mid\mathcal G_k^-,I_k=i,b_{k,i}]=0,
\quad
\mathbb E[|\xi_{i,k}|^2\mid\mathcal G_k^-,I_k=i,b_{k,i}]\le\sigma_i^2.
\]

Here `G_k^-` is the pre-application sigma-field: it contains all past applied
updates and the predictable metadata of the arriving packet, including its
agent and birth index, but excludes that packet's as-yet-unrevealed innovation.
This conditioning is a substantive assumption.  It requires non-informative
completion and packet innovations that remain centered after conditioning on
all metadata used to select the arrival.  It is not implied by marginally
unbiased policy gradients, and it fails for first-arrival selection when
completion time depends on the sampled trajectory or gradient noise.

Repeating the proof and retaining the second-moment terms gives

\[
\begin{aligned}
\mathbb E[\mathcal V_{k+1}\mid\mathcal F_k]
\le{}&\mathcal V_k
-\frac\alpha2\sum_i p_i|\nabla_i f(x^k)|^2\\
&+\frac12\sum_i p_i
\left(L_{ii}\alpha^2+D^2w_i\alpha^3\right)\sigma_i^2. \tag{9}
\end{aligned}
\]

For a horizon-dependent constant step satisfying (1), summing (9) yields

\[
\begin{aligned}
\frac1K\sum_{k=0}^{K-1}\mathbb E\|\nabla f(x^k)\|^2
\le{}&\frac{2(\mathcal V_0-f_{\inf})}{\alpha p_{\min}K}\\
&+\frac{1}{p_{\min}}
\sum_i p_i\left(L_{ii}\alpha+D^2w_i\alpha^2\right)\sigma_i^2. \tag{10}
\end{aligned}
\]

Thus `alpha` of order `K^(-1/2)` gives the usual `K^(-1/2)` stationarity
scaling, subject to the stability ceiling.  The implementation exactly
enumerates centered Rademacher innovations and verifies (9) on coupled random
quadratics.  Equation (10) still excludes Markov bias, critic error and
cross-packet dependence; it must not be cited as the MARL result.

## Verification record

On 2026-09-01, the theorem-facing package passed 15 tests in 0.20 seconds in
the `ust2` environment.  The complete `experiments/` regression then passed
939 tests with 7 skips in 124.78 seconds using the repository `.venv`, whose
NumPy and CVXPY versions were 2.2.2 and 1.6.5.  Running bare `pytest` from the
repository root is not the registered regression command: it also collects
historical repository snapshots below `tmp/`, and the `ust2` environment alone
does not contain CVXPY.  No historical snapshot or environment was changed.
