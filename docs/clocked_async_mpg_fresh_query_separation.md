# Fresh-query separation from fully utilized frozen-policy batching

Status: proved lower/separation witness for a fixed global barrier.  It does
not establish superiority over every partially asynchronous or speculative
off-policy algorithm.

## Two indistinguishable potential instances

Let

\[
f_a(x)=\frac a2x^2-x,
\qquad a\in\{a_-,a_+\},
\qquad 0<a_-<a_+.
\]

At `x_0=0`, both instances return the same deterministic gradient `-1`.
Therefore any number of rollout/gradient packets queried at this frozen point
contains exactly the same first-order information.  A synchronous shadow-batch
learner may use every such packet, but its output after the first adaptive
round must be the same `X` for both instances.

The best possible one-round worst-case gradient is

\[
\begin{aligned}
\inf_X\max_{a\in\{a_-,a_+\}}|aX-1|
&=\frac{a_+-a_-}{a_++a_-}\\
&=: \Delta_{\rm one}. \tag{1}
\end{aligned}
\]

The equality follows by balancing the two affine errors at
`X=2/(a_-+a_+)`.  It applies to any deterministic use of the full frozen
batch; identical deterministic packets cannot reveal the hidden curvature.
A randomized minimax statement follows by the corresponding two-point Yao
argument, but is not needed by the current deterministic witness.

## Same fixed step, two fresh asynchronous queries

Use the conservative fixed step `alpha=1/a_+`, which satisfies the
block-smooth descent ceiling for both instances.  After `m` fresh sequential
gradient queries and updates,

\[
|\nabla f_a(x_m)|=|1-a/a_+|^m.
\]

The worst instance is `a=a_-`.  After two queries,

\[
\Delta_{\rm async}=(1-a_-/a_+)^2
<\frac{a_+-a_-}{a_++a_-}=\Delta_{\rm one}, \tag{2}
\]

because `(1-r)^2 < (1-r)/(1+r)` for every `r=a_-/a_+` in `(0,1)`.
Hence every target

\[
\Delta_{\rm async}<\varepsilon<\Delta_{\rm one} \tag{3}
\]

requires at least two adaptive frozen-policy rounds, while the asynchronous
block obtains it with two fresh returns from that agent.

## Two-agent potential-game embedding

Embed the witness in the one-state continuous-action identical-interest game

\[
\Phi_a(x,y)=-f_a(x)-\frac{L_y}{2}(y-y^\star)^2.
\]

Agent 1 controls `x`; agent 2 controls `y`.  One safe fresh query/update solves
the known-curvature `y` block.  This is a smooth Markov potential game with a
degenerate one-step transition, so it is a valid lower-bound subclass rather
than evidence for rich Markov dynamics.

Let agent 1 and agent 2 have independent exponential packet rates
`lambda_f` and `lambda_s`.  The asynchronous time needed for two sequential
agent-1 queries and one agent-2 query is

\[
\begin{aligned}
\mathbb E T_{\rm async}
={}&\frac2{\lambda_f}+\frac1{\lambda_s}\\
&-\left[
\frac1{\lambda_f+\lambda_s}
+\frac{\lambda_f}{(\lambda_f+\lambda_s)^2}
\right]. \tag{4}
\end{aligned}
\]

The bracket is the expected minimum of an Erlang-2 and an exponential random
variable.  A fixed all-agent barrier needs two adaptive rounds, with expected
time

\[
\mathbb E T_{\rm barrier}
=2\left(
\frac1{\lambda_f}+\frac1{\lambda_s}
-\frac1{\lambda_f+\lambda_s}
\right). \tag{5}
\]

Subtracting (4) from (5) gives

\[
\frac1{\lambda_s}
-\frac{\lambda_s}{(\lambda_f+\lambda_s)^2}>0. \tag{6}
\]

Thus the accuracy interval and strict elapsed-time advantage are both
nonempty for every pair of positive service rates.  All completed frozen
packets are charged; their limitation is adaptive query depth, not discarded
compute.

## Comparator boundary

This witness defeats a strong **fixed global-barrier, on-policy** comparator
that uses all extra frozen-policy packets.  It does not defeat an algorithm
that releases a converged block from future barriers, applies partial updates
inside a round, or launches speculative rollouts at multiple future policies.
The first two are already forms of asynchronous/partial-barrier learning.  The
last changes the data law and must charge and analyze off-policy correction.

The witness therefore establishes that full utilization alone does not remove
the value of asynchronous fresh policy queries.  It does not prove broad
empirical superiority of the rate-balanced learner.

## Validation

Six deterministic tests verify the minimax formula against a dense direct
grid, the strict two-query accuracy interval for five curvature ratios, exact
geometric contraction, the Erlang/exponential formula against 500,000 fixed-
seed samples, strict time advantage for heterogeneous rates and a complete
numerical certificate at `a_-/a_+=1/2`.
