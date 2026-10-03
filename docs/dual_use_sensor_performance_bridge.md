# From dual-use sensing errors to last-iterate stability

## Scope

This note closes one exact interface for the Lyapunov-clocked optimism
program.  It is not yet the Markov policy-gradient theorem.  It applies when
the arriving context has two positive, predictable one-step energy
multipliers (q_{k,0},q_{k,1}) that are valid for the action actually taken.
The zero-delay randomly clocked linear game has this property in the
clock-balanced metric.  A nonlinear or delayed game needs its own valid
multiplier construction before using the result.

## Anytime last-iterate certificate

Let (V_k\geq0) be the state energy and let the causal action (u_k\in\{0,1\})
satisfy

\[
\mathbb E[V_{k+1}\mid\mathcal F_k]
\leq q_{k,u_k}V_k,
\qquad q_{k,u_k}>0,
\]

where (q_{k,u_k}) is (mathcal F_k)-measurable.  Then

\[
M_k=\frac{V_k}{V_0\prod_{j<k}q_{j,u_j}}
\]

is a nonnegative supermartingale.  Ville's inequality therefore implies that,
with probability at least (1-\delta), simultaneously for all (k),

\[
\log V_k
\leq \log V_0+\sum_{j<k}\log q_{j,u_j}+\log(1/\delta).
\]

This is the missing logical step between a cumulative log-drift objective and
a last-iterate stability statement.  It does not require iid contexts or an
independence assumption between the controller and its past.  It does require
the one-step conditional multiplier itself to be valid and predictable.

## Exact price of sensor-induced action errors

Let (u_k^\star) be any counterfactual schedule evaluated on the same
exogenous multiplier table and define

\[
L(u)=\sum_{k<K}\log q_{k,u_k}.
\]

For the binary action pair,

\[
L(u)-L(u^\star)
\leq
\sum_{k:u_k\ne u_k^\star}
\left|\log q_{k,0}-\log q_{k,1}\right|.
\]

Consequently, if the exact-geometry comparator has
(L(u^\star)\leq-\kappa K) and the weighted sensor disagreement is at most
((\kappa-\kappa')K), the causal sensor schedule retains certified rate
(kappa'>0).  Combined with the anytime bound,

\[
V_K\leq V_0\exp\{-\kappa'K\}/\delta
\]

on the same event.  Thus a sensing theorem does not need to reproduce the
oracle action at every event; it needs to control the *gain-weighted* action
mistakes.  Errors near the phase boundary are automatically cheap, whereas
errors in strongly rotational regions are expensive.

## Why this does not finish the paper

The current LCO-S0 likelihood is a development model for a nonlinear
fingerprint of two perturbed gradients.  It has no time-uniform Markov-noise
coverage theorem.  The next proof must upper-bound its gain-weighted action
mistakes from fully charged observations.  In addition, additive stochastic
gradient/critic error introduces an affine term in the drift recursion and
requires a separate martingale/noise-floor argument.  Neither issue is hidden
inside the supermartingale result above.

The executable checks are in
`experiments/clocked_async_mpg/sensor_performance_bridge.py`.  Exhaustive tests
over every pair of binary length-four schedules verify the deterministic
mismatch inequality.
