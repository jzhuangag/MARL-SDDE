# Freshness is a resource: Lyapunov-scheduled sensing for asynchronous MARL

## Unified research question

Consider centralized training with decentralized execution in a cooperative or
Markov-potential game.  Agent `i` owns policy block `theta_i`.  Its rollout and
gradient packet is computed asynchronously while other agents update their own
blocks.  Single-flight ownership makes the packet **self-fresh** in
`theta_i`, but it becomes **strategically stale** because the teammate policy
profile has changed.

The central question is not merely how much to shrink a delayed gradient:

> Under a limited actor-transition and wall-clock budget, when should the
> learner spend additional on-policy interaction to re-measure the current
> strategic gradient, and how should it combine that measurement with a
> completed self-fresh/cross-stale packet?

This is a sensing--estimation--learning problem.  Freshness is costly in RL
because it requires new environment interaction; it is not a free timestamp.
The proposed main line is therefore **Lyapunov-Scheduled Freshness Fusion
(LSFF)**, not another staleness-dependent learning-rate rule.

## Why the previous controller failed

The fresh-seed Layer-0 pilot rejected scalar sample-split filtering.  Its second
trajectory was independent but sampled at packet birth, so it did not observe
the arrival policy.  The controller could only multiply the stale proposal by
a scalar and discarded the validation half for learning.  It learned, and its
queue was active, but it lost about 9% mean return to the strong full-data raw
baseline and worsened the heterogeneous lower tail.

LSFF changes the controlled variable.  It does not choose a smaller stale
step.  It decides whether to buy a current-policy gradient measurement; when it
does, it fuses both estimates and changes the update direction.

## Conditional model

At completion event `k`, let

- `g_k = partial_i Phi(theta_k)` be the current partial gradient of the
  potential;
- `b_k` be the packet-birth gradient estimator;
- `f_k` be an independent arrival-fresh estimator, available only after paying
  refresh cost `c_k`;
- `Delta_k` bound strategic gradient rotation between packet birth and arrival;
- `sigma_b,k^2` and `sigma_f,k^2` bound the trace variances of the two
  estimators.

Conditionally on the event history, assume

```
b_k = g_k + h_k + xi_b,k,     ||h_k|| <= Delta_k,
f_k = g_k + xi_f,k,
E[xi_b,k | F_k] = E[xi_f,k | F_k] = 0,
E||xi_b,k||^2 <= sigma_b,k^2,
E||xi_f,k||^2 <= sigma_f,k^2,
E[xi_b,k^T xi_f,k | F_k] = 0.
```

For finite-horizon on-policy Monte Carlo policy gradients, the fresh estimator
is conditionally unbiased when the arrival rollout is generated from the
current joint policy and its baseline is action-independent and frozen within
the rollout.  Markov/mixing bias must be added to `Delta_k` when a different
estimator is used; it cannot be silently treated as variance.

If the partial gradient is `L_cross`-Lipschitz in teammate policy
total variation, an observable bound is

```
Delta_k <= L_cross * sum_{j != i} sqrt(KL(pi_j^birth || pi_j^current) / 2).
```

The mean-KL implementation is valid only for the reference-state distribution
appearing in the corresponding gradient-Lipschitz assumption.  A uniform-state
claim would require a uniform KL certificate.

## Theorem 1: MSE-optimal freshness fusion

After refreshing, use

```
g_hat_k(w) = (1-w) b_k + w f_k,      0 <= w <= 1.
```

Define

```
A_k = sigma_b,k^2 + Delta_k^2,
B_k = sigma_f,k^2.
```

Then

```
E[||g_hat_k(w)-g_k||^2 | F_k]
    <= (1-w)^2 A_k + w^2 B_k = U_k(w).
```

The unique minimizer when `A_k+B_k>0` is

```
w_k^* = A_k / (A_k+B_k),
U_k^refresh = A_k B_k / (A_k+B_k).
```

With no refresh, `U_k^stale=A_k`.  The exact value of buying freshness is

```
R_k = U_k^stale-U_k^refresh = A_k^2/(A_k+B_k).
```

Thus the fusion automatically becomes an equal average when both estimators
are unbiased and equally noisy, and moves continuously toward the arrival
gradient as cross-policy drift grows.  No candidate scan, generic QP, Hessian,
or covariance matrix is required; the arithmetic is O(d) and the extra state
is scalar per agent.

**Proof.**  Expand the conditional squared error.  The centered cross term is
zero by conditional independence, and the squared conditional bias is at most
`(1-w)^2 Delta_k^2`.  Differentiating the resulting convex quadratic gives the
stated minimizer.  Substitution gives `U_k^refresh` and `R_k`.

## Lyapunov control of estimation-risk debt

Let `u_k` be one if a fresh rollout is purchased.  The incurred MSE upper bound
is

```
U_k(u_k) = A_k-u_k R_k.
```

For a declared long-run MSE budget `U_bar`, introduce

```
Q_{k+1} = [Q_k + U_k(u_k)-U_bar]_+,
L(Q_k) = Q_k^2/2.
```

The one-step drift satisfies

```
Delta L_k <= C + Q_k (U_k(u_k)-U_bar),
```

where `C` is half a uniform square-increment bound.  Minimizing this drift plus
`V c_k u_k` gives the exact threshold

```
u_k = 1  iff  Q_k R_k > V c_k.
```

This is the central Lyapunov action.  High strategic drift increases `A_k` and
therefore the value `R_k`; accumulated estimation-risk debt increases the
shadow price of not refreshing; a high environment/wall-clock cost suppresses
refresh.  The algorithm chooses a sensing action and an estimator, not an
agent count or a learning rate.

For every sample path, the queue recursion gives the deterministic certificate

```
sum_{k<K} U_k(u_k) <= K U_bar + Q_K.
```

Therefore mean-rate stability, `E[Q_K]/K -> 0`, implies the declared average MSE
budget.  This feasibility statement does not require i.i.d. clock events.  A
cost-optimality gap against a stationary randomized refresh policy additionally
requires a stated exogeneity/ergodicity condition and a Slater action; it is
not yet claimed for fully endogenous MARL trajectories.

## Budget-facing dual formulation

The research question is more naturally stated as minimizing estimation risk
under actor-transition and wall-clock refresh budgets.  Let resource `r` have
per-refresh cost `c_r,k` and allowed average `c_bar_r`.  Define

```
Z_r,k+1 = [Z_r,k + c_r,k u_k-c_bar_r]_+.
```

Minimizing multi-resource drift plus `V U_k(u_k)` gives

```
u_k = 1  iff  V R_k > sum_r Z_r,k c_r,k.
```

Thus `V` selects a point on the risk--resource frontier, while each queue is an
online shadow price for a physical budget.  A finite-horizon remaining-budget
check can veto the action without invalidating the resource certificate.  For
every realized sequence,

```
sum_{k<K} c_r,k u_k <= K c_bar_r + Z_r,K.
```

This is the primary implementation interface because it represents the two
resources in the problem statement directly.  The risk-debt formulation is
the dual orientation of the same frontier: minimize refresh cost subject to a
declared MSE budget.  They are not two unrelated controllers.

## Theorem 2: from sensing risk to potential convergence

Suppose `Phi` is upper bounded and `L`-smooth and the owner applies

```
theta_{i,k+1} = theta_{i,k} + eta g_hat_k,
```

with `eta <= 1/(4L)`.  Conditional smoothness and Young's inequality yield

```
E[Phi(theta_{k+1})-Phi(theta_k) | F_k]
 >= (eta/2-L eta^2)||g_k||^2
    -(eta/2+L eta^2) U_k(u_k).
```

Consequently,

```
(1/K) sum_{k<K} E||g_k||^2
 <= 4(Phi_max-Phi_0)/(eta K)
    + 3 U_bar + 3 E[Q_K]/K.
```

**Proof.**  Write `g_hat_k=g_k+e_k`.  Use
`g_k^T g_hat_k >= ||g_k||^2/2-||e_k||^2/2` and
`||g_hat_k||^2 <= 2||g_k||^2+2||e_k||^2` in the smoothness
lower bound, take conditional expectation, sum, and apply the sample-path queue
certificate.

This theorem makes the roles consistent: Lyapunov drift controls an estimator
risk that directly appears in the finite-time potential-gradient bound.  Under
an additional gradient-domination or policy-gradient-to-Nash-gap condition for
the Markov potential game, the stationarity bound converts to an approximate
Nash guarantee.  That conversion must be stated separately for the selected
policy class.

## Wall-clock and Markov extensions

The existing single-flight clock theorem supplies arbitrary-order service
accounting and a wall-clock completion bound.  LSFF adds the service time and
actor transitions of every purchased refresh.  The final rate must be stated
in total charged actor transitions and elapsed service time, not only update
count.

The theorem above is finite-horizon/on-policy and does not need an SDDE.  A
hybrid SDDE can approximate the joint evolution of policy error, cross-policy
path length, and the reset-like refresh process, but it is an interpretation
layer rather than the correctness argument.  The actual proof uses discrete
conditional Lyapunov drift because refreshes are event-driven jumps.

## Literature boundary

- [DC-ASGD](https://proceedings.mlr.press/v70/zheng17b.html) compensates a
  shared-model delayed gradient through a Taylor/Hessian approximation.
- [ASAP.SGD](https://proceedings.mlr.press/v162/backstrom22a.html) develops
  staleness-adaptive step-size rules.
- [adaptive asynchronous mini-batching](https://proceedings.mlr.press/v267/attia25a.html)
  adapts to delay quantiles by batching/filtering stale gradients.
- [Zeno++](https://proceedings.mlr.press/v119/xie20c.html) validates fully
  asynchronous shared-model gradients using a lazily refreshed held-out
  validation gradient and a hard accept/reject score.
- [SAT](https://arxiv.org/abs/2607.18722) contracts PPO/GSPO trust regions as a
  function of observed single-policy rollout staleness.

LSFF cannot claim novelty for validation, staleness filtering, adaptive
stepsizes, trust regions, or delay compensation individually.  Its potentially
new object is the **budgeted on-policy freshness-sensing problem induced by
distinct strategic policy blocks**: self-fresh/cross-stale gradient bias,
closed-form MSE fusion, and a Lyapunov refresh schedule whose controlled risk
enters a potential/Nash finite-time bound.  This novelty survives only if the
remaining theory and standard MARL experiments validate the complete chain.

## Two go/no-go gates before GPU

1. **Performance-bound gate.**  Close the conditional Markov estimator bounds,
   cross-policy bias certificate, queue stability/cost statement, and
   wall-clock composition without treating endogenous states as exogenous.
2. **CPU oracle-headroom gate.**  Under equal actor-transition charges, show a
   regime in which an oracle refresh schedule strictly improves the
   return--probe-cost Pareto frontier over never, always, and fixed-period
   refresh.  Then show the observable LSFF schedule captures a material
   fraction of that headroom on fresh development seeds.

Failure of either gate stops the standard GPU benchmark.  Passing both permits
a separately preregistered MPE/MAMuJoCo/SMACv2 sequence; it does not make the
CPU pilot formal paper evidence.
