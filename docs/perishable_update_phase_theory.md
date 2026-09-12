# When policy updates expire: phase and Lyapunov derivation

Date: 2026-09-01.
Status: proved deterministic/conditional lemmas plus an open finite-time
programme.  This document does not claim the main theorem is complete.

## One paper-level question

In asynchronous heterogeneous-agent CTDE, when is a completed unilateral
policy proposal still useful, and how should a learner schedule valid proposals
without waiting at the slowest-agent barrier?

The answer is organized around one object: **cross-agent freshness debt**.
Applying agent `i`'s update changes the current gradient of every pending agent
`j`; that loss of validity is an endogenous externality of learning the same
joint policy.  It is absent when multiple workers estimate one shared-policy
gradient and absent when agents merely execute macro-actions at different
times.

## Assumptions and event state

Let `Phi(theta)` be a differentiable cooperative Markov potential to maximize,
with parameter blocks `theta=(theta_1,...,theta_n)`.  Assume block smoothness

```text
Phi(theta + e_i h) - Phi(theta)
  >= <grad_i Phi(theta), h> - (L_i/2)||h||^2,
```

and cross-gradient sensitivity

```text
||grad_j Phi(theta + e_i h) - grad_j Phi(theta)||
  <= C_(j,i) ||h||.
```

Agent `i`'s completed proposal `g_hat_i` was computed from a joint-policy
snapshot at its birth event.  Let `Z_i` be the accumulated cross-policy bias
bound since that birth event and `r_i` a Markov-sampling/confidence radius.  On
the declared coverage event,

```text
||g_hat_i - grad_i Phi(theta_current)|| <= B_i := Z_i + r_i.
```

This is a theorem-facing interface.  A neural implementation must separately
justify how it estimates `C_(j,i)`, `Z_i` and `r_i`; a PPO sampled ratio is not
silently treated as a uniform policy-distance certificate.

## Lemma 1: certified gain and the freshness phase

For update `theta_i^+ = theta_i + alpha g_hat_i`, block smoothness and
Cauchy-Schwarz give

```text
Phi(theta^+) - Phi(theta)
  >= alpha (||g_hat_i||^2 - B_i ||g_hat_i||)
     - (L_i/2) alpha^2 ||g_hat_i||^2.                 (1)
```

Proof: write the inner product in the block-smoothness inequality as

```text
<grad_i Phi, g_hat_i>
 = ||g_hat_i||^2 + <grad_i Phi-g_hat_i, g_hat_i>
 >= ||g_hat_i||^2-B_i||g_hat_i||.
```

Define the freshness ratio

```text
rho_(i,k) = B_i / ||g_hat_i||,
```

with `rho=+infinity` for a zero proposal.

- `rho<1`: a nonzero step has a positive certified interval.  Ignoring other
  pending agents, the gain-maximizing step is

  ```text
  alpha_gain = clip((||g_hat_i||-B_i)/(L_i||g_hat_i||), 0, alpha_bar).
  ```

- `rho>=1`: no update along `g_hat_i` can have a uniformly positive first-order
  certificate from this information.  The uncertainty ball contains the
  gradient `g_hat_i-B_i g_hat_i/||g_hat_i||`, whose inner product with the
  proposal is nonpositive.  A uniformly safe causal rule must refresh, collect
  more information, or make no claim.

This is the phase boundary.  It is relative signal-to-cross-debt, not timestamp
age alone.

### Proposition 1: low-load static near-optimality

The phase claim must also explain negative results.  Suppose throughout a
declared interval that `B_i/s_i <= epsilon < 1`, all block smoothness constants
equal the known bound `L`, and the cap does not bind.  The *fixed* step

```text
alpha_static = (1-epsilon)/L
```

has certified gain at least

```text
(1-epsilon)^2 s_i^2/(2L).
```

Even a fresh proposal cannot obtain more than `s_i^2/(2L)` from the quadratic
smoothness certificate.  Consequently the maximum certificate-level headroom
left to any freshness-adaptive rule is at most

```text
1-(1-epsilon)^2 = 2 epsilon-epsilon^2.                 (1a)
```

This is not a universal return-ratio theorem: it is a sharp statement about
the same one-event certificate used by the controller.  It predicts that weak
cross coupling or short proposal lifetimes leave little dynamic value, so a
strong fixed rule should tie the adaptive scheduler rather than be treated as
a failed experiment.

## Lemma 2: accepted updates make other proposals expire

If agent `i` applies `alpha g_hat_i`, then every still-pending proposal `j`
incurs at most

```text
Z_j^+ <= Z_j + C_(j,i) alpha ||g_hat_i||.             (2)
```

The processed agent immediately starts its next computation from the new
joint-policy snapshot, so its new debt is `Z_i^+=0`.  Equation (2) follows by
the cross-gradient sensitivity inequality and the triangle inequality.  It
shows why optimizing each proposal independently is not valid: one accepted
step consumes the freshness margins of the other agents.

## The Lyapunov scheduler

Use the real debt Lyapunov function

```text
H_k = (1/2) sum_j Z_(j,k)^2 - V Phi(theta_k),          (3)
```

where `V>0` trades immediate potential progress against preservation of pending
work.  This is not an external safety queue.  Its state is the certified bias
of proposals that actually exist in the asynchronous learner.

At completion of agent `i`, combine (1)-(2).  For `s_i=||g_hat_i||`, an upper
bound on the one-event drift is

```text
Delta H_i(alpha) <= -Z_i^2/2
  + alpha s_i sum_(j != i) C_(j,i) Z_j
  + alpha^2 s_i^2/2 sum_(j != i) C_(j,i)^2
  - V alpha (s_i^2-B_i s_i)
  + V L_i alpha^2 s_i^2/2.                            (4)
```

The exact minimizer of the action-dependent quadratic upper bound is

```text
numerator_i = V(s_i^2-B_i s_i)
              - s_i sum_(j != i) C_(j,i) Z_j,

denominator_i = s_i^2 (V L_i + sum_(j != i) C_(j,i)^2),

alpha_i^* = clip(numerator_i / denominator_i, 0, alpha_bar).  (5)
```

For `s_i=0`, set `alpha_i^*=0`.  If multiple proposals complete at the same
event, evaluate (4) for each and process the smallest bound first.  Updating all
debts after a selected action is `O(n)` once proposal norms and cross-sensitivity
bounds are available.  There is no Hessian inverse, covariance matrix,
preconditioner or finite catalogue of learning rates.

Equation (5) simultaneously determines online step size, admission and update
order.  The two forces are not separate heuristics:

- the `V(s_i^2-B_i s_i)` term rewards certified learning progress;
- the `sum C_(j,i)Z_j` and `sum C_(j,i)^2` terms price the damage to unfinished
  agents' proposals.

## Conditional one-event theorem

On the coverage event and under the smoothness assumptions:

1. (4) upper-bounds the actual drift of (3);
2. (5) minimizes that quadratic upper bound over `[0,alpha_bar]`;
3. any accepted step has `B_i<s_i` and a nonnegative certified potential gain;
4. because zero is feasible, the selected bound is at most `-Z_i^2/2`, the
   drift obtained by discarding the completed proposal and refreshing it.

The first two claims follow from Lemmas 1-2 and scalar quadratic minimization.
For claim 3, a positive numerator implies
`V(s_i^2-B_i s_i)>0`; (5) is no larger than the gain-only optimizer, so the
right side of (1) is nonnegative.  Claim 4 follows by comparison with
`alpha=0`.

If `Phi` is upper bounded, telescoping (4) gives a finite cumulative bound on
the squared debt observed at completion events.  This is a valid stability
statement for the theorem-facing deterministic/coverage-event system.  It is
not by itself a stationarity theorem.  The following result makes that
implication precise.

## Theorem 1: finite-time event and full-gradient stationarity

This theorem uses one drift weight `V`; it is not an offline catalogue of
agent-specific learning rates.  Define

```text
Kappa_i(V) = V L_i + sum_(j != i) C_(j,i)^2,

R_k = [s_k - Z_(I_k,k) - r_k
             - (1/V) sum_(j != I_k) C_(j,I_k) Z_(j,k)]_+ .       (6)
```

Assume:

1. `Phi(theta) <= Phi_star`, proposal norms and block gradients are at most
   `G`, and the smoothness/cross-sensitivity inequalities hold;
2. the confidence event in the first section holds simultaneously for the
   first `K` completions;
3. every agent completes at least once in every `D` consecutive completion
   events;
4. `0<L_min<=L_i`, `C_(j,i)<=Gamma` (with `C_(i,i)=L_i`) and the
   public cap satisfies

   ```text
   alpha_bar >= max_i V/Kappa_i(V).
   ```

Let

```text
A_V = (1/2) sum_i Z_(i,0)^2 + V(Phi_star-Phi(theta_0)).
```

Then the executable update (5) satisfies

```text
sum_k Z_(I_k,k)^2 <= 2 A_V,

sum_k R_k^2 <= Q_R := 2 Kappa_max(V) A_V/V^2,         (7)
```

where `Kappa_max(V)=max_i Kappa_i(V)`.  If
`delta_k=alpha_k s_k` is the accepted policy-path length, then

```text
sum_k delta_k^2 <= Q_delta := Q_R/L_min^2.            (8)
```

Moreover, with `rbar_K^2=K^(-1) sum_k r_k^2`, the block gradient seen at
completion events obeys

```text
(1/K) sum_k ||grad_(I_k) Phi(theta_k)||^2
 <= [4 Q_R + 32 A_V
     + 4 S_max n Gamma^2 D^2 Q_delta/V^2]/K
    + 16 rbar_K^2,                                     (9)
```

where `S_max=max_i sum_(j!=i) C_(j,i)^2`.

The full joint gradient at every event obeys the explicit bound

```text
(1/K) sum_k ||grad Phi(theta_k)||^2
 <= (2D/K)[4 Q_R + 32 A_V
            + 4 S_max n Gamma^2 D^2 Q_delta/V^2]
    + 32 D rbar_K^2
    + 2 n Gamma^2 D^2 Q_delta/K
    + 2 n D G^2/K.                                    (10)
```

For fixed positive `V`, zero initial debt, bounded `D` and exact gradients,
(10) is `O((D+nD^2)/K)`.  Choosing a horizon-dependent `V` changes constants
and the aggressiveness/freshness tradeoff but is not required for convergence.
With stochastic proposals the result converges to the declared
confidence-radius floor.  This is finite-time stationarity, not global
optimality of a non-concave neural policy.

### Proof

The action-dependent part of (4) is

```text
-V s_k R_k alpha + (s_k^2 Kappa_(I_k)(V)/2) alpha^2.
```

The residual satisfies `R_k<=s_k`, so the unconstrained minimizer is at most
`V/Kappa_(I_k)(V)` and the public cap does not bind.  Its improvement over zero
is exactly

```text
V^2 R_k^2 / (2 Kappa_(I_k)(V)).
```

Hence

```text
H_(k+1)-H_k
 <= -Z_(I_k,k)^2/2
    -V^2 R_k^2/[2 Kappa_(I_k)(V)].
```

Telescoping and `H_K>=-V Phi_star` proves (7).  The executed displacement is

```text
delta_k = V R_k/Kappa_(I_k)(V) <= R_k/L_min,
```

which proves (8).

From (6), in both the positive- and zero-residual cases,

```text
s_k <= R_k + Z_(I_k,k) + r_k
       + (1/V) sum_(j != I_k) C_(j,I_k)Z_(j,k).
```

The coverage event therefore implies

```text
||grad_(I_k) Phi(theta_k)||
 <= R_k + 2Z_(I_k,k) + 2r_k
    + (1/V) sum_(j != I_k) C_(j,I_k)Z_(j,k).          (11)
```

Because a proposal is reset at its completion, condition 3 gives

```text
Z_(j,k)^2
 <= Gamma^2 D sum_(m=k-D)^(k-1) delta_m^2.            (12)
```

Summing (12) over events and pending agents yields
`sum_(k,j) Z_(j,k)^2<=n Gamma^2D^2Q_delta`.  Therefore Cauchy--Schwarz gives

```text
sum_k [sum_(j!=I_k) C_(j,I_k)Z_(j,k)]^2
 <= S_max n Gamma^2 D^2 Q_delta.
```

Squaring (11), using `(a+b+c+d)^2<=4(a^2+b^2+c^2+d^2)`, and applying these
sums proves (9).

For any event `k` and block `i`, let `tau_i(k)` be its most recent completion.
The fairness condition makes `k-tau_i(k)<=D`.  Cross sensitivity and
Cauchy--Schwarz give

```text
||grad_i Phi(theta_k)-grad_i Phi(theta_(tau_i(k)))||^2
 <= Gamma^2 D sum_(m=k-D)^(k-1) delta_m^2.
```

Each completed-block term can be reused by at most `D` later events.  Applying
`(a+b)^2<=2a^2+2b^2`, summing over blocks and charging the at-most-`D` initial
events by `G^2` proves (10).  Every term in the result corresponds to an
observable proposal event, a declared confidence radius or a stated model
constant.

### Random completion and wall-clock corollary

The bounded-gap condition is deterministic.  For a random completion process,
apply the theorem on the event `D_K<=d`, where `D_K` is the largest per-agent
completion gap among the first `K` events.  If the completion process has a
verified geometric tail, a union bound gives `D_K=O(tau_c log(nK/delta))`
with probability at least `1-delta`.  Likewise, if the event counter satisfies
`Pr{K(T_wall)>=nu T_wall}>=1-delta_clock`, substitute
`K=floor(nu T_wall)` in (10).  These are explicit clock/completion assumptions;
the proof does not turn an arbitrary unbounded straggler process into a bounded
one.

### Markov-policy-gradient radius

For episodic on-policy proposals, suppose a single-trajectory block-gradient
estimator is unbiased at the birth policy and every coordinate lies in
`[-g_i,g_i]`.  The average of `m_i` independently reset trajectories has the
simultaneous coordinate-union radius

```text
r_(i,k) = g_i sqrt(2 d_i log(2 d_i n K/delta)/m_i),    (14)
```

with total failure probability at most `delta`; (14) follows directly from
coordinate Hoeffding and a union bound.  The Markov dependence *within* a
trajectory is already part of the bounded trajectory estimator and is not
counted again as independent samples.  For continuing trajectories, (14)
must be replaced by a Markov-chain Bernstein radius with the verified pseudo
spectral gap or mixing-time factor.  Paulin's Bernstein inequalities provide
that interface; the theorem does not silently reuse overlapping blocks.

Approximate-critic bias, off-policy reuse and a learned cross-sensitivity
constant must each be added to `r_(i,k)`.  They are not covered by the word
"Markov" alone.

## Exact high-load witness

The two-coordinate concave potential

```text
Phi(x,y) = -(x^2+y^2)/2 - c*x*y,   0<c<1,
```

has cross sensitivity `C_(y,x)=c`.  With `c=0.9`, initial
`(x,y)=(-1.2,1)`, and step `alpha=0.5`, the stale slow-coordinate gradient is
`0.08`.  One fast-coordinate update changes it to `-0.055`.  The exact
cross-debt is `0.135`, so `rho=1.6875`; applying the stale slow proposal lowers
the potential by `0.003`.  This is not a benchmark result.  It proves that the
high-load region is nonempty and that timestamp freshness without cross-policy
sensitivity cannot certify direction.

### Proposition 2: finite-horizon wall-clock separation

A second exact instance separates the two standard asynchronous compromises
using the *same* drift rule (5).  Use the same potential and start from
`(x,y)=(1,0)`.  Agent `x` completes at wall-clock one; agent `y`'s proposal,
born at time zero, completes at wall-clock `M`.  No corrective proposal
completes between `M` and the horizon `2M`.  This deterministic completion
trace is a degenerate finite-state Markov completion process.

Set `V=sqrt(M)`, `Kappa=V+c^2`, and use the public nonbinding cap.  PUB's first
fast step is exactly `V/Kappa`, leaving

```text
x_1 = c^2/Kappa.
```

The slow proposal's resulting debt makes the next fast freshness residual
exactly zero, so the drift rule does not spend the pending slow proposal to
chase the remaining `O(1/sqrt(M))` fast-coordinate error.  When the slow
proposal completes, rule (5) either improves the potential or rejects it by
the one-event theorem.  Hence its integrated regret is bounded by

```text
Reg_PUB(2M)
 <= 1/2 + (2M-1)c^4/[2(sqrt(M)+c^2)^2] = O(1).        (13)
```

The barrier cannot move before `M`, so its regret is at least `M/2`.
For a like-for-like accept-all comparator, use the same fixed step
`V/Kappa` but ignore freshness debt.  Its stale slow update leaves a regret
that tends to `c^2/2`; because no correction completes before `2M`, its
integrated regret is `Omega(M)`.  Thus the separation is not created by giving
accept-all a larger learning rate.  The construction is intentionally
minimal: it proves the requested wall-clock separation, not broad empirical
superiority.  A practical experiment must still test the theory-defined phase
against strong signal-, age- and trust-region baselines.

The deterministic audit checks 144 exact stale-bias instances, 576 gain bounds,
144 closed-form optimizers and the sign-flip witness.  Numerical slack is at
floating-point roundoff and the maximum closed-form versus 2,001-point grid
gap is `1.203e-4`; see `perishable_update_phase_algebra.json`.

## What is still missing for the main theorem

The low-load and high-load phase propositions and the event-time stationarity
implication are now closed under their explicit assumptions.  The following
obligations remain open and block a new efficacy experiment:

1. instantiate a continuing-trajectory `r_i` and all approximate-critic terms
   for the eventual neural algorithm; the episodic bounded-estimator case is
   closed by (14);
2. replace known `C_(j,i)` with a conservative executable certificate or state
   it as a theorem-only constant and prove the practical estimator's coverage;
3. show that the executable proxy preserves nontrivial acceptance in the
   high-load identifiable region.

Until items 1-3 close, the work is a promising theorem interface, not an ICML
contribution.  No new CPU efficacy population or GPU benchmark is authorized.

## Literature boundary

Sequential HARL establishes the cooperative agent-wise performance foundation
([JMLR 2024](https://jmlr.org/papers/v25/23-0488.html)); AFedPG supplies a
different shared-policy federated delay model
([ICLR 2025](https://openreview.net/pdf?id=5DUekOKWcS)); finite-time
asynchronous stochastic approximation supplies a single-operator reference
([COLT 2020](https://proceedings.mlr.press/v125/qu20a.html)); and recent
staleness-adaptive trust regions prevent claiming generic mismatch-dependent
clipping as new ([arXiv:2607.18722](https://arxiv.org/abs/2607.18722)).  The
candidate novelty must reside in cross-agent proposal perishability, the phase
boundary and event-time scheduling of distinct policies in one Markov game.
