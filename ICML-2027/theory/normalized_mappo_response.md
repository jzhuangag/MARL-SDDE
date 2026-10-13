# Observable neural response with normalization and optimizer state

This interface concerns a finite, independent trajectory-validation objective.
It is not an identity between critic tracking error, PPO surrogate, and true
team return. Participation q here counts rollout streams, not agents in a game.

## Exact lifted empirical-risk state

Let f_theta(s) be the centralized critic's normalized output, and let m and a>0
be the observed ValueNorm mean and standard deviation. On a frozen sensor
buffer with Monte Carlo labels y, define

V(theta,m,a) = mean_s [(a f_theta(s)+m-y)^2].

The learning plant also includes Adam's first/second moments and update index.
Those states determine the actual parameter displacement. Even when the
potential does not explicitly depend on the moments, omitting them from a
predictive transition model is invalid.

For an actual candidate displacement d in z=(theta,m,a), the observable
quadratic forecast is g(z)^T d + 0.5 d^T H(z) d. If H is M-Lipschitz along
the segment z+td, the exact Taylor integral gives

|V(z+d)-V(z)-g(z)^T d-0.5 d^T H(z)d| <= M ||d||^3/6.

This follows by integrating the Hessian difference against (1-t) and applying
its operator-norm Lipschitz bound. With tanh, positive LayerNorm epsilon, a
compact parameter region, and finite labels, such an M exists. The current
implementation measures the remainder; it does not produce a usable global
numerical M or turn its measurements into a confidence certificate.

## Why normalization is part of the decision interface

Write d=(d_theta,d_m,d_a). Ignoring the last two coordinates drops even the
first-order term V_m d_m+V_a d_a. ValueNorm can move substantially on an early
update even when theta barely changes. A parameter-only forecast therefore
has no general third-order remainder bound for the actual denormalized risk.
The experiment compares both forecasts without modifying HARL's normalizer.

## Actor and transfer boundaries

The actor diagnostic is the smooth, unclipped empirical PPO surrogate plus
entropy on a fixed buffer. Actual training still uses the pinned clipped MAPPO
update. The diagnostic is not claimed to equal the clipped objective outside
its inactive clipping region, or to equal a policy-return improvement.

At a later training stage let P_new and P_old denote current and cached sensor
potentials. For one action z->z+d, the transfer error decomposes exactly into
the old-potential Taylor remainder and

[P_new(z+d)-P_old(z+d)] - [P_new(z)-P_old(z)].

The latter is objective/distribution motion. Charging a sensor or writing a
Lyapunov expansion does not bound it. Independent current-stage truth episodes
measure the held-out error and expose whether cached response rankings transfer.
No general Markov/policy-motion certificate is claimed without an additional
bound on this term.

## Qualification versus an executable controller

The response study deliberately pays for shadow candidate updates to observe
their displacements. Gradient/Hessian forecasts use only sensor trajectories;
separate truth trajectories supply diagnostics. All rollout, optimizer-row,
and higher-order differentiation work is recorded. The resulting experiment
qualifies a response model; it is not yet a low-overhead deployed controller
or a budget-fair return-superiority experiment.
