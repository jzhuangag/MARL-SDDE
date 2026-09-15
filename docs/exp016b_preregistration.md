# EXP-016B preregistration: Premature Adaptation under Finite Learning Horizons

This is a prospective, outcome-free preregistration following the independent
T-018 erratum commit `b893bb54fef78168774a6c5607e4d7e43e8db2ad`. It authorizes no trajectory in this
commit and does not revive EXP-016A.

## Question and scope

When reliable identification is feasible but identification plus delay cost
is not yet amortized by remaining learning benefit, does learning-aware
fallback outperform information-only probing? The primary contrast is inside
the finite learning-value zone, not after both policies converge above
`B_value`.

## Frozen design

- Configuration SHA-256: `6dfdf87521700c2ddae9b81947e0ecc01ee33ebcf5fcda34b09e9e3c3f7f7ee5`
- Corrected T-018 grid: `c5d2dd5ddac7540888d708ab59d4e3954994da018951797a79d05200ef0ee2db`
- Finite scenarios: 96 for Layer A; 48 for Layer B
- Search-censored descriptive scenarios: 8, excluded from finite-threshold gates
- Policies: 8
- Fresh paired pilot seeds: 96
- Layers: Gaussian common-factor learner and affine delayed Markov TD transfer

Layer A must use actual common-factor trajectories, individual observations,
and the actual downstream learner; analytic risk is not an observed outcome.
Layer B must use actual TD updates and delay queues, complete dual-budget
charging, and a stability-screened catalogue, with no actor-critic,
preconditioner, or hidden rho/theta/regime input.

The information-only policy may use only the public model pair, mixing
certificate, selected `(q,b)`, overhead, and budget ray. It may not access:
downstream_risk, wrong_commit_loss, epsilon_safe, oracle_action, hidden_theta, hidden_regime, outcome_data.

## Static execution decision

Estimated single-process CPU is `5.263`
hours, peak memory `8.0` GB, and disk
`1.101` GB. Local CPU is recommended for a
future pilot after this commit. HPC4, `/project`, and GPU remain unauthorized.
