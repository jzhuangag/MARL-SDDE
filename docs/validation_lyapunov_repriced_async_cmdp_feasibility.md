# Arrival-repriced asynchronous primal--dual feasibility decision

## Decision

**Stop before the declared 432-cell run.**  Arrival-time dual repricing has a
clean exact identity and is operationally cheap, but a bounded development
scan found no robust queue/primal time scale that met both broad risk and broad
constraint-violation gates.  It is therefore insufficient as an ICML main
algorithm and is not escalated to sampled Markov learning or GPU work.

This is an explicitly outcome-guided development result, not a preregistered
experiment or statistical claim.  The scan was used for its stated purpose:
to avoid freezing and running a larger experiment when the mechanism was
already too narrow.

## Verified pieces

For a packet containing separate birth-policy reward and cost gradients,

```text
arrival_direction - birth_price_direction
  = -(lambda_arrival-lambda_birth) * cost_gradient
```

holds algebraically.  Two hundred randomized signed instances passed at
machine precision.  The exact constrained quadratic optimizer satisfies its
KKT equation and active constraint to `1e-10`.  Inactive-constraint controls
make the birth- and arrival-priced trajectories byte-for-byte equal.

The implementation is in
`experiments/clocked_async_mpg/repriced_async_primal_dual.py`; four targeted
tests pass.  The model is an exact constrained-potential abstraction, not an
RL benchmark.

## Bounded development scan

The scan used 32 active heterogeneous-delay cells for each hyperparameter
pair:

- construction seeds: `31013`, `31019`;
- agents: 4 and 6;
- cost coupling: 0.5 and 1.5;
- active budget fractions: 0.35 and 0.60;
- two-tier and skewed service profiles;
- primal steps: 0.10, 0.18, 0.30, 0.45;
- Lyapunov tradeoffs `V`: 1, 2, 4, 8, 12.

Across all 20 pairs:

- maximum fraction of cells with at least 10% composite-risk improvement:
  `0.46875`, below the required `0.60`;
- maximum fraction with at least 25% violation improvement: `0.625`, attained
  at step 0.10 and `V=4`, where the aggregate risk ratio was `0.9847` and no
  cell achieved 10% risk improvement;
- best aggregate risk ratio: `0.8855` at step 0.30 and `V=4`, but only `0.28125`
  of cells achieved the required risk improvement and only `0.4375` achieved
  the violation improvement;
- median asynchronous/barrier proposal ratio: `3.10` throughout.

The last number confirms generic throughput headroom under heterogeneous
services.  It does not establish value for repricing: the repricing-specific
birth comparison is the gate that fails.

## Research consequence

Do not run the unexecuted 432-cell configuration, introduce sampled seeds,
claim a constrained-MARL result, or select a favorable safety benchmark.  A
more elaborate step controller could be invented after these outcomes, but it
would return to the same pattern of repeatedly modifying a narrow online rule
after its intrinsic-value screen fails.

The reusable result is a diagnostic: dual-price staleness is a real and
exactly removable error term, but removing it alone is not a broad learning
advantage.  The remaining ICML program should not make asynchronous control
itself the headline unless the task supplies a structural objective that
cannot be matched by static or ordinary current-price training.
