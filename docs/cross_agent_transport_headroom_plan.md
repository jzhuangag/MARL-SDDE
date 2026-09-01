# Cross-agent transport outcome-free headroom plan

## Material Passport

- Artifact ID: `CAGT-HEADROOM-STATIC-20260901`
- Artifact type: outcome-free analytic feasibility scan
- Origin workflow: ARS experiment-agent plan
- Parent commit: `3c9d142`
- Scientific population outcomes: none
- Formal or GPU authorization: none

## Question

Before proving a wall-clock separation or designing a stochastic pilot, determine whether exact cross-agent gradient transport has at least 10% finite-horizon wall-clock value over a strong nontransport comparator envelope after charging one Hessian-vector-product service cost per transported completion.

This is a deterministic design calculation on known quadratic potentials.
It is not efficacy evidence and will never enter a paper result table.

## Frozen analytic grid

The grid crosses:

- agent count `n in {3,5,8}`;
- equicoupling `c in {0.2,0.5,0.8}`;
- deterministic worker patterns `one_straggler` and `two_tier`;
- slow latency `M in {4,12}`;
- sparse, alternating and ramp initial conditions;
- fully charged Hessian-vector overhead `h in {0,0.25,1}`.

There are 324 cells: 54 low, 216 transition and 54 high.
The wall-clock horizon is 96.
Exact transport uses one coordinate-smoothness step and pays `h` at every processed completion.

## Strong comparator envelope

Each cell is compared to the outcome-aware minimum-regret member of 54 nontransport policies:

- seven raw asynchronous fixed steps;
- seven fresh serial fixed steps;
- seven synchronous barrier fixed steps;
- nine age-decay rules;
- twelve wall-clock delay-adaptive steps;
- twelve observable cross-debt-ratio gates.

The comparator envelope is deliberately favorable to the baseline.

## Frozen gates

All gates are mandatory for continuing the theorem program.

1. All 17,820 analytic trajectories are finite and complete.
2. High-phase geometric transport/envelope regret ratio is at most 0.90.
3. At least 60% of high-phase cells have at least 10% transport gain.
4. Transition geometric ratio is at most 0.95.
5. Low-phase geometric ratio is at most 1.05.
6. In the high phase with full HVP overhead `h=1`, geometric ratio is at most 0.95.
7. Exact transport takes no potential-decreasing coordinate step.

Failure of any gate stops the current full-transport performance route.
Passing all gates permits only completion of the wall-clock theorem and a separate stochastic-pilot preregistration.

## Commands

```powershell
.\.venv\Scripts\python.exe -m experiments.policy_update_backpressure.transport_headroom validate
.\.venv\Scripts\python.exe -m experiments.policy_update_backpressure.transport_headroom run --output docs/cross_agent_transport_headroom_results.json
```

The run command is forbidden until this plan, runner and tests are committed together.
