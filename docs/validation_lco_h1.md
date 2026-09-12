# LCO-H1 frozen headroom validation

## Decision

LCO-H1 passes all ten preregistered gates.  This is a positive CPU
mechanism/headroom result for the exact-phase, current-oracle linear subclass.
It authorizes development of an observable Markov-geometry certificate and the
associated Lyapunov performance theorem.  It is not formal evidence and does
not authorize a standard MARL benchmark, GPU, HPC4, or formal seeds.

The result closes the narrow question that stopped several earlier programs:
the target problem has nontrivial dynamic value even against a strong
equal-budget fixed schedule.  It does not yet show that the phase can be
identified causally from Markov policy-gradient data.

## Frozen provenance

- preregistration commit:
  `5af97804a1444b88a7a30534aa4d3da7d4b94d07`;
- configuration:
  `docs/lco_headroom_config.json`;
- configuration SHA-256:
  `58caceeea755d8a1057073eeae0cca9284abc0f4f8e139c695c7d834eb54f6b8`;
- seeds: 81001--81032;
- 8,640 stochastic paths, 270 cells, and 8,847,360 asynchronous coordinate
  events;
- local CPU only, Python 3.11.13, NumPy 2.2.2, SciPy 1.15.1;
- primary command:
  `.\.venv\Scripts\python.exe -m experiments.clocked_async_mpg.run_lco_headroom run --config docs/lco_headroom_config.json --output-dir experiments/clocked_async_mpg/results/lco_h1_primary`;
- reproduction command: the same command with output directory
  `lco_h1_reproduction`.

The primary artifact was created at 2026-09-01 23:15:16 after approximately
41 minutes 32 seconds.  The isolated reproduction artifact was created at
2026-09-02 01:08:21 after approximately 111 minutes 49 seconds under local CPU
frequency/load variation.  Runtime was not a scientific metric.

Both `summary.json` files are 10,027,924 bytes and have the same SHA-256:

`0791d578c3c83ccce9c46d26120b6fd2e2ae1b68d522cee28ce2a9938e993ce0`.

## Frozen gate results

| Gate | Frozen requirement | Observed | Result |
|---|---:|---:|---:|
| L1 | finite values and exact accounting | 8,640/8,640 finite; no duplicate spec; allowance audit passed | pass |
| L2 | mean separated-dynamic log-rate gain at least 0.005 | 0.06561024 | pass |
| L3 | improved separated-dynamic cells at least 0.80 | 48/48 = 1.0 | pass |
| L4 | contracting separated-dynamic cells at least 0.80 | 48/48 = 1.0 | pass |
| L5 | median phase-oracle gain capture at least 0.60 | 0.78618228 | pass |
| L6 | potential-phase anchor fraction at most 0 | 0 | pass |
| L7 | potential controller/never absolute error at most 1e-12 | 0 | pass |
| L8 | each arrival-group gain at least 0.002 | 0.03588343 / 0.07227040 / 0.08867689 | pass |
| L9 | maximum budget overshoot at most 0 | 0 | pass |
| L10 | failed gate stops escalation | no failed gate; only CPU development authorized | pass |

The accounting audit independently confirmed 8,640 unique specifications, all
32 registered seeds, exact `floor(budget * horizon)` allowances, and zero
controller or phase-aware-comparator overshoot.  The controller purchased no
optimistic oracle on any potential event, including mixed-phase paths.

## Descriptive robustness and negative boundaries

The paired per-seed mean separated-dynamic gain was positive for 32/32 seeds.
Its mean was 0.06561024, standard deviation 0.00892349, and descriptive
t-based 95% interval `[0.06239298, 0.06882750]`.  These intervals are not
formal inference because LCO-H1 is a headroom pilot and the best fixed mask is
selected on the same pilot cells.

The target result spans all three arrival probabilities, both persistence
levels, rotation fractions 0.25/0.5/0.75, and budgets 0.25/0.5.  It is not
uniform over every registered regime:

- at normalized step 0.2 the subgroup mean gain is 0.00268984, below the
  aggregate L2 threshold, although 31/32 paired seed averages are positive;
- in the pure stationary rotational population, the controller is contractive
  in 77.78% of cells and is worse than the strong fixed schedule by 0.00296573
  on average; the descriptive paired interval is
  `[-0.00408830, -0.00184316]`;
- over all 270 cells, not just the separated-dynamic target, the controller
  improves on the best fixed schedule in 60.74% and ties in 20%;
- phase-oracle gain capture ranges from 0.2367 to 1.1103.  Values above one do
  not mean a clairvoyant optimum was beaten: the registered comparator is a
  phase-aware same-count schedule, not a pathwise state-and-arrival oracle.

Accordingly, LCO-H1 supports phase-adaptive resource allocation in switching
potential/rotational geometry.  It does not support universal no-harm or
universal dominance over fixed optimism.

## Scope limitations

The decision observes the true phase, uses a noiseless current-parameter
oracle, has zero rollout delay, and evolves a two-dimensional linear game.
The phase process is exogenous.  No TD residual, policy-gradient estimator,
critic error, Markov mixing certificate, nonlinear policy, or standard MARL
return is present.  The result is therefore a sensor ceiling and theorem
falsification tool, not the final algorithm.

Before any GPU work, a separate outcome-free design must provide all of:

1. a predictable observable certificate using only already-paid past
   gradients/secants or explicitly charged probes;
2. a low-complexity implementation with no full covariance, Hessian inverse,
   or uncharged lookahead;
3. a proved queue/resource reserve and last-iterate Lyapunov bound for the same
   executable action;
4. a CPU feasibility gate requiring headroom retention across every core
   clock/geometry group, with new development and confirmation seeds;
5. an explicit stop rule if the certificate collapses to never/always optimism
   or loses the strong fixed comparator.

## Final authorization

Continue CPU theory and observable-sensor development.  Do not register formal
seeds and do not run standard MARL, GPU, or HPC4 from LCO-H1 alone.
