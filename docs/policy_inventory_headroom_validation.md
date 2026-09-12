# Policy-inventory outcome-free headroom validation

## Material Passport

- Artifact ID: `PIC-HEADROOM-STATIC-20260901-RESULT`
- Frozen preregistration commit: `31ed16c`
- Design hash: `f3324deff3276d034c52293f0d0c56b68a6e9aa7a19548e3ce258c4fcb788d52`
- Artifact type: deterministic analytic design ceiling
- Scientific efficacy population: none
- Formal/GPU evidence: none
- Raw result: `docs/policy_inventory_headroom_results.json` (local only)
- Raw bytes: `51,376,836`
- Raw SHA-256: `0840be12d7d79d3903ae1912d2ec02cf8d5ebed79bbfc21e0e6bb00d4c99e1e3`

## Frozen result

All 155,520 declared state-policy evaluations completed with finite steps,
objectives and utilities.  The continuous controller recovered the base
certified step in every zero-pending state, took nonzero steps in every
maximum-inventory state, and pointwise dominated the selected ray-level
baseline.  These structural checks do not establish material value.

| Metric | Frozen result | Gate | Decision |
|---|---:|---:|---:|
| aggregate dynamic / strong-baseline utility | 1.015039 | at least 1.05 | **fail** |
| rays with at least 5% gain | 4.1667% | at least 60% | **fail** |
| zero-pending base recovery | 100% | at least 99% | pass |
| maximum-inventory nonzero update | 100% | at least 30% | pass |
| pointwise dominance | true | true | pass |
| complete and finite | true | true | pass |

Only 4/6 mandatory gates pass.  `all_gates_pass=false` and
`stochastic_pilot_authorized=false`.

## Diagnostic decomposition

The following values were not success gates and are reported only to explain
the stop decision.

- Median ray-level gain is 1.4454%.
- Alternating workload has the largest mean gain, 2.7871%; steady-high,
  bursty and steady-low workloads remain between 1.24% and 1.80%.
- Only three rays reach 5%, all in alternating/mixed-or-rotating geometry with
  policy variance 0.75.
- The strong ray-level winners are count shrinkage in 31/72 rays and
  absolute-linear shrinkage in 23/72; risk thresholds, fixed scales and total
  risk shrinkage cover the remainder.

Thus simple, globally selected causal shrinkage rules capture nearly all of the
pointwise convex optimum on the declared workload variation.  The failure is
not caused by a shield that always rejects, a missing dynamic action, numerical
failure, or an artificially weak comparator.

## Decision

Stop the current **policy-inventory controller as a positive ICML mainline**.
Do not launch its stochastic Gaussian-game pilot, formal seeds, standard MARL
benchmark or GPU job.  Do not change the frozen workload formulas, baseline
families, gates or thresholds and do not use the three favorable rays as a new
confirmation population.

The exact Gaussian factorization and physical-risk drift identity remain valid
mathematical observations.  They may be retained as supporting analysis or a
future diagnostic, but a conditional convergence theorem for a controller with
only 1.5% optimistic analytic headroom is not an adequate paper contribution.

The accumulated evidence now rejects the broader strategy of repeatedly
seeking a sophisticated online controller that narrowly improves a
well-tuned static rule on stationary or mildly switching affine/quadratic
surrogates.  A further successor requires a substantively different research
premise with an endogenous task-level decision and a paper-level benchmark,
not another reparameterization of participation, step size, graph, transport,
or shrinkage.

