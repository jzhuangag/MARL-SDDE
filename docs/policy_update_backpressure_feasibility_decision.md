# Policy-update backpressure feasibility decision

Date: 2026-09-01.
Status: completed exact CPU development audit; not a preregistration or
manuscript result.

## Decision

**STOP THE UNIVERSAL POLICY-UPDATE BACKPRESSURE CONTROLLER.**

Both predeclared problem-value gates fail against the per-cell strong baseline.
No implementable PUB controller, sampled pilot, formal seed registry, standard
MARL benchmark or GPU run is authorized by this result.  The first-pass result
and the beam-saturation audit are both retained; neither gate, cell nor metric
was changed after observing the outcome.

The result also exposes a structured, theory-relevant split: dynamic scheduling
has substantial value under persistent strong cross-agent coupling, and almost
none under weak coupling or rapidly rotating stragglers.  This split permits a
single bounded next action: derive an outcome-independent phase theorem before
running any new efficacy experiment.  It does not permit filtering the current
144 cells, relabeling them as a success, or tuning a controller on them.

## Declared problem-value screen

The scan contains 144 exact cells:

- three agents, two Markov states and binary actions;
- discount factors `0.6` and `0.85`;
- transition-focus probabilities `0.65` and `0.9`;
- coupling levels `0.55`, `0.75` and `1.0`;
- persistent, bursty and rotating-straggler latency families;
- four initial factorized policies;
- wall-clock horizon 24.

For every cell, the strong baseline is selected from 143 candidates: fixed
trust radii, fixed age decays, fixed direct-TV gates, fresh serial learning and
barrier batch learning.  Hyperparameters are selected per cell and are
therefore deliberately favorable to the baseline.  The exact optimal
factorized deterministic policy supplies the wall-clock regret reference.

The dynamic comparator is a feasible finite-width non-myopic beam schedule over
five update scales.  It is a lower bound on achievable dynamic value, not an
oracle ceiling or an executable controller.

## Frozen first-pass outcome

| Gate or diagnostic | Result |
|---|---:|
| Cells | 144 |
| Static candidates per cell | 143 |
| Beam width | 64 |
| Median wall-clock regret reduction | **7.2183%** |
| Cells with at least 5% reduction | **54.1667%** |
| Maximum reduction | 47.7142% |
| Minimum reduction | -20.0247% |
| Median reduction at least 10% | **FAIL** |
| At least 60% of cells reach 5% | **FAIL** |

`authorized_next_step` is therefore `false` in the machine-readable result.

## Beam-saturation audit

The original baselines, cells, horizon, metric and gates were reused without
change.  Only the dynamic search width was increased.

| Width | Median reduction | Fraction at least 5% | Gate 1 | Gate 2 |
|---:|---:|---:|---:|---:|
| 64 | 7.2183% | 54.1667% | fail | fail |
| 128 | 7.6829% | 55.5556% | fail | fail |
| 256 | 7.0328% | 54.1667% | fail | fail |

Beam search is not nested in width because extra high-looking parents can
displace different continuations during pruning.  Even the retrospective
per-cell portfolio choosing the best feasible schedule found at widths 64,
128 and 256 remains below both gates: median `8.0619%`, with `56.9444%` of cells
at or above 5%.  This portfolio is diagnostic only and does not overwrite the
width-64 first pass.

## Phase signal, not a post-hoc success population

For the best-of-width diagnostic portfolio:

| Declared subgroup | Cells | Median reduction | Fraction at least 5% |
|---|---:|---:|---:|
| coupling 0.55 | 48 | 0.0000% | 12.5000% |
| coupling 0.75 | 48 | 24.0292% | 83.3333% |
| coupling 1.00 | 48 | 18.3198% | 75.0000% |
| persistent heterogeneity | 48 | 21.4058% | 70.8333% |
| bursty slow agent | 48 | 20.0017% | 66.6667% |
| rotating straggler | 48 | 0.0000% | 33.3333% |

These are descriptive development outcomes.  They cannot justify deleting
weak-coupling cells or preregistering only the favorable cells.  They motivate
a mathematical question: whether a dimensionless load combining cross-policy
sensitivity, proposal lifetime and persistence separates a region where
dynamic scheduling is necessary from one where a strong static rule is already
near-optimal.

## Allowed successor: theory before experiments

A successor is allowed only if it changes the research claim from universal
controller superiority to a phase characterization and completes the following
without reading new outcomes:

1. define an observable/theorem-facing coordination-load parameter from the
   cross-agent policy-gradient Jacobian and the completion process;
2. prove a low-load upper result showing why static scheduling can be
   near-optimal;
3. prove a high-load separation example where barrier and accept-all rules
   incur nonvanishing wall-clock regret but a causal scheduler does not;
4. derive the executable Lyapunov rule and its finite-time error terms;
5. freeze eligibility and failure conditions before collecting a disjoint CPU
   confirmation population.

If these proof obligations cannot be closed, the phase successor stops too.
The current outcomes may be used only as exploratory motivation and a later
held-out qualitative check; they cannot set the threshold or count as formal
evidence.

## Reproduction

Source:
`experiments/policy_update_backpressure/feasibility.py`

Targeted tests:
`experiments/policy_update_backpressure/test_feasibility.py`

Commands:

```text
.\.venv\Scripts\python.exe -m pytest experiments\policy_update_backpressure\test_feasibility.py -q
.\.venv\Scripts\python.exe -c "from pathlib import Path; from experiments.policy_update_backpressure.feasibility import main; main(Path('docs/policy_update_backpressure_feasibility_results.json'))"
.\.venv\Scripts\python.exe -c "from pathlib import Path; from experiments.policy_update_backpressure.feasibility import beam_saturation_scan; beam_saturation_scan(Path('docs/policy_update_backpressure_feasibility_results.json'),(128,256),Path('docs/policy_update_backpressure_beam_saturation.json'))"
```

Six targeted tests pass after the saturation helper is added.  No sampled
trajectory, old formal result, GPU, HPC4 or remote storage was used.

