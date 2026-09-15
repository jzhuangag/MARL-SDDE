# Perishable-update phase CPU pilot validation

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-01
- Verification Status: VERIFIED
- Version Label: `perishable_update_phase_pilot_validation_v1`
- Source preregistration commit: `10967c4baada647c52802f9eca755637e3b5ad7e`
- Overall Confidence: RED_FLAG for the central performance hypothesis; SOLID for execution and reproduction

## Decision

The preregistered pilot is a reproducible failure.
Six of twelve mandatory gates pass.
The mechanism must not proceed to formal CPU seeds or a nonlinear/GPU benchmark under this design.
No gate, seed, cell, comparator or outcome was changed after the preregistration commit.

## Execution and reproduction

Both the original run and clean reproduction completed on the local CPU with exit code zero.
Each contains all 74,304 registered rows and every registered endpoint is finite.
The two 52,336,350-byte trajectory files are byte-identical:

```text
SHA-256 cbc1e1ad3c54fb52a9a7a7a126c5367eaa873d9c6d7e3a2ba91aa83919801611
```

The independently generated analyzer summaries are also byte-identical:

```text
SHA-256 7a829158424a5a8e1856feb8307b97f0c6ca5b23d8364984d8084a907cb748db
```

## Primary findings

| Phase | Geometric PUB/strong-envelope regret ratio | Median regret gain | Cells with at least 5% gain | Gate |
|---|---:|---:|---:|---|
| low | 1.01198 | -0.9561% | 0/12 | pass low-load margin |
| transition | 1.14450 | -3.5974% | 0/12 | fail |
| high | 1.19609 | -20.4943% | 0/12 | fail |

The failure is not caused by numerical divergence, missing rows, or an accept-all/reject-all collapse.
High-load median acceptance is 0.4656, all PUB endpoints are finite, and PUB accepts zero realized potential-decreasing steps.
Instead, the exact one-event safety certificate is too conservative for cumulative wall-clock performance against the cellwise selected 42-policy comparator envelope.

The secondary final-gradient gate also fails.
The high-load geometric gradient ratio is 2.17084 against a maximum of 1.05.
The transition gradient ratio is numerically very large because several selected comparators reach an exact zero gradient while PUB leaves a small nonzero residual; this does not affect the already decisive primary-regret failure.

The registered complexity gate fails as well.
Mean arithmetic operations per event are 22.446 for three agents and 43.680 for five agents, a ratio of 1.94597 against the frozen maximum 1.71667.
The implementation recomputes the scores of all remaining simultaneous completions after each selection, so the ready-set ordering layer is not linear in the size of a simultaneous batch even though each individual score is `O(n)`.

## Gate ledger

| Gate | Result | Observation |
|---|---|---|
| G1 complete and finite | pass | 74,304/74,304 finite rows |
| G2 high-load regret | **fail** | 1.19609 > 0.85 |
| G3 broad high-load gain | **fail** | 0/12 vs at least 60% |
| G4 transition regret | **fail** | 1.14450 > 0.95 |
| G5 low-load safety | pass | 1.01198 <= 1.03 |
| G6 strict phase ordering | **fail** | all median gains are negative |
| G7 nontrivial acceptance | pass | 0.4656 in `[0.05,0.95]` |
| G8 high-load final gradient | **fail** | 2.17084 > 1.05 |
| G9 vectorized complexity | **fail** | 1.94597 > 1.71667 |
| G10 no harmful accepted steps | pass | zero |
| G11 exact reproduction | pass | byte-identical raw and summary files |
| G12 provenance freeze | pass | source hashes and seeds match preregistration |

## Theory-to-algorithm implication

The experiment supports the local certified-ascent lemma but rejects the proposed bridge from that lemma to superior finite-horizon regret for the implemented controller.
The Lyapunov drift bound uses magnitude-only worst-case cross-debt, whereas the exact quadratic dynamics can contain directional cancellation.
A method that never takes a certified harmful step can therefore still make less useful progress than a strong fixed or threshold policy.
The existing high-load separation construction is not sufficient evidence that the same executable PUB rule wins on the registered Markov-completion population.

This result closes the current universal/phase-conditioned PUB performance route.
Any successor must first derive a non-myopic or direction-aware performance guarantee in an outcome-free theory commit and must receive a new experiment identity; it cannot reuse these pilot outcomes for confirmation.

## Statistical and methodological fallacy scan

Coverage: 11/11 types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Phase and cell directions were inspected; all 12 high-load cells are worse, so the aggregate high-load failure is not a reversal. |
| Ecological fallacy | NOTE | Inference is restricted to the registered cells and does not claim individual-agent behavior outside them. |
| Berkson's paradox | CAUTION | The best comparator is selected within each cell from the same pilot outcomes, intentionally biasing the envelope in favor of the baseline; this can enlarge the estimated deficit but cannot create a positive PUB result. |
| Collider bias | NOTE | No post-treatment covariate or conditioned common effect enters the registered comparison. |
| Base-rate neglect | NOTE | No diagnostic conditional probability is interpreted. |
| Regression to the mean | NOTE | Cells were fixed analytically before outcomes and were not selected for extreme observed performance. |
| Survivorship bias | NOTE | There is no attrition: all registered rows are present and finite. |
| Look-elsewhere effect | NOTE | All practical gates are preregistered and jointly mandatory; no favorable subset is promoted. |
| Garden of forking paths | NOTE | Seeds, cells, comparators, aggregation and stopping rule were frozen in an independent commit. |
| Correlation versus causation | NOTE | The simulator is an intervention experiment with common random numbers; the claim is nevertheless restricted to the registered model. |
| Reverse causality | NOTE | Policy assignment precedes measured wall-clock regret in the simulator. |

No null-hypothesis significance test or confidence interval was preregistered.
The gate assessment is therefore a deterministic practical-effect audit, not a significance claim.

## Reproducibility verdict

- Method: deterministic clean rerun with identical code, seeds and environment
- Verdict: REPRODUCIBLE
- Raw artifact: exact SHA-256 and byte-size match
- Analyzer summary: exact SHA-256 match
- Timing metrics: deliberately excluded from the scientific artifact

Raw outputs remain local under the two ignored result directories.
The machine-readable validation record is `docs/perishable_update_phase_pilot_validation.json`.
