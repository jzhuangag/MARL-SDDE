# T-057A formal CPU validation

## Formal decision

T-057A passes all fourteen preregistered formal gates. The result is formal
empirical evidence for the registered claim: a fully charged, two-agent
fingerprint classify-and-commit controller adapts fixed participation to
cross-agent trajectory correlation in delayed multi-agent fixed-policy
Markov TD, outperforming the strong overhead-specific fixed-q baseline across
three exact public Gymnasium kernels.

The decision does not authorize a general actor--critic, nonlinear function
approximation, arbitrary-dependence, or deep-MARL claim. GPU and HPC4 remain
unauthorized for T-057A.

## Primary and simultaneous inference

The formal unit is the master seed. All 50,000 bootstrap replicates resample
complete columns of 256 master seeds across every registered cell and
comparator. Pilot seeds are excluded. Endpoint rows are never treated as
independent samples.

| Estimand | Point | Frozen one-sided bound | Threshold | Decision |
|---|---:|---:|---:|---|
| Aggregate controller / strong fixed-q | 0.872378 | 0.882586 (95%) | <=0.95 | pass |
| CliffWalking | 0.865051 | 0.885529 (Bonferroni 98.333%) | <=0.97 | pass |
| FrozenLake 8x8 | 0.881031 | 0.907634 (Bonferroni 98.333%) | <=0.97 | pass |
| Taxi | 0.871127 | 0.887630 (Bonferroni 98.333%) | <=0.97 | pass |
| Delay 0 | 0.872480 | 0.889333 (Bonferroni 97.5%) | <=0.97 | pass |
| Delay 8 | 0.872276 | 0.889209 (Bonferroni 97.5%) | <=0.97 | pass |
| Inactive-cell no-harm | 1.014405 | 1.020415 (95%) | <=1.05 | pass |
| True-rho oracle proximity | 1.013378 | 1.017592 (95%) | <=1.20 | pass |

The aggregate point estimate is a 12.7622% risk reduction. The controller
improves 81.48% of the 54 oracle-active cells; its preregistered one-sided 95%
lower bound is 79.63%, above the 60% breadth threshold. The fingerprint
standardized residual RMSE is 0.91832 and participation is directionally
nonincreasing with correlation.

## Cost and comparator integrity

Every endpoint charges 96 independent two-agent probe blocks, all probe
environment transitions, learning messages, the candidate-specific integer
horizon, and delay reserve. The strong fixed-q comparator pays no probe and
receives the full total message budget. It is q=4 at overhead 8 and q=16 at
overhead 32. Hence the formal improvement is not relative to all-agent or an
under-tuned fixed baseline.

The controller's online decision requires only a Bernoulli match count and a
three-action scan. It uses no covariance estimate, Hessian, inverse,
preconditioner, or online model solve.

## Reproduction and software validation

The primary run contains 21,504 endpoints, 84 cells, and 256 isolated formal
seeds. A clean rerun required 1,687.31 seconds and produced byte-identical
artifacts. The verified duplicate was removed after comparison; primary
artifacts were retained.

| Artifact | SHA-256 |
|---|---|
| `endpoints.csv` | `19ebc6aaa5c99652bd51a70e9257d1641b4f6db1ba4d32c22247d1886a89f964` |
| `cells.csv` | `d15841a0602c72e8d3d3650967326cac7efe19b87d5a23d6007b2b7767f09fe4` |
| `summary.json` | `17a36b8bbd39734d7c9e52bed1f7aa212d7a096474ac207c3aa275c6ac496a1a` |

The complete `dependence_delay_linear` suite reports 377 passed, zero
failures, and zero errors. The formal analyzer itself reruns byte-identically;
its JSON SHA-256 is
`43dfc9351c6f66074dfdd7242a000d070a724f0e26cc0aca0088fecbeeec8ddc`.

## F11 implementation disclosure

The frozen analyzer's F11 implementation compared only the replayed and
stored gate dictionaries, whereas the preregistered prose required the full
summary to replay from endpoints. This was under-enforcement in code. A
separate, stricter post-result integrity audit recomputed every field and
checked complete coverage. It found zero duplicate cell-seed rows and a
maximum absolute numerical difference of `3.22e-15`, below `1e-12`; therefore
the textual F11 intent is satisfied and the formal decision is unchanged.
The strict audit reruns byte-identically with SHA-256
`a029474be7cf6b288be92d654a615a3d92d1b85c1b9fb4c335d3eaef2c2bb426`.

## Theoretical alignment

T-056 closes the end-to-end finite-budget loop for this claim. Conditional on
the independent Binomial fingerprint count, the selected fixed-q action has
an exact delayed PR risk containing transient bias, Markov lag covariance,
integer horizon, both budgets, probe cost, and delay. Averaging those risks
over the exact count distribution yields the controller's finite expected
risk. T-050's participation phase is the stationary limit and explains why
the optimal q decreases with correlation; it is not substituted for the
sampled formal outcome.

