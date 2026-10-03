# T-073 continuous-QP architecture calibration validation

## Decision

T-073 passes all nine frozen descriptive architecture criteria and reproduces
byte-for-byte. It is the first implementation in this chain to preserve all
learning transitions, exceed the aggregate and cell-coverage criteria, remain
positive for every delay, and meet a measured low-complexity gate.

This does not yet authorize a new-seed pilot. The calibration is explicitly
outcome-informed, strict coverage is only 55.56% against a 55% criterion, and
two scientifically important strata remain negative. The next design must add
a persistent mixing/SNR confidence state before preregistration rather than
loosening these strata or reusing the old seeds.

## Frozen results

| Quantity | Result |
|---|---:|
| Nonstationary improvement vs strong static | 8.4573% |
| Strictly improved nonstationary cells | 55.5556% |
| Single-switch / alternating improvement | 5.1862% / 11.6155% |
| Delay 0 / 1 / 3 improvement | 8.6715% / 9.2555% / 7.4354% |
| Stationary controller/static ratio | 1.08087 |
| Accepted continuous nonlocal rate | 22.8865% |
| Rollback rate | 76.2391% |
| Mean QP iterations per recipient-decision | 28.8935 |
| Mean / maximum safety debt | 0.04262 / 2.48556 |

Every endpoint uses all 240 environment transitions for learning, zero extra
probe transitions, and no more than 18 message units. All Q1--Q9 pass.

## Mechanism and remaining phase boundary

Compared with T-072, continuous covariance-aware weighting raises the
high-noise (`noise_scale=0.5`) improvement from -0.1595% to +1.4839%, and the
aggregate improvement from 7.6746% to 8.4573%. This is evidence that the
covariance term is operational rather than decorative.

However, target-scale 0.1 remains -5.5949%, and temporal-correlation 0.9 is
-0.8512%. Five within-block observations do not reliably distinguish weak
heterogeneity from persistent Markov noise. T-017 already rules out uniform
adaptation over unrestricted unknown mixing. A successor must therefore carry
fingerprints across blocks, regularize the QP with a predictable confidence
radius, and state guarantees only for separated mixing or independently
certified mixing bounds. Low-confidence decisions must contract continuously
toward the local vertex.

## Reproduction and integrity

The frozen local-CPU run completed in 504.72 seconds; the clean reproduction
completed in 443.50 seconds. `endpoints.csv`, `cells.csv`, and `summary.json`
are byte-identical. Their SHA-256 hashes are stored in the machine-readable
validation file. No GPU, HPC4, nonlinear benchmark, new seed, formal run, or
`/project` write occurred.

## Statistical-fallacy scan

Coverage is 11/11. No cell or seed was filtered, no missing endpoint was
discarded, and all nine criteria are reported. The four-seed parameter audit
and reuse of T-071A seeds are explicit, preventing reuse as independent
evidence. Aggregation is decomposed by schedule, delay, scale, noise, spatial
correlation, and temporal correlation; the negative strata are retained.
There is no collider adjustment or agent-to-population inference. Causal
interpretation is restricted to the common-random-number simulator, whose
target schedules precede controller decisions.
