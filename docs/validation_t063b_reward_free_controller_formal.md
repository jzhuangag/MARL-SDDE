# T-063B formal validation

## Decision

T-063B is a **qualified formal failure under the frozen Amendment 1 analysis
plan**. The prospective primary run is strongly positive, its independent
reproduction is byte-identical, and the corrected collision and replay gates
pass. One mandatory efficacy gate does not pass: the lower confidence bound
for strict cell breadth is below the preregistered 0.60 threshold.

No threshold, seed, cell, comparator, or analysis rule was changed after seeing
the outcome. T-063A remains a separate formal failure and is not reclassified.

## Formal result

| metric | point | one-sided bound | threshold | result |
|---|---:|---:|---:|---|
| aggregate controller/strong | 0.828309 | 0.845939 upper | 0.95 | pass |
| Asterix | 0.837884 | 0.882160 upper | 0.98 | pass |
| Breakout | 0.803835 | 0.839887 upper | 0.98 | pass |
| Seaquest | 0.843773 | 0.886885 upper | 0.98 | pass |
| delay 0 | 0.828233 | 0.849406 upper | 0.97 | pass |
| delay 8 | 0.828384 | 0.849716 upper | 0.97 | pass |
| true-rho oracle proximity | 1.013351 | 1.019975 upper | 1.15 | pass |
| strict-cell breadth | 0.642857 | 0.547619 lower | 0.60 | **fail** |

The aggregate geometric error reduction is `17.1691%`. The point strict
fraction is `54/84 = 0.642857`; the complete-seed cluster-bootstrap lower bound
is `0.547619`, so the frozen inferential gate fails despite the positive point
estimate.

All other mandatory gates pass: coverage and seed isolation, finite/full cost,
aggregate/task/delay efficacy, true-rho proximity, participation direction,
fingerprint calibration, aggregate collision, numeric summary replay, byte-exact
reproduction, and the full test manifest.

## Collision and replay audit

The corrected aggregate collision statistic is 49 matches over 147,456 trials,
with rate `0.0003323025` and one-sided upper probability `0.0004216056`, below
the preregistered bound `0.0007716049`. The Amendment 1 recursive numeric replay
gate passes at absolute and relative tolerance `1e-12`; all three primary and
reproduction artifacts also match byte-for-byte.

## Provenance

- configuration SHA-256: `d49c8a0a5eba25686649262e2822440c2b6e8b8e386c6e31e52ac4c2560c7bb`;
- primary/reproduction endpoints SHA-256: `e9a279a542aef22c802b360e01e917526a43d8f4eb8821c95493dddca0e92505`;
- primary/reproduction cells SHA-256: `5763e7addbbd2da11c017d1e91b5772a72a800e434591df5ddc17ff0e67efd2e`;
- primary/reproduction summary SHA-256: `785df50d772e8df1686e16cda3a00a3bd4e752fc7269747b15ac5b6711c1154d`;
- bootstrap: 50,000 complete-master-seed replicates, seed 63001;
- runtime: 45,153.664 seconds, four local CPU workers;
- GPU/HPC4/project storage: not used.

The machine-readable analyzer output is
`docs/validation_t063b_reward_free_controller_formal.json`.

## Consequence

T-063B provides reproducible, strong positive evidence for the aggregate,
task-wise, delay-wise, collision-calibrated controller under the registered
fixed-policy delayed nonlinear-feature Markov-TD scope, but it does not satisfy
the preregistered all-gates formal claim because breadth uncertainty remains too
large. The result is suitable as qualified evidence, not as an unqualified
claim that the controller wins in at least 60% of cells.
