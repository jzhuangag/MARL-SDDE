# EXP-019A exact-Blackjack CPU pilot validation

## Decision

EXP-019A is a prospective pilot failure: 6/8 frozen gates pass.  Formal seeds,
MinAtar, HPC4, and GPU execution are not authorized.  The frozen thresholds,
seeds, q grid, cells, step size, and endpoints remain unchanged.

## Provenance and validity

- preregistration commit:
  `f8a0242756904b0531c516d1a51fd7fbe9460cee`;
- configuration SHA-256:
  `299ff2c40ace9620040e75dca3214f24a350033cbe169f73a497bb2c84b8e0d8`;
- 32 new CRN pilot seeds, 72 cells, 4,608 endpoint rows;
- all rows finite, exact Bellman residual gate passed, and no message or
  environment budget violation occurred;
- local CPU only; no external benchmark, GPU, HPC4, or `/project` write.

## Frozen gates

| Gate | Threshold | Result |
|---|---:|---:|
| Complete and finite | all rows | pass |
| Dual-budget validity | zero violations | pass |
| Aggregate MSVE-AUC gain | ratio <=0.95 | **0.999833, fail** |
| Active directional transfer | >=60% cells | 22/36 = 61.111%, pass |
| Inactive no-harm | ratio <=1.02 | 1.000000, pass |
| Terminal MSVE gain | ratio <=0.98 | **0.999714, fail** |
| Exact-value residual | <=1e-10 | pass |
| CPU/task scope | no GPU/external task | pass |

The aggregate AUC improvement is only `0.016654%`; the terminal improvement
is only `0.028645%`.  Within active cells the AUC ratio is `0.99966694` and
the terminal ratio is `0.99942719`.  Directionality barely clears its gate,
but the effect is scientifically negligible and cannot authorize formal.

## Failure diagnosis

The registered 280-parameter static proxy predicted 6.8798% aggregate value,
but realized only 0.01665%; the realized/static transfer fraction is 0.2421%.
The discrepancy is structural rather than a seed-power issue:

- the proxy contains only the correlation-limited variance term divided by
  usable updates;
- finite-horizon TD error also contains contraction/bias from the initial
  parameter and the stale-parameter recursion;
- at low rho the proxy selects q=32, but the reduced number of updates makes
  q=32 slightly worse than the q=16 fallback;
- at high rho the selected smaller q is directionally better, but only by
  roughly 0.1--0.2% per cell.

Thus T-029 correctly found a variance-resource phase diagram but did not
establish that its oracle value transfers to the registered learning
dynamics.  The next admissible step is a bias-aware finite-time selector or
exact tabular mean-square audit.  Another sampled controller, larger seed
count, Asterix pilot, or GPU run is not justified yet.

## Exact reproduction

An isolated rerun is byte-identical:

- `metrics.csv` SHA-256:
  `dc76e83ce1e5ef5976aa4ccc465d4804bbf557d6233e3f84e6dc035c3780c4ac`;
- `summary.json` SHA-256:
  `40de5018afcc2ad09b95835a76d2f59717901045870ac94c0f435ba1e73860a8`;
- `manifest.json` SHA-256:
  `6d76f8aba8756e60c948ae554dd5df6eac16dfe79b4f4839c80e20815a12005e`.

The complete experiment test suite reports `295 passed, 7 skipped`.
