# TSP-CURVE-001 confirmation validation

## Status

TSP-CURVE-001 is a seed-separated CPU confirmation of the convergence-curve
claims used in the TSP manuscript.
The public design, candidate policies, metrics, development seeds, confirmation
seeds, and bootstrap rule were frozen in commit `b8927cd` before the
confirmation trajectories were generated.

## Design separation

- Development population: 16 seeds (`31000001`--`31000016`).
- Confirmation population: 64 disjoint seeds (`32000001`--`32000064`).
- Strong fixed comparator: `q=4`, selected only from the development population
  by geometric normalized parameter-error AUC over all 12 registered cells.
- Confirmation cells: persistence in `{0, 0.9, 0.98}`, cross-agent correlation
  in `{0, 0.9}`, and maximum delay in `{0, 8}`.
- Compared policies: the joint Lyapunov certificate and fixed
  `q in {1,4,16,32}`; every fixed comparator receives its own certified Markov
  spacing and step in each cell.
- Resource accounting: identical message and environment budget for every
  policy; all checkpoints use charged resource fraction rather than update
  count.

## Confirmatory results

The complete confirmation population contains 157,440 finite, budget-valid
checkpoint rows.
Relative to the development-selected strong fixed `q=4` comparator:

| Metric | Ratio | Reduction | Paired-bootstrap 95% CI for ratio |
|---|---:|---:|---:|
| Normalized parameter-error AUC | 0.900913 | 9.9087% | [0.891362, 0.910190] |
| Normalized return-estimation-error AUC | 0.886789 | 11.3211% | [0.867796, 0.907230] |
| Terminal normalized parameter error | 0.735166 | 26.4834% | descriptive |
| Terminal normalized return-estimation error | 0.783277 | 21.6723% | descriptive |

The joint certificate strictly improves both AUC metrics in 9 of 12 cells and
ties the fixed comparator in the remaining 3 cells.
No registered cell is worse.
The return metric is the squared error of the fixed-policy discounted-return
estimate at the registered initial state; it is not an episodic control-return
claim.

## Recovery and provenance

The confirmation runner wrote the complete `metrics.csv` and
`action_table.csv` before failing while serializing NumPy integer keys in the
JSON summary.
No scientific trajectory was rerun.
The serialization cast was corrected, and
`recover_convergence_summary.py` regenerated only the deterministic summary
from the existing complete metrics.

- Raw metrics SHA-256:
  `643cb48d97d725d9e9d481e0e21d0a8c699d3f64aef7bb3d968e2af7e5d194b8`
- Action table SHA-256:
  `72ba8a0000d3e2c3d774ce68ed8cdad12844a135a930b5bce4842a843e4addfd`
- Manuscript aggregate CSV SHA-256:
  `071516e5d05b36bb2a1cf345a8ab369713c6c31c406115b4f1e2817aca7af6d5`
- Trajectory-generating source SHA-256:
  `dc4b74231c6882b0eaffb408c1599ac30150f600f6cd6817a3eea0091e3e9532`

The raw confirmation directory remains excluded from Git because it is a
reproducible experiment artifact.
The aggregate curve data, figure, code, frozen configuration, and validation
record are included in the repository.
