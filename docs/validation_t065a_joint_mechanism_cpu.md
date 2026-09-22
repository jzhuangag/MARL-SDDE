# T-065A CPU mechanism-pilot validation

## Decision

T-065A is a positive mechanism pilot: all twelve frozen gates pass after an
independent clean-rerun audit.  It supports continuing the discrete online
joint `(q, eta)` program to a theorem-aligned delayed affine-TD design.

It is not formal evidence and is not sufficient for an ICML submission.  In
particular, this experiment evaluates the certified one-block drift score, not
terminal TD risk on a standard RL benchmark.  It uses point estimates rather
than time-uniform conservative confidence bounds.  No formal seeds, GPU job,
HPC4 job, or standard-RL pilot is authorized by this result alone.

## Frozen provenance

- Discrete foundation commit: `a88963e1e98038e2d80778d8cbff04cd0c43caeb`
- Independent preregistration commit:
  `dcaacc192e8a97824af59998eec4444f2a1ad539`
- Frozen configuration SHA-256:
  `782DA5C20008A7B0D8F3B6BC7013F817202A585E540905595F7F2FC651BB331C`
- Workload: 324 cells, 64 registered seeds, 20,736 endpoints.
- Hardware: local CPU only.
- Sensor/evaluation random streams are separated by deterministic labelled
  SHA-256 seed derivation.

## Primary results

| Metric | Frozen threshold | Observed |
|---|---:|---:|
| Median cell normalized regret to clairvoyant joint action | <= 0.10 | 0.00003303 |
| 90th-percentile cell normalized regret | <= 0.35 | 0.0369219 |
| Observable joint beats fixed `(8,0.05)` | >= 60% cells | 92.9012% |
| Joint oracle beats both one-dimensional actions | >= 30% cells | 96.9136% |
| Observable joint beats both one-dimensional actions | >= 50% cells | 75.3086% |
| q nonincreasing with rho | >= 90% matched paths | 96.2963% |
| eta nondecreasing with state signal | >= 90% matched paths | 100% |
| Dual-budget validity | all endpoints | 100% |

The median absolute errors of both selected `q` and `eta` are zero; exact
integer-q agreement with the clairvoyant action is 70.0328% over endpoints.
Fingerprint estimation has rho RMSE 0.03312.  The untruncated noise estimator
is nearly unbiased (mean bias -0.00146; RMSE 0.58597 over the deliberately
wide noise grid).  Clipping the noisy signal estimate at zero induces a
positive mean bias 0.01823; this is acceptable for the frozen point-estimate
pilot but must be replaced by lower/upper confidence bounds in the theorem-
facing controller.

## Gate ledger

The runner establishes G1--G11.  It intentionally writes `G12=false` in each
single-run summary because byte reproducibility cannot be established by the
run being audited.  The separate clean reproduction then established G12:

| Artifact | Bytes | SHA-256 | Reproduction |
|---|---:|---|---|
| `endpoints.csv` | 5,352,016 | `B187AE05C0B8D77D2C0C1BE65B3D13CED2CFFE610EBA36854765368C68D843FF` | exact |
| `cells.csv` | 51,917 | `FCAB67429B20F11D8F64753AF7E839AC5899463A9ACD6D0F8D489C6D685512DE` | exact |
| `summary.json` | 174,761 | `59505F260A461D1EC3CB5B836EA17C9D79FD6E5FBC668EF387EEB86AD8C4E389` | exact |

Thus G1--G12 all pass without changing any frozen seed, coefficient, gate, or
threshold.  The full endpoints and clean-reproduction outputs remain local;
the compact cell table and summary are versioned.

## Scientific interpretation

This rejects the immediate concern that online joint control is merely a
cosmetic combination of q-only and eta-only rules: the true joint optimum is
strictly better than both one-dimensional restrictions in 96.91% of the
registered coefficient cells, and the observable sensors recover enough of
that value to beat both in 75.31%.

It does not yet show that the same advantage survives real TD trajectories,
finite learning horizons, Markov bias, sensor opportunity cost in terminal
risk, or strong task-by-budget baselines.  Those are the next falsification
targets.  The next stage must freeze a delayed affine-TD runner in which the
probe costs reduce actual available learning updates and the evaluation
metric is terminal/averaged prediction risk rather than the controller's own
surrogate score.
