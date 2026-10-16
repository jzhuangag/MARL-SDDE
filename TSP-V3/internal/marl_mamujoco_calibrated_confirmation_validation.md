# TSP-V3 MaMuJoCo calibrated-catalogue confirmation validation

## Decision

**Pass.** The fresh-seed confirmation experiment
`TSP-V3-MARL-MAMUJOCO-CAL-CONF-001` passes every frozen mandatory gate and
authorizes admission of the confirmed MaMuJoCo return evidence into the IEEE
TSP manuscript.

The scientific source and preregistration were frozen at commit
`b599cbdc6a4d9c73031019a7ebe3509059772e90`. No task, method, seed, budget,
controller constant, threshold, or analysis rule was changed after observing
confirmation outcomes.

## Execution and payload audit

The 80 registered runs were executed on HPC4 A30 GPUs in ten nonoverlapping
eight-cell waves. Every array cell completed with exit code `0:0`.

| Global indices | Slurm array |
|---:|---:|
| 0--7 | 1923247 |
| 8--15 | 1923718 |
| 16--23 | 1924845 |
| 24--31 | 1925596 |
| 32--39 | 1925936 |
| 40--47 | 1926310 |
| 48--55 | 1927057 |
| 56--63 | 1927784 |
| 64--71 | 1928294 |
| 72--79 | 1928831 |

For every cell, the audit verified the registered global-cell mapping, finite
progress, exact message and environment accounting, a clean frozen source
checkout, pinned HARL commit
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`, absence of evaluation-return use
by the controller, and the per-artifact `SHA256SUMS`. The complete lattice has
16 controller runs and 16 runs for each fixed comparator
`q in {1,2,4,8}` across independent and shared coupling.

## Frozen confirmation results

| Quantity | Confirmed value | Frozen gate |
|---|---:|---:|
| Independent selection of `q=8` | 8/8 = 100% | at least 87.5% |
| Shared selection of `q=2` | 8/8 = 100% | at least 87.5% |
| Independent AUC mean vs. fixed `q=8` | -0.5419% | paired lower bound at least -2% |
| Independent paired lower bound | -0.6581% | at least -2% |
| Shared AUC mean vs. fixed `q=2` | -0.9969% | paired lower bound at least -2% |
| Shared paired lower bound | -1.2394% | at least -2% |
| Equal-mixture AUC mean vs. fixed `q=8` | **+8.3408%** | at least +5% |
| Equal-mixture paired lower bound | **+1.1074%** | at least 0% |
| Positive mixture pairs | **7/8** | at least 7/8 |
| Exact one-sided sign-test p-value | **0.03515625** | at most 0.05 |
| Equal-mixture final return vs. fixed `q=8` | **+4.6367%** | at least -2% |
| Maximum probe message fraction | 0.768% | at most 1% |
| Maximum selection overhead fraction | 3.08364e-7 | at most 2% |

The result confirms the intended mechanism: the fully charged controller
selects the high-parallelism action under independent sampling and the
lower-correlation-efficient action under shared sampling. It remains within
the registered 2% noninferiority margin of the corresponding regime-specific
fixed action, while outperforming the strongest single fixed comparator on the
registered mixture distribution.

## Reproduction and hashes

The frozen analyzer was executed twice into disjoint output directories. The
CSV and JSON outputs were byte-identical. A third replay-enabled invocation
set the replay gate and produced the final passing decision.

| File | SHA-256 |
|---|---|
| `summary.csv` | `9cf2e97daa289363545075f74c75ee16afbb094fff1f693c53a08c3004d13ad4` |
| `gate.json` | `1e9ce4a04122736adfa1b49339ea16a7d5f5483dededd5a3e83a6029d6f7d031` |
| `figure.pdf` | `018be5e4260679bcd5f4a920cf331c24ad68f222bbda626e519ed6275425abdb` |

Remote evidence remains under
`/scratch/jzhuangag/MARL-SDDE-TSP-V3-MAMUJOCO-CAL-CONF-001`. The compact
confirmed analysis outputs are versioned under
`TSP-V3/results/marl_mamujoco_calibrated_confirmation_20260930`.
