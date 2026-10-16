# T-054 paired seed-cluster power audit

## Decision

T-053A remains an honest 11/12 pilot failure. In particular, its frozen
per-task gate failed for CliffWalking and is not reinterpreted as a pass.
T-054 is a read-only, post-pilot design audit. It authorizes only a separately
preregistered 64-seed CPU confirmation pilot using entirely new master seeds;
it does not authorize formal evidence, GPU work, HPC4 work, or reuse of the
eight T-053A seeds.

## Provenance and estimand

The audit reads the frozen T-053A endpoint table (SHA-256
`e4871aace2a0f25ca3d6a64a927f14a3e28e0108e723093fe1a665aba8cb0798`)
and the T-052A exact-binomial static results (SHA-256
`30e70295fb877edb2af371f73785bcc5aa2b51a165d15c078242b56fe30905a4`).
It creates no new learning trajectory.

For each task, a master seed is the resampling cluster because the same common
and private innovation bank is reused across the controller and comparators.
Let \(\bar C_j\) and \(\bar F_j\) denote the controller and strong-fixed risks
averaged across master seeds in cell \(j\). The frozen task statistic is

\[
  R=\exp\!\left\{\frac{1}{J}\sum_{j=1}^J
  \log\frac{\bar C_j}{\bar F_j}\right\}.
\]

The audit resamples complete seed columns, not endpoint rows. Its delta-method
influence also preserves all within-seed, across-cell dependence. The 50,000
bootstrap replicates are diagnostic intervals rather than post-hoc hypothesis
tests.

## Results

| Task | Observed ratio | T-052A ratio | Cluster bootstrap 95% interval | Bootstrap Pr(ratio <= 0.97) | Conservative seeds |
|---|---:|---:|---:|---:|---:|
| CliffWalking | 0.994533 | 0.862234 | [0.849310, 1.167223] | 0.42244 | 48 |
| FrozenLake 8x8 | 0.750072 | 0.862295 | [0.611268, 0.902560] | 0.99812 | 56 |
| Taxi | 0.946403 | 0.862190 | [0.867048, 1.021628] | 0.76654 | 12 |

The CliffWalking estimate is therefore genuinely inconclusive at eight seeds:
its uncertainty interval contains both the T-052A stationary prediction and a
ratio above one. This does not prove that CliffWalking will pass with more
seeds. It does show that the failed point estimate is not evidence of a
sampler or budget-accounting error and is statistically too imprecise to
separate the two scientific explanations.

## Conservative design calculation

For each task, the influence-function standard deviation is inflated to its
one-sided 97.5% chi-square upper confidence bound. The required sample size is

\[
 n=\left\lceil
 \left[
 \frac{z_{0.95}\,\sigma_{U}}
 {\log(0.97)-\log(R_{\rm theory})}
 \right]^2
 \right\rceil.
\]

The maximum taskwise requirement is 56. The next power-of-two design with a
minimum of 64 seeds is therefore 64. This is a prospective design choice for a
new experiment identifier, not a modification of T-053A.

## Reproducibility

The machine-readable result is `docs/t054_paired_power_audit.json`, SHA-256
`3c18c64f10dd6d60263c699967e398f8a9adbd29c2003c50e695c73a52850b9d`.
An independent rerun was byte-identical. The audit uses 50,000 bootstrap
replicates with RNG seed 54001. Four targeted tests cover the influence
calculation, deterministic cluster bootstrap, power monotonicity, and the full
frozen-input execution path.

