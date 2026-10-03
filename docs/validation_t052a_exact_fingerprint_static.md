# T-052A exact fingerprint static gate: validation

## Decision

T-052A passes all 12 mandatory gates and authorizes a separate T-053 local
CPU sampled-pilot preregistration. It does not authorize immediate sampled
execution, formal seeds, GPU, HPC4, or a nonlinear benchmark.

The exact Binomial controller, after charging every probe message, has:

| Metric | Result |
|---|---:|
| Aggregate geometric improvement | 13.7761% |
| Overhead 8 improvement | 15.6112% |
| Overhead 32 improvement | 11.9010% |
| Oracle-active cells improved | 81/81 = 100% |
| Maximum expected ratio over all cells | 1.022059 |
| Frozen no-harm limit | 1.05 |

The worst cell is CliffWalking, delay 8, overhead 32, and correlation 0.3.
Both its oracle and strong fixed action are (q=16), and the exact expected
ratio is 1.0220586. Thus the static design retains a 2.8 percentage-point
margin to the frozen no-harm gate.

## Relation to the failed T-051A result

T-051A remains failed at 1.0501064. T-052A does not change that result. It
uses a new prospective theorem and experiment identifier: with two probe
actors the match count is exactly Binomial, so the 97 possible outcomes can
be integrated rather than bounded by a generic Hoeffding tail. The new exact
calculation improves the aggregate full-cost certificate from 12.0161% to
13.7761% and reduces the maximum cell bound from 1.0501064 to 1.0220586.

## Reproduction

The result JSON SHA-256 is
`30e70295fb877edb2af371f73785bcc5aa2b51a165d15c078242b56fe30905a4`.
A clean rerun was byte-identical; its duplicate directory was removed. The
audit used 126 analytic cells and zero sampled trajectories, seeds, GPU jobs,
or HPC4 jobs.

## Next gate

T-053 must be independently preregistered before execution. It must sample
the public regenerative Gymnasium kernels under the unchanged
marginal-preserving trajectory-switch coupling, implement the observable
two-agent fingerprint count, retain the (10^{-4}) contraction horizon,
charge both budgets and delay, and compare with the exact strong fixed
baselines. Any sampled failure will stop formal execution without altering
the T-052A population or thresholds.
