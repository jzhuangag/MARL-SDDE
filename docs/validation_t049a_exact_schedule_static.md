# T-049A exact standard-task schedule scan: validation

## Decision

T-049A is a mandatory-gate failure and does not authorize a learned EXP-021A
pilot. Five of ten gates pass. No sampled trajectory, seed, formal run, GPU,
or HPC4 job was used.

The result is stronger than the earlier scalar-proxy audit because every cell
uses the exact vector temporal-difference drift, the complete matrix-valued
gradient lag covariance, the exact delayed Polyak--Ruppert readout, and a
marginal-preserving cross-agent trajectory coupling on FrozenLake-v1,
CliffWalking-v0, and Taxi-v3.

## Frozen primary results

| Quantity | Result | Gate |
|---|---:|---:|
| Full-cost oracle / strong fixed geometric ratio | 1.009880 | V3 fail |
| Full-cost oracle improvement | -0.987956% | threshold at least 5% |
| Strictly improved cells | 71/252 = 28.1746% | V4 fail |
| Nonconstant post-probe / cellwise fixed ratio | 1.172773 | V5 fail |
| Nonconstant improvement | -17.2773% | threshold at least 1% |
| Nonconstant strictly improved cells | 0/252 | V6 fail |
| Adjacent-correlation direction | 100% | V7 pass |
| Selected support | fixed q=1,4,16 only | V8 fail |

V1, V2, V7, V9, and V10 pass. V3, V4, V5, V6, and V8 fail. No threshold or
population was changed after execution.

## Mechanism diagnosis

The exact correlation direction is present: oracle mean participation is
nonincreasing in every adjacent-correlation comparison. Its magnitude is not
large enough. A descriptive decomposition of the frozen rows shows that,
even before charging probes, the cellwise fixed oracle improves the strong
fixed baseline by only 2.0754% geometrically and strictly in 34.13% of cells.
The taskwise improvements are 2.4994% on FrozenLake, 1.8165% on CliffWalking,
and 1.9087% on Taxi.

The complete no-probe schedule oracle equals the cellwise fixed oracle in all
252 cells. The best nonconstant schedule is 14.1667% worse geometrically and
never strictly improves a cell. Charging the frozen probe changes the already
small fixed-action ceiling from a 2.0754% gain to a 0.9880% loss.

Therefore this result does not show that the exact theory or implementation is
wrong. It shows that stationary homogeneous correlation, a fixed policy, and
the registered finite horizons do not create a useful time-varying schedule
phase. The fixed-(q) correlation phase exists but lies below the practical
adaptation threshold.

## Reproduction and provenance

The first completed run took about 110 seconds after computational Amendment
1. A clean run in a separate output directory took 116.6 seconds. The three
artifacts are byte-identical across runs:

- rows: `35b0fbe5bc542fd3dfcaacaf37655405862e5a977d244ea494651ee5bf75a1d1`;
- task constants: `daf8bd1f5a6d3ece2c163b3ee04d1ea69d93db0a83d874f305c7ebe006dc5a5a`;
- summary: `c1289bfe97e5bdbcb644d0eeff2c5d18e682fc7d3e4a172f97d10518bcd00284`.

The retained primary artifacts are under
`experiments/dependence_delay_linear/results/t049a_exact_schedule_static`.
The duplicate clean-rerun directory may be removed after this hash record is
committed.

## Next authorized step

The only authorized continuation is T-050 theory: determine whether fixed
participation is asymptotically optimal for stationary homogeneous
correlation, and derive an explicit break-even horizon for correlation
adaptation. A new positive controller experiment would require a genuinely
different theorem class, such as prospectively defined persistent
episode-level correlation regimes, and a new experiment identifier. T-049A
may not be rerun with altered tasks, schedules, costs, or gates.
