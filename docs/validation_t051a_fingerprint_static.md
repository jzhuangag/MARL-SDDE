# T-051A fingerprint static gate: validation

## Decision

T-051A is an honest 11/12 gate failure. It does not authorize a sampled CPU
pilot, formal seeds, GPU, or HPC4. No gate, task, block count, or result was
rounded or changed.

The main mechanism results are strong:

- aggregate full-cost leading improvement: 12.0161%;
- overhead 8 improvement: 12.7837%;
- overhead 32 improvement: 11.2418%;
- expected improvement in 63/81 oracle-active cells: 77.7778%;
- every collision, contraction, direction, dual-budget, taint, and finite
  result check passes.

The single failure is S7. The frozen maximum no-harm ratio is 1.05, while the
largest Hoeffding upper bound is 1.0501063679. The worst cell is FrozenLake,
delay 8, overhead 32, and correlation 0.3. Its oracle and strong baseline are
both (q=16); the excess over the gate is 0.0001063679 in ratio units. This
value is not rounded down.

## Probe lengths

The exact shortest state fingerprints attaining independent collision at
most 0.01 are:

| Task | Transitions | Exact collision probability |
|---|---:|---:|
| FrozenLake 8x8 | 2 | 0.00519779 |
| CliffWalking | 12 | 0.00956779 |
| Taxi | 0 | 0.00815347 |

All 96 two-agent probe blocks, their hash messages, their environment
transitions, the learning messages, learning transitions, and delay are
charged separately.

## Reproduction

The result has SHA-256
`d57f7a3f0c6d70e099bd4f5f79d18bd434c18f0670a3143102e8017da80090f6`.
A clean second execution was byte-identical, after which the duplicate output
directory was removed. No sampled trajectory or seed was generated.

## Scientific interpretation and next admissible step

The 12% aggregate certificate shows that the stationary fixed-participation
route has practical standard-task value. S7 fails because T-051 used a
generic Hoeffding tail for a bounded block statistic. With (q_p=2), each
block match is in fact an exactly Bernoulli observation with known success
probability (c_L+(1-c_L)\rho). A new T-052 theory audit may derive and test
the exact binomial action-selection risk. That is a sharper theorem, not a
retrospective change to T-051A. T-051A remains permanently failed and cannot
be used as sampled or formal evidence.
