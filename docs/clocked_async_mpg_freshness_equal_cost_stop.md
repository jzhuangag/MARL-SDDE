# Equal-cost freshness falsification

## Decision

The freshness-only MPE path stops before seed expansion.  A six-agent,
batch-four smoke with a new development seed fails against never-refresh,
always-refresh, and an equal-cost always-extra-birth control.  No confirmation,
formal run, or GPU benchmark is authorized.

## Why this comparison is decisive for the current implementation

`always_extra_birth` purchases a second independent trajectory batch under the
packet-birth joint policy.  `always_refresh` purchases the same-size batch at
packet completion under a commit barrier.  Both acquisitions are fully charged
for actor transitions.  Their batch service durations are identical, although
the fresh acquisition serializes the centralized learner as required by the
exact timing contract.  Hence a fresh-barrier gain can no longer be explained
only by receiving twice as many Monte Carlo trajectories.

The runner now accepts any batch size of at least two, uses the unbiased trace
estimate of the covariance of the batch mean, and supplies always/fixed-period
birth-augmentation controls.  Fourteen targeted Python 3.9 tests and 119
package tests pass.

## Development-only result

All four modes use exactly 61,200 actor transitions.  This is a one-seed
falsification smoke, not an estimate of population performance.

| Mode | Final return |
|---|---:|
| LSFF | -250.8456 |
| never refresh | -243.5846 |
| always fresh barrier | -243.6555 |
| always extra birth | -245.4299 |

LSFF is below the three controls by 7.2610, 7.1901, and 5.4158 return units,
respectively.  Its mean certificate contains only 1.856% strategic-bias square;
the remainder is the inflated ordinary Monte Carlo variance term.  All four
methods also finish below their common initial return, so this simplified
frozen-baseline policy-gradient runner is not a strong training algorithm.

The raw summary is ignored under
`tmp/harl_freshness_equal_cost_n6_b4_smoke`; its SHA-256 is
`E40BA2BD479AD9C52D6DF86D846A942C385CFC4A945032CCD0E1BD5DC65FC9D2`.

## Scientific consequence

The conditional-risk theorem and its synthetic confirmation remain correct,
but the MPE evidence does not show that buying a current-policy measurement is
useful in a standard task.  Increasing seeds would estimate a mechanism that
has already failed its headroom screen.  Changing only the task, delay, or
learning rate to create a positive result would be outcome-driven benchmark
selection.

The broader asynchronous-MARL question can survive only by treating stale data
through a genuine bias--variance--freshness choice: reuse it, statistically
correct it, or pay to refresh it.  That new direction requires its own theorem
and outcome-free CPU headroom audit; it must not inherit a positive claim from
this stopped experiment.
