# Validation: coupled actor--critic certificate nonvacuity

Date: 2026-09-05.

Preregistration commit: `cc24102`.

Decision: **STOP the per-packet all-schedule high-probability shield.**  This
does not invalidate the conditional finite-time theorem.  It shows that its
current distribution-free implementation is too expensive to serve as the
paper's practical online controller.

## Frozen audit outcome

The exact static audit contains 576 owner cases and matches frozen scenario
hash `5c15424c8461ef023a78d3e479bd4931104e2f2493d375050fc966a512d912a9`.
It deliberately sets every coordinate sample variance to zero, removes all
version staleness, and gives the actor the exact critic.  Therefore the only
statistical cost is the unavoidable bounded-range term of the observable
empirical-Bernstein certificate.

| Quantity | Result |
|---|---:|
| Jointly nonvacuous within 8,192 charged transitions | 20.8333% |
| Jointly nonvacuous within 16,384 charged transitions | 30.5556% |
| Median minimum charged transitions | 65,536 |
| `H>=4` jointly nonvacuous within 16,384 | 14.5833% |
| Median actor minimum charged transitions | 65,536 |
| Median critic minimum charged transitions | 16,384 |

By horizon, the 8,192-transition fraction is 62.5% at `H=2` and exactly zero
at `H=4,8`.  Median required transitions are 8,192, 65,536 and 131,072 for
`H=2,4,8`, respectively.  Increasing discount worsens the range term, but the
horizon effect is already decisive.

## Gate ledger

| Gate | Result |
|---|---:|
| N1 frozen population/hash | pass |
| N2 finite values and positive critic curvature | pass |
| N3 at least 50% practical nonvacuity | **fail** (20.8333%) |
| N4 median at most 16,384 transitions | **fail** (65,536) |
| N5 at least 25% long-horizon extended nonvacuity | **fail** (14.5833%) |
| N6 optimistic assumptions preserved | pass |
| N7 byte-exact reproduction | pass |

Primary and reproduction artifacts are byte-identical:

- `cells.json`: `5312BDF41C4B849EFD9D0A11C9CF8BE25CCFA523988A49EBFA53CC937EFCEC87`;
- `summary.json`: `F2BA6572DB3D5D848866717BF907F144BEADDFB5ED1467997D460EF7BDE2AF01`.

No stochastic trajectory, return, formal seed, GPU job or HPC4 resource was
used.  The ignored local artifacts remain under
`experiments/clocked_async_mpg/results/coupled_actor_critic_certificate_nonvacuity/`.

## Scientific consequence

Realized sample variance, critic bias and version staleness can only make the
tested radius larger.  It would therefore be misleading to proceed with this
shield and report that it is "safe but conservative."  Its required data per
decision is incompatible with the intended low-complexity standard-RL method.

The broader coupled-timescale question remains viable because the exact-moment
headroom scan showed 26.7% gain over best fixed timescales and 5.19% over the
online diagonal rule.  The next admissible design is an expectation-level,
predictable sample-split Lyapunov controller: a small independent sensor batch
chooses `(alpha,beta)`, and an independent fully charged update batch prevents
selection bias.  It must prove expected convergence and beat equal-cost
sample-split baselines before any standard MARL benchmark is authorized.
