# Stochastic clocked-MPG confirmation validation

## Decision

**Pass S1--S12.**  The pathwise-certified single-flight learner retains a large,
fully charged wall-clock advantage over a fully utilized frozen-policy barrier
under heterogeneous service, using sampled fixed-horizon Markov REINFORCE
packets.  This authorizes design of the standard nonlinear MARL GPU stage.  It
does not make the project ICML-ready and does not authorize a universal
dominance claim.

## Provenance and correction history

- stationary-mark preregistration: `40cf489dd4e034ea0642739c64ae50f28ecf1df0`;
- pathwise Amendment 1: `57ce613797148fffc475d2c3791214218d2dff0d`;
- version-2 config SHA-256:
  `36915d80d62344e154181ceef1037753d957131231a35e4d3212a5e89fc5b589`;
- source hashes recorded in the result:
  - drift: `80c9a3b1a6563d4278a1a2c46f8f6bd1aad9b7d79413ce44d23426ca6259f5c3`;
  - runner: `7446e644a6991a5eceb1099bf2ad1153826321da8d0f1cc8713395354494420f`;
  - simulator: `e109574207de0b99d35a39ea98c57bcf8a50836600be6a1f4b16c17a6d0c8471`.

Version 1 was interrupted before any result directory or outcome was written
after an outcome-free audit found that fixed event probabilities did not match
the bounded-renewal simulator's conditional completion marks.  Its touched
namespace was excluded.  Version 2 uses a max-over-arriving-block pathwise
history certificate, `delta=1`, a new namespace and unchanged scientific grid.

## Frozen population

- 64 fresh seeds;
- 16 coupling-by-service cells;
- six policies;
- 6,144 endpoint rows;
- packet horizon 16 and batch size 16;
- 256 joint Markov transitions per completed packet;
- couplings `{0,.08,.16,.24}` and service ratios `{1,2,4,8}`;
- service ratios `{2,4,8}` form the 12-cell heterogeneous primary population;
- target normalized potential gap `.3`, terminal time 180.

All endpoint values were finite.  Every policy reached the target in every
seed-cell.  Every asynchronous realized event delay stayed below its registered
bound.  Completed, terminal-partial and barrier-cancelled transition work passed
the frozen accounting identities.

## Primary results

Over the 12 heterogeneous cells, comparing pathwise-certified asynchronous
learning with the fully utilized barrier:

| Metric | Registered result |
|---|---:|
| geometric cell-median wall-clock ratio | **0.4313525930** |
| geometric cell-median transition-work ratio | **0.4296873191** |
| cells with lower median wall-clock | **12/12** |
| paired target coverage | **1.000 in every cell** |
| terminal-gap geometric ratio | **0.9692623241** |

The wall-clock and transition-work gains track each other.  The result is not
obtained by hiding partial rollouts: unfinished work at the terminal horizon
and work cancelled at a barrier are both charged.  The work gain instead comes
from fresh adaptive query depth; the barrier can batch all fast packets that
finish while waiting, but those packets were born at the same frozen round
policy.

## Registered rate--coupling phase

Cell-median certified/barrier wall-clock ratios are:

| coupling \ service ratio | 1 | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.968 | 0.498 | 0.255 | 0.128 |
| 0.08 | 1.001 | 0.660 | 0.393 | 0.271 |
| 0.16 | 1.241 | 0.766 | 0.504 | 0.389 |
| 0.24 | 1.450 | 0.878 | 0.596 | 0.463 |

For every coupling, more service heterogeneity strengthens the barrier-free
advantage.  For every service ratio, stronger policy interaction weakens it.
This is the coherent empirical phase predicted by the pathwise history term:
heterogeneous useful compute pulls toward asynchrony, while teammate-policy
staleness pulls toward a barrier.

Homogeneous service is not a positive primary claim.  At coupling `.16` and
`.24`, the certified method is slower than the barrier when the service ratio
is one.  This is retained as the negative side of the phase rather than removed
from the population.

## Strong raw-asynchronous reference

Raw common-step asynchronous learning remains faster.  The pathwise
certificate/raw ratios over the heterogeneous primary population are:

- wall-clock: `1.5857576502`;
- transition work: `1.5839871601`.

They pass the preregistered certificate-cost ceiling `2.0`, but this gate is a
cost bound, not a superiority test.  At coupling `.24`, service ratio `8`, the
cell-median time cost reaches `3.26117`.  A paper must show this full frontier:
raw async is an empirical speed reference, certified async has a theorem, and
the barrier is fresh but straggler-limited.

Terminal behavior is likewise phase-dependent.  Six high-coupling primary
cells have terminal-gap ratios above one even though all 12 reach the target
earlier.  The aggregate ratio is favorable, but the evidence does not support
uniform terminal no-harm.  A decaying-step standard-MARL implementation must
separate time-to-quality from final-return claims.

## Frozen gate ledger

| Gate | Result |
|---|---:|
| S1 schema, uniqueness, finite endpoints | pass |
| S2 per-primary-cell paired coverage >= .95 | pass |
| S3 primary time ratio <= .75 | pass (`.43135`) |
| S4 primary charged-work ratio <= .75 | pass (`.42969`) |
| S5 at least 10/12 primary cells faster | pass (`12/12`) |
| S6 terminal-gap geometric ratio <= 1.05 | pass (`.96926`) |
| S7 certificate time/work cost <= 2.0 | pass (`1.58576/1.58399`) |
| S8 complete transition accounting | pass |
| S9 registered event delay | pass |
| S10 namespace disjointness | pass |
| S11 every-policy target coverage >= .95 | pass (`1.0`) |
| S12 byte-exact reproduction | pass |

## Reproduction and tests

Primary ignored artifact directory:

`experiments/clocked_async_mpg/results/stochastic_confirmation_v2_20260901`

Reproduction directory:

`experiments/clocked_async_mpg/results/stochastic_confirmation_v2_20260901_reproduction`

Byte-identical hashes:

- `endpoints.jsonl`:
  `1f5710591661eb057fa2c7a78818f8710c7973e85bbaeae7a5dbb1cde831e185`;
- `summary.json`:
  `128a8760834dd21c2dda0e0a307c2716fa5ebd1847c2035ab6debc1d197e5127`.

The first run completed from 14:42:28 to 14:56:39.  The identical reproduction
completed from 14:56:53 to 16:10:53 under sustained local CPU throttling.  The
runtime difference is an execution-environment observation; scientific files
are byte-identical and contain no runtime field.

The final full repository regression passed:

`985 passed, 7 skipped in 645.50 s`.

An additional outcome-free deterministic check evaluated the windowed full-
gradient inequality on 120 random coupled quadratics; the maximum left/right
ratio was `0.15925455`.  This checks algebra, not efficacy.

## What this establishes and what remains

Established for one executable low-complexity mechanism:

1. self-fresh single-flight packets pay only cross-agent policy staleness;
2. a pathwise Lyapunov--Krasovskii step exists for arbitrary bounded completion
   order, Markov packet bias and centered noise;
3. sampled Markov packets preserve a large positive time/work phase versus a
   strong fully utilized barrier;
4. the predicted rate--coupling directions hold across the full grid.

Still required for an ICML submission:

1. integrate the pathwise window lemma, wall-clock conversion and conditional
   Nash conversion into a publication-grade appendix;
2. prove or sharply delimit the MPG interiority/distribution-mismatch constant;
3. give a matching separation/lower family, not only an upper bound;
4. validate on standard nonlinear MARL tasks against MAPPO/HAPPO-family,
   barrier, raw asynchronous and lag-corrected asynchronous baselines;
5. report real GPU utilization, policy lag, transitions, time-to-return and
   final return with new preregistered seeds.

The next empirical stage requires GPU/HPC4.  It must not begin until its task
families, delay injection, baselines, metrics and seeds are separately frozen.
