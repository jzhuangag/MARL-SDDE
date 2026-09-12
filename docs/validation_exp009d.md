# EXP-009D validation: progressive anytime-safe controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validation
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: exp009d_validation_v1

## Outcome

Progressive anytime certification improves high-persistence efficiency and
retains exact safety, but it does not establish a uniform near-oracle result.

- Scenario/seed metrics: 1,536.
- Predictable action rows: 15,360.
- Passed gates: 5/7.
- Failed gates: near-oracle expected risk and worst-baseline improvement.

## Safety and refinement

The time-uniform Clopper--Pearson sequence covered the true persistence at all
decision times in 99.479% of seeds, above the preregistered 98.5% threshold.
Every action on a fully covered seed was exactly stable; the largest exact
radius was 0.993797.

At \(p=0.98\), median gaps decreased materially:

- first updating block: 174--207;
- final block: 112--128.

Thus exploitation-time transitions do refine the mixing certificate without
violating predictability. Median participation remained correlation
responsive: 18 under \(\rho=0\) and 1 under \(\rho=0.9\).

## Efficiency boundary

The worst scenario-median online/oracle ratio improved from static EXP-009C's
10.464 to 7.571, so progressive reuse is valuable. It still exceeds the
registered threshold five. Progressive online beat worst-mixing in only three
of 12 scenarios.

The worst-mixing comparison exposes an additional state issue: the block
selector repeatedly optimizes a risk expression whose initial-error term is
normalized to one. Once the actual error is small, this favors excessive
contraction and a larger additive-noise floor. A practical state-adaptive
controller should carry a scalar Lyapunov upper surrogate between blocks.

The high-\(p\), delayed oracle gap is more fundamental. With simultaneous 99%
coverage and finite transitions, uncertainty in \(1-p\) produces a large
relative uncertainty in decorrelation time. The experiments do not support a
uniform constant-factor oracle claim as \(p\to1\).

## Mainline decision

Promote predictable anytime safety, joint \(q,b,\eta\) adaptation,
correlation-responsive participation, and progressive certificate refinement.

Reject uniform near-oracle efficiency for all mixing rates and any claim that
the progressive method always beats worst-case tuning. The theorem should
contain an explicit confidence-uncertainty penalty that worsens with the
inverse spectral gap.

## Artifacts

- `progressive_metrics.csv`
- `progressive_actions.csv`
- `summary.json`
- `fig1_progressive_ratios.png`
- `fig2_gap_refinement.png`
