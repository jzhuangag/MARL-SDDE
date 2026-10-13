# CPU development validation

## Material Passport

- Origin: academic-research-suite / experiment-agent.
- Date: 2026-10-13.
- Status: VERIFIED by a disjoint byte-identical replay.
- Scientific source/preregistration: `f584d76`.

## Decision

All eight frozen development gates and the independent replay requirement pass.
The separately reserved CPU confirmation seeds may now be frozen in an
independent commit. No nonlinear MARL or manuscript admission follows from this
controlled qualification.

## Population and results

All 360 contexts and 16 new calibration seeds are retained (46,080 method rows).
The full q={1,2,4,8} catalogue is present in every context. Known temporal
persistence is 0, 0.6, or 0.9; delay is 0 or 2. Both resource budgets include
calibration, and environment work means total actor transitions.

- Geometric terminal-MSE ratio versus the strongest unpaid single fixed q:
  0.825329, paired-seed 95% interval [0.824262, 0.826397].
- Ratio versus the paid true-context oracle: 1.003575 [1.002278, 1.004873].
- Ratio versus correlation-only selection: 0.866011 [0.865032, 0.866992].
- Interior-q qualified contexts: 106; correct q selection: 98.5259%.
- Joint parameter interval coverage: 98.9063%, against nominal 99% and the
  frozen empirical floor 98.5%. A nominal probability is not an exact observed
  frequency requirement.
- Every action risk interval and selection-regret bound is valid on the joint
  coverage event.
- All 189 Monte Carlo checkpoint checks pass; each uses 4,096 fresh learning
  replicas and the preregistered 4.5-standard-error diagnostic tolerance.

Fixed-q comparators have deliberately privileged true-context optimal steps
and spacings and pay no calibration cost. The primary advantage is averaged
over the prespecified population, not universal cellwise dominance.
The `myopic` method is a one-step decision followed by commitment; it is not a
continually replanning online scheduler. Its large gap is not promoted as a
strong-online-baseline claim.

## Exact replay

| File | SHA-256 |
|---|---|
| endpoints.csv | 801ae56bdd1ecf7b3c05cb2e429042e86fc55ab64a2aa303c1d03923c21eecfd |
| certificates.csv | de271e9c0bdab0cd856aad3d12f4ba7d4f465bc10467fe6f912ba36573348f64 |
| curves.csv | 9f2f2217bf2531fd74d135565f0b290c395873d990dc218ab8d34554e03b4a2e |
| monte_carlo.csv | e07a4cb71bd5576149e1c2acf1d4946429382b951b75e12b50003071fc5c9f0c |
| summary.json | ff0f71d49e2faeaa86c2bfc9d3c70ad5f24556e6d2c2429f36770ba3757ea4d0 |

The endpoint/certificate tables and replay are retained locally; compact
summaries, curves, hashes, and this decision are versioned.

## Statistical fallacy audit

All 11 experiment-agent categories were checked. Grouped context results are
retained; population averages are not attributed to every individual context.
No outcome-based cell exclusion, extreme-seed selection, attrition, hidden
retry, or retrospective threshold change occurred. Paired inference uses seed
aggregates rather than treating 46,080 rows as independent replicates.
Calibration variation is distinct from Monte Carlo training variation. The
Gaussian process is a controlled model, not evidence of nonlinear MARL task
generalization. Diagnosis/survival-specific base-rate fallacies are inapplicable.

## Remaining experiment requirements

Actual online adaptive baselines and a measured response model for the actual
MARL optimizer remain to be qualified. The controlled conditional-MSE curves
must not be relabeled as multi-agent team-return curves.
