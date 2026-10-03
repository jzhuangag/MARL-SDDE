## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-blind estimator and complexity validation
- Origin Date: 2026-09-07
- Verification Status: VERIFIED AGAINST FROZEN COMMIT AND HPC4 ARTIFACTS
- Version Label: pistonball_exact_score_audit_result_v1

# Exact sparse Pistonball score audit: validation and stop decision

## Decision

The frozen audit fails one of six mandatory gates. The Pistonball policy-cache
mainline is stopped. No exact-score headroom matrix, pilot, formal seeds, or
performance claim is authorized, and the failed degree gate must not be
changed retrospectively.

This is a structural benchmark mismatch rather than another zero-score
failure. Exact counterfactual learning values, cache reset energy, and queue
prices were all active, and runtime was acceptable. However, the registered
Pistonball causal cone did not have the assumed local degree: observed reverse
evaluations ranged from 9 to 18, versus the frozen maximum of 8. Because the
exact scorer uses one current gradient, one null gradient, and one gradient
for every eligible edge, this corresponds to 7--16 eligible donors rather than
at most six. The standard task's effective policy dependence is too broad for
the claimed low-degree implementation.

## Frozen provenance

- Source commit: `8962d1a1f5784820d9b69b22bdbfd081bc16c4f8`
- Branch: `codex/joint-ms-exp007c`
- Slurm array: `1825809_0` (exact) and `1825809_1` (no refresh)
- Resources per task: one A30, eight CPUs, 16 GB RAM
- Seed: `79008`; development-only and excluded from all future pilot/formal
  populations
- Artifact root:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/artifacts/exact-score-8962d1a1f5784820d9b69b22bdbfd081bc16c4f8`
- Log root:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/logs/exact-score-8962d1a1f5784820d9b69b22bdbfd081bc16c4f8`
- No `/project` output was written.

Both tasks completed with exit code `0:0`; all four stdout/stderr files are
empty. The exact and no-refresh raw JSON files and analyzed summary pass their
stored SHA-256 manifest. Raw hashes are:

- exact: `c8401e5af6c6b60c26f6d90f73884db6b1e0954e7f174a7d1d7c316d1260157f`
- no refresh: `d577d211f8087d9188b961f5d82176fd6369b4b47b6acf9220cc4248bb2321f3`
- summary: `e8a05835bb7864dbeb91e8cba58045e5b8abaa08245d3c326753a6a3f884ef93`

## Gate ledger

| Gate | Frozen requirement | Result | Status |
|---|---:|---:|---|
| E1 signed component | nonzero | nonzero | pass |
| E2 cache component | nonzero | nonzero | pass |
| E3 signed favorable | at least 1% | 48.8706% | pass |
| E4 queue active | positive fraction | 45.9959% | pass |
| E5 sparse reverse bound | maximum at most 8 | maximum 18 | **fail** |
| E6 runtime | exact/control at most 3 | 1.37542 | pass |

There were 487 scored launches. Reverse evaluations had minimum 9, median 10,
p90 13, and maximum 18. The exact controller selected an edge in 53.5934% of
scored launches and spent 261 of 512 permitted refresh units. The scale-only
calibration returned a clipped learning weight of `1e8` and cache-debt weight
`88009.15148128525`; these values are not authorized for a performance run.

Returns were deliberately excluded from all gates. For completeness, both
runs stayed at `-13.4831460674157` over this short 1,024-launch audit. This is
neither evidence for nor against efficacy because the audit used unit score
weights and was not powered or registered for learning performance.

## Consequence for the unified claim

The exact estimator repairs two genuine implementation problems: it is
nonzero under the smooth critic and uses only replay completed before launch.
It also removes the finite-displacement Taylor approximation at modest measured
runtime. What it does not repair is the sparse-interaction premise on standard
Pistonball. Continuing by pruning donors, relaxing the reverse-call gate, or
using these outcomes to choose a different cone would create a new method and
invalidate the outcome-free stop rule.

Accordingly, the current collection supports a conditional Lyapunov scheduling
theory and several audited negative boundaries, but it does not support an
ICML-level positive standard-benchmark paper. The next research decision must
be made at the problem/formulation level rather than by another Pistonball
estimator patch.
