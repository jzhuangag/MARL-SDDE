## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free interface validation
- Origin Date: 2026-09-07
- Verification Status: 6/7 GATES PASS; NUMERIC RESET EVALUATOR FAILS
- Version Label: pursuit_cache_semantics_qualification_v1_validation

# Pursuit cache semantics v1 validation

The frozen outcome-free audit is a reproducible `6/7` failure. Primary and
isolated reproduction JSON are byte-identical with SHA-256
`4747c502b88ff12cf86695bd512802ab1887f7a3cee007f1e0b6a5371a59a35e`.

The audit executed 226 non-null refreshes and charged 2,217,512 policy bytes.
All 226 launch-time refreshes persisted after a zero-weight receipt. Pursuit
changed its candidate support 244 times, including 90 strictly positive active
edge cache-energy jumps. The topology-motion identity had maximum error
`1.7763568394002505e-15`.

Gate C3 failed: the maximum error from subtracting two full 56-edge cache
energies and comparing the difference with one local reset was
`3.4949135852002655e-07`, above the frozen absolute `1e-10` threshold. The
operation is algebraically correct but numerically ill-conditioned because it
subtracts two large, nearly equal totals. The failure is retained and the
threshold is not changed.

The permitted correction is an evaluator change: compute the selected edge's
before/after energy difference directly, while retaining the full-energy
subtraction as a descriptive cancellation diagnostic. That correction requires
a new frozen source hash and untouched seeds. No efficacy outcome, reward, GPU,
or HPC4 resource is involved.
