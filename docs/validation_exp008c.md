# EXP-008C validation: locally expanding Markov TD

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validation
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: exp008c_validation_v1

## Outcome

EXP-008C is numerically valid and isolates the temporal mechanism that was
inactive in EXP-008B.

- Numerical checks: 4/4 passed.
- Scientific gates: 5/7 passed.
- Registered exact cells: 72/72 completed.
- Conditional means: \(-0.545\) and \(1.855\).
- Stationary mean: \(0.655>0\).

The two failures concern the frozen unit-constant gap rule. They are retained
and prevent that rule from being stated as a theorem.

## Numerical validity

The independently enumerated conditional operator and moment-form operator
agreed to maximum absolute error \(2.78\times10^{-17}\). All exact Markov,
i.i.d., and mode-conditioned first-moment boundaries were valid; the maximum
eigen-residual was \(5.21\times10^{-15}\). The \(p=0.5\) i.i.d. reduction and
the \(q=1\) correlation invariance checks were exact.

## Confirmed mechanisms

### Temporal persistence is an essential stability variable

At \(p=0.98\), all 24 registered cells had an exact Markov boundary at most
half the i.i.d. boundary. The observed Markov-to-i.i.d. ratios were
0.01925--0.03956. Hence matching the same-time sample law is insufficient:
temporal ordering changes the admissible step by roughly 25--52 times.

The uninflated i.i.d. rule was unstable in all 24 high-persistence cells; its
largest exact Markov spectral radius was 1.68188.

### Participation is correlation limited

Conditional sharing never improved the exact boundary in any registered
\(q\ge2\) cell. Under \(\rho=0.9\), the largest \(q=32\)-to-\(q=16\)
boundary gain was only 1.00078, and it was always no greater than the
corresponding independent gain. Thus agent count is mechanistic but saturates:
additional agents cannot average away a persistent common regime.

### Mean stability is not mean-square stability

All 72 exact mean-square boundaries were at most 80% of their exact
mode-conditioned first-moment boundaries. The observed ratios were
0.1734--0.4947. A proof based only on mean delayed dynamics would therefore
miss the controlling instability.

## The frozen gap rule: close but not safe

The spectral-gap inflation

\[
\chi_{\rm gap}=p/(1-p)
\]

captured the correct scale. The resulting step was 65.8%--105.6% of the exact
boundary in all cells. Equivalently, the exact inflation required by the
parallel-sum formula was 0.636--1.056 times \(\chi_{\rm gap}\).

However, the rule exceeded the exact boundary in some cells; its largest
spectral radius was 1.000535. Therefore:

1. the unit constant is rejected as a safety theorem;
2. the good scaling supports a spectral-gap/integrated-correlation mainline;
3. a proof-derived safety factor must be fixed before the next validation;
4. exact boundary fitting cannot be used to choose that factor.

## Mainline implication

The combined EXP-008B/C evidence supports a conditional, not universal,
claim:

- Markov persistence need not matter when every mode is benign;
- it becomes decisive when stationary monotonicity hides a persistent locally
  expanding mode;
- same-time \(K(q,\rho)\), delay, and temporal mixing are distinct inverse-step
  margins;
- participation gains saturate under a shared persistent state even when
  more agents are available.

This is the correct basis for a paper about correlation- and state-adaptive
participation. A generic claim that “Markov correlation always shrinks the
boundary” would be false.

## Next action

Derive a conservative mixing inequality before freezing the next rule. The
next experiment must test that proof-derived constant and a predictable
low-memory estimator; it must not retune the factor against EXP-008C.

## Artifacts

- `expanding_markov_boundaries.csv`
- `direct_validation.json`
- `summary.json`
- `fig1_exact_boundary_scaling.png`
- `fig2_rule_tightness.png`
- `fig3_required_inflation.png`
