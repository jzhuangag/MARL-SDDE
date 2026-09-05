# EXP-009C validation: joint \((q,b,\eta)\) controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validation
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: exp009c_validation_v1

## Outcome

Joint safe search over participation, decorrelation gap, and step improves the
static-pilot controller but still fails the worst-case near-oracle gate.

- Registered exact expected-risk rows: 9,216.
- Passed gates: 6/7.
- Certificate coverage: 98.893%.
- Covered exact actions stable: 1,519/1,519.

## Improvements

The largest scenario-median online-to-oracle ratio fell from EXP-009B's
11.6347 to 10.4641. Online beat worst-mixing in all eight low/medium
persistence scenarios. The selected gap was genuinely active: the 12 scenario
medians contained seven distinct values. Participation remained strongly
correlation responsive, with overall median \(q=18\) for \(\rho=0\) and
\(q=1\) for \(\rho=0.9\).

## Near-oracle failure

The registered maximum ratio of five was not met. At \(p=0.98,D=2\), the two
median ratios were 10.46 and 8.65. Typical actions expose the cause:

- online gaps: 125--136;
- oracle gaps: 85--93;
- delayed online updates are therefore too few to remove the initial-error
  term, even though the selected steps are safe.

This is not repaired by optimizing \(q\), \(b\), or \(\eta\) against the same
one-shot upper confidence bound. With only about 41 expected switches in a
2,048-transition \(p=0.98\) pilot, a 99% upper bound necessarily has
substantial relative uncertainty in \(1-p\), and decorrelation time is
proportional to its reciprocal.

## Mainline decision

The static-pilot method supports the following claims:

1. predictable high-confidence safety;
2. exact adaptation of \(q,b,\eta\);
3. correlation-limited participation;
4. improvement over a worst-mixing baseline outside the most persistent
   regime.

It does not support a uniform near-oracle claim.

The next estimator should be progressive: start conservatively, update an
anytime upper confidence sequence using raw transitions observed between
updates, and use only past counts for the next block. Such a controller can
reuse exploitation-time state transitions instead of spending the entire
information budget in a one-shot pilot. A time-uniform coverage proof is
required before this extension is called safe.

## Artifacts

- `joint_qbe_runs.csv`
- `summary.json`
- `fig1_joint_policy_error.png`
- `fig2_joint_actions.png`
