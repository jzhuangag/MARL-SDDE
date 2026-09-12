# EXP-009B validation: finite-budget safe controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validation
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: exp009b_validation_v1

## Outcome

Joint finite-budget optimization of \(\eta\) fixes the robust-baseline failure
but does not fix high-persistence oracle competitiveness.

- Registered runs: 9,216.
- Passed gates: 5/8.
- Safety gates: all passed.
- Efficiency gates: robust-baseline improvement passed; oracle
  competitiveness and correction efficacy failed.

## Confirmed improvement

Online-UCB beat worst-mixing in all eight \(p\le0.9\) scenarios, compared with
three of eight in EXP-009A. The exact expected-error medians show the intended
bias--variance correction: the controller no longer uses the homogeneous
rate-optimal step when a smaller step lowers the finite-budget noise floor.

The controller remained predictable and safe:

- certificate coverage: 98.893%;
- every covered exact radius below one;
- zero trajectory divergences;
- median \(q\): 6 under \(\rho=0\), 1 under \(\rho=0.9\).

## Remaining failure

The largest scenario-median exact expected-error ratio to oracle was 11.63.
The two \(p=0.98,D=2\) cells had ratios 11.63 and 9.85. EXP-009B's largest
ratio was slightly worse than the frozen EXP-009A expected-error reference
11.39, so the correction-efficacy gate also failed.

The i.i.d.-naive instability rate fell from 100% to 75%, below its registered
90% gate. This is not a safety defect in the proposed method: finite-budget
risk optimization selected a smaller naive step that happened to stabilize
some delayed cells. It does mean “ignoring mixing always diverges” is too broad
a claim; the defensible claim is that it can force a severe stability or
efficiency loss.

## Root cause and next correction

Both EXP-009A and EXP-009B fix the decorrelation error at
\(\delta=\mu/(4L)\) before optimizing \(q,\eta\). At high persistence, the
online upper certificate selects a median gap around 72, whereas the oracle
uses 47. Since gap directly reduces the number of updates, no scalar step can
recover the lost budget.

Theorem 3 permits every
\(0\le\delta(b)<\mu/(2L)\). Therefore the next controller should jointly
search safe integer \(b\), participation \(q\), and scalar \(\eta\), using the
same finite-budget risk and confidence bound. This is the algorithm originally
suggested by the theorem and is not a relaxation of safety.

## Decision

Retain the finite-budget scalar-\(\eta\) solver and predictable certificate.
Do not claim near-oracle performance yet. Run one final exact expected-risk
audit of joint \((q,b,\eta)\) selection before moving to larger linear-TD
benchmarks.

## Artifacts

- `finite_budget_runs.csv`
- `summary.json`
- `fig1_expected_policy_error.png`
- `fig2_online_actions.png`
