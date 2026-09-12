# EXP-009A validation: predictable mixing-certificate controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validation
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: exp009a_validation_v1

## Outcome

EXP-009A establishes predictable safety but rejects the current
near-oracle-efficiency claim.

- Registered policy runs: 9,216.
- Passed gates: 5/7.
- Failed gates: oracle competitiveness and robust-baseline improvement.

## What passed

1. The one-sided certificate covered the true persistence in 98.893% of 1,536
   online pilots, above the preregistered 98.5% threshold.
2. Every covered action was exactly mean-square stable; the largest exact
   radius was 0.98118.
3. Online-UCB had zero trajectory divergences.
4. The i.i.d.-naive policy was exactly unstable in 100% of \(p=0.98\) runs.
5. Participation responded in the correct direction: the overall median
   selected \(q\) was 6 under \(\rho=0\) and 2 under \(\rho=0.9\).

These results validate the predictability and safety architecture. The pilot
uses no exploitation samples, and every action is fixed from past data.

## What failed

The largest scenario-median online-to-oracle final-error ratio was 16.31,
above the registered limit of five. The two worst cells both combined
\(p=0.98\) and delay two. The high-persistence online median gap was about
70--72, compared with the oracle gap 47, leaving materially fewer updates
after the charged 2,048-transition pilot.

Online-UCB beat worst-mixing in only three of eight low/medium-persistence
scenarios. The failure is especially clear in zero-delay cells, where the
current controller's rate-optimal step contracts quickly but creates a larger
additive-noise floor.

## Diagnosed algorithmic error

The registered action selector minimized the homogeneous contraction
coefficient first and only then compared \(q\) using a residual term. It did
not jointly optimize \(\eta\) for the finite remaining update budget and
additive noise. Consequently it is a stability/rate controller, not yet the
finite-budget risk controller claimed in the theory program.

The correction is determined by the theorem, not by result tuning: minimize

\[
c_{\rm sharp}(\eta)^{\lfloor U/(2D+1)\rfloor}
+\frac{\eta^2\Omega_q}{1-c_{\rm sharp}(\eta)}
\]

over the entire proved-safe scalar interval for every \(q\). This preserves
the same confidence certificate and exact safety.

## Decision

- Retain the predictable UCB certificate and theorem-safe action construction.
- Withdraw any near-oracle or worst-baseline efficiency claim for EXP-009A.
- Preregister a finite-budget scalar-\(\eta\) optimization using the same
  pilots, cells, and baselines.
- Do not change pilot length or confidence level in that confirmation; doing
  so would confound the diagnosed step-size correction with estimator tuning.

## Artifacts

- `controller_runs.csv`
- `summary.json`
- `fig1_policy_error.png`
- `fig2_online_actions.png`
