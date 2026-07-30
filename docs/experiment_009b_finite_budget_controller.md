# EXP-009B preregistration: finite-budget safe controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp009b_preregistration_v1

## Single registered correction

EXP-009B changes only the scalar step selection diagnosed in EXP-009A. All 12
scenarios, 128 seeds, pilot length, confidence level, resource costs,
participation candidates, additive-noise model, and six policies remain
unchanged.

For each candidate \(q\), minimize over the proved sharp safe interval

\[
\widehat R(q,\eta)
=c_{\rm sharp}(q,\eta)^{\lfloor U(q)/(2D+1)\rfloor}
+\frac{\eta^2\Omega_q}{1-c_{\rm sharp}(q,\eta)}.
\]

The solve is one dimensional and deterministic. The controller selects the
minimizing \((q,\eta)\), with ties going first to smaller \(q\), then smaller
\(\eta\). Every baseline uses the same finite-budget scalar optimization under
its own registered information set. No pilot, confidence, or model parameter
is retuned.

## Endpoints

EXP-009A endpoints are retained. Efficiency gates use exact expected final
squared error conditional on the selected action; Monte Carlo final error is
reported as a reproduction check rather than used for the primary ratio.

## Preregistered gates

1. **Certificate coverage:** at least 98.5%.
2. **Conditional exact safety:** every covered online action has exact radius
   below one.
3. **Online trajectory safety:** zero divergences.
4. **Naive failure detectable:** i.i.d.-naive is exactly unstable in at least
   90% of \(p=0.98\) runs.
5. **Expected oracle competitiveness:** in every scenario, the median paired
   online-to-oracle exact expected-error ratio is at most five.
6. **Expected robust-baseline improvement:** for \(p\le0.9\), online-UCB has
   lower median exact expected error than worst-mixing in at least six of eight
   scenarios.
7. **Participation response:** median online \(q\) under \(\rho=0.9\) is no
   larger than under \(\rho=0\).
8. **Correction efficacy:** EXP-009B lowers the largest scenario-median
   online-to-oracle expected-error ratio relative to the frozen EXP-009A action
   rule.

Safety failures invalidate the controller. Efficiency failures retain a safe
method but reject the near-oracle claim.
