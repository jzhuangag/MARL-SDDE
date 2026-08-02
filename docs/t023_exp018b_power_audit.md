# T-023 prospective EXP-018B power audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + validate
- Origin Date: 2026-08-02
- Verification Status: ANALYZED
- Evidence: pilot-informed design audit, not formal evidence

T-023 uses the 64 EXP-018A pilot seeds only to assess whether a separate
192-seed formal experiment is computationally and statistically plausible. It
does not reverse EXP-018A's 5/7 failure.

The prospective primary endpoints are the aggregate median and p90 relative
errors between the empirical random-projection variance ratio and
`rho+(1-rho)/q`. Variances are first averaged over the 16 frozen projections
inside each task/mixing/checkpoint/rho/q cell. Two one-sided 97.5% bootstrap
upper bounds provide Bonferroni familywise alpha 0.05.

Using 2,000 frozen bootstrap resamples of size 192, the projected upper bounds
are 0.08818 for the median endpoint and 0.41008 for the p90 endpoint, below the
pre-existing tolerances 0.20 and 0.50. The static feasibility gate therefore
passes.

This calculation is design information, not proof that a formal experiment
will pass. A new preregistration must use fresh seeds, make q=1 a single
cross-rho common-random-number baseline, make the analyzer path-independent,
retain all q/rho cells for calibration, and omit strict-order requirements for
theoretically sub-5% adjacent contrasts.
