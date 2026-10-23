# EXP-016B independent formal validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-01
- Verification Status: VERIFIED
- Version Label: exp016b_formal_validation_v1

## Verdict

EXP-016B formal passes all preregistered P1--P12 gates.  This is formal
synthetic and affine delayed Markov-TD evidence for the finite learning-value
threshold: when identification is feasible but its probe and delay cost cannot
yet be amortized, learning-aware fallback improves terminal learning risk over
information-only probing.  The result does not by itself establish a general
nonlinear MARL claim.

The implementation and inputs were frozen in commit `cc4877a` before formal
outcomes.  The run used 192 new paired CRN seeds, 2,752,512 rows, local CPU,
Python 3.11.13, and no GPU, HPC4, or `/project` storage.

## Registered findings

| Population | Paired mean difference | Simultaneous lower bound | Relative improvement | Result |
|---|---:|---:|---:|---|
| Layer A primary | 0.1767168 | 0.1611933 | 54.4267% | pass |
| Layer B affine TD | 0.0726347 | 0.0576749 | 10.8523% | pass |
| Delay-active | 0.1645572 | 0.1462769 | 49.1212% | pass |
| Message-binding | 0.2095151 | 0.1798575 | 77.6872% | pass |
| Environment-binding | 0.1464321 | 0.1213675 | 43.1302% | pass |

Scenario-level directional and practical coverage is `77/96 = 0.8020833`,
above the frozen `0.60` threshold.  All 17,280 neutral-Z seed rows were retained
outside the primary estimand.  All 56 registered Layer-B tasks were retained.
All 96 theorem-facing safety scenarios satisfy `S_mean <= epsilon_safe`; the
minimum slack is `6.5392e-06`.

## Gate and reproducibility audit

P1--P11 pass under the unchanged analyzer.  The analyzer deliberately leaves
P12 false before an external reproduction comparison.  A second complete run
then produced byte-identical raw metrics, cell summary, scenario summary, and
core-results JSON.  The independent comparison is recorded in
`exp016b_formal_reproduction_audit.json`; therefore P12 passes and the final
registered decision is 12/12 PASS.

- Raw metrics SHA-256: `68d6b7de757b1fc2159e0a12f909a154f23e2605ddae033bd5e6b1a582d8bd6e`
- Cell summary SHA-256: `08a72e82075a4896ab41635f9b01341850ed8c6715448ed652bb33cd53026076`
- Scenario summary SHA-256: `a052ecd55002178e375ca40b9d2d19717a346d48ead9a220d6ae6c7a8793473a`
- Core results SHA-256: `4f8f9d7c48e96c3bf1d2095bf0150686fcc249955da75ca94e412fb86a166baa`

## Statistical fallacy scan

Coverage: 11/11.  There is no aggregate/subset sign reversal; inference remains
at the registered scenario-family level; scenario and seed selection preceded
formal outcomes; no outcome-conditioned inclusion or adjustment is used; no
deployment prevalence is inferred; CRN blocks were not selected by extremes;
all scenarios, tasks, policies, seeds, neutral cells, and censored descriptive
cells were retained; the family, thresholds, and direction were frozen; the
formal implementation preceded outcome generation; the randomized simulation
contrast is not generalized causally to uncontrolled deployments; and reverse
causality is inapplicable to the policy intervention.

## ICML decision

The mechanism is now strong enough to remain an ICML 2027 main-line candidate:
it has a defensible coupled threshold theorem, a negative unrestricted-mixing
boundary, a low-complexity controller, and independent formal mechanism/affine
evidence.  It is not yet submission-complete.  The remaining mandatory
empirical gap is external nonlinear breadth with communication-matched strong
baselines, controlled cross-agent correlation, realistic delays, ablations,
and wall-clock accounting.  That next stage is the first stage for which a GPU
is justified.
