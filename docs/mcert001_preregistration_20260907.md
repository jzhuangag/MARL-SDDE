## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: independent confirmation preregistration
- Origin Date: 2026-09-07
- Verification Status: OUTCOME-FREE; EXECUTION FORBIDDEN BEFORE COMMIT
- Version Label: MCERT-001-prereg-v1

# MCERT-001 independent Markov-certificate confirmation

## Purpose and separation

MCERT-DEV-001 passed all development targets, but its seeds were touched by a
runner-invoking test before commit.  This confirmation uses 128 untouched seeds
`95000,...,95127`.  The scientific model, sample grid, primary point, charging,
confidence level, gates, and analyzer are unchanged.  Only the audit ID,
independent seeds, and seed count differ.

The confirmation addresses one theorem--implementation question: whether the
state-conditional tabular Markov lower bound is nonvacuous and directionally
correct with fully charged data.  It is not a return experiment and cannot be
reported as standard MARL efficacy.

## Frozen configuration

- two-state kernel `[[0.9,0.1],[0.1,0.9]]`;
- positive-edge scores `[0.25,-0.25]` and `[-0.25,0.25]` plus the null;
- packet horizon four;
- sample sizes per positive edge: `256,512,1024,2048,4096,8192`;
- primary point: 4,096 per edge, 8,192 transitions charged in total;
- event failure probability 0.05 and simultaneous action count three;
- 128 seeds `95000,...,95127`, disjoint from development.

Configuration SHA-256 is
`8fafb90456be3537228482411dddc63caab55bdd734c5f3088e334843f96b3d3`
and is statically asserted in the source before this preregistration commit.
Expected output is 1,536 rows.

## Mandatory gates

1. correct edge selected in at least 99% of primary seed--state cases;
2. median certified/exact value recovery at least 50%;
3. fifth-percentile recovery at least 40%;
4. incompatible edge positive in zero primary cases;
5. median recovery nondecreasing over the six sample sizes;
6. exact charge `2n` for every row;
7. local robust-DP scalar work at most 128 per two-action decision;
8. every numerical row finite;
9. primary and clean reproduction JSON byte-identical;
10. targeted and maintained-package tests pass.

Any failed gate stops this certificate from serving as the theorem-facing
safety interface.  No threshold, seed, or model change is permitted after
execution.  A pass authorizes integration into a CPU tabular asynchronous
controller and a separate outcome-free neural-interface design; it does not
authorize a Pursuit efficacy claim by itself.

## Execution boundary

Run primary and clean reproduction locally on CPU only after the independent
preregistration commit exists.  Do not use HPC4, a GPU, `/project`, old formal
outcomes, or development seeds.
