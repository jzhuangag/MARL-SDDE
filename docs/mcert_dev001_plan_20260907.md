## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: development experiment plan
- Origin Date: 2026-09-07
- Verification Status: DEVELOPMENT ONLY; SEEDS TOUCHED BY RUNNER TEST
- Version Label: MCERT-DEV-001-plan-v1

# MCERT-DEV-001: Markov certificate nonvacuity development

## Status

This is a development audit, not an independent preregistered confirmation.
During static testing, the test suite invoked the stochastic runner on seeds
`93000,...,93063`.  Those seeds are therefore permanently development-only.
No result obtained from them may be presented as held-out formal evidence.

## Question

Does the proved count-uniform, state-conditional Markov certificate become
nonvacuous at a local sample size small enough to be plausible as a common
warm-up/replay phase, while rejecting the state-incompatible edge and retaining
local computation independent of the total agent population?

This audit validates the theorem--algorithm interface only.  It is not a
standard MARL result, does not use return, and cannot authorize a paper claim
about Pursuit efficacy.

## Development model

- two local relevance states with transition matrix
  `[[0.9,0.1],[0.1,0.9]]`;
- two positive communication actions with score vectors
  `[0.25,-0.25]` and `[-0.25,0.25]`, plus a zero-score null;
- four-transition future packet;
- 64 development seeds `93000,...,93063`;
- per-positive-action transition counts
  `256,512,1024,2048,4096,8192`;
- event failure probability `0.05` and simultaneous action count three;
- every action-specific Markov transition is charged, so total identification
  transitions at sample size `n` are `2n`.

The score observations are deterministic conditional on the state.  The
transition law is learned independently for each positive action; no symmetry
or shared-kernel information is used by the certificate.  This deliberately
charges the conservative interface.

## Development targets

At the 4,096-transition primary design point, the intended targets are:

1. correct edge selected in at least 99% of seed--state cases;
2. median lower-bound/exact-value recovery at least 50%;
3. fifth-percentile recovery at least 40%;
4. incompatible edge has a positive lower bound in zero cases;
5. median recovery is nondecreasing over the six sample sizes;
6. every transition is charged exactly;
7. declared local robust-DP scalar work is at most 128 operations per
   two-action decision (`2 actions x 4 steps x 2^2 states = 32` here);
8. all reported numerical rows are finite.

These are development targets only.  If they pass, an independent
confirmation must freeze new seeds, the unchanged model and gates, source and
configuration hashes, and byte-identical reproduction before execution.

## Reproducibility

The source is `run_markov_certificate_nonvacuity.py`.  Primary and clean
reproduction JSON files must be byte-identical.  Run on local CPU only; no
HPC4, GPU, or `/project` storage is authorized.
