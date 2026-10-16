# PDSG-EST-001 preregistration: sparse signed-alignment estimator

Date: 2026-09-06

Status: frozen before runner implementation and before any sampled result.

## Research question

Can the expectation-level sparse signed-alignment score be estimated accurately
enough to preserve action ordering after Markov correlation, Taylor remainder,
the declared sparse tail, and a fully charged independent control/update split?

This is an estimator feasibility gate.  It does not train a policy, evaluate a
return, reuse PDSG-001 outcomes, or constitute formal paper evidence.

## Frozen model

The audit uses four-dimensional sparse quadratic owner-gradient models with a
no-refresh action and `Delta` refresh actions.  Current-gradient, base-gradient,
and local-Jacobian statistics follow stationary Gaussian AR(1) processes with
the registered correlation and variance.  The true candidate additionally
contains deterministic Taylor and omitted-tail perturbations whose norms equal
their registered envelopes.  Candidate cache-reset terms and communication
prices are observed exactly.

Each row generates two independent Markov blocks of length `m`.  The control
block estimates the action scores and chooses the update mass; the update block
is generated independently under the chosen true candidate.  The charged
sample count is exactly `2m`; neither half is free.

The scenario generator is fixed by seed 93000.  Its true best--second-best
margin defines the favorable/adverse populations using the thresholds in the
manifest.  This is an outcome-free model property, not a filter on noisy pilot
performance.

## Frozen analysis

For every action and seed, record the true optimized score, estimated optimized
score, registered expectation-error bound, selected action, oracle action,
score regret, and charged control/update samples.  Aggregate first within a
scenario across the 64 new seeds and then across frozen scenario populations.

The uniform score-error allowance is the sum of per-action expectation bounds.
The relative separation statistic is twice this uniform allowance divided by
the true best--second-best margin.  This is intentionally conservative and is
not reported as a high-probability certificate.

## Mandatory gates and stopping

Gates E1--E9 are exactly those in `pdsg_est001_manifest_20260906.json`.  Any
failure among E1--E8 forbids clean reproduction and any successor learning
pilot.  No gate, seed, population, noise level, or model parameter may be
changed after seeing sampled output.  A passing result would authorize only a
separate benchmark-tail audit, not a policy-learning experiment.

The run is local CPU only.  No GPU, HPC4, remote storage, formal seeds, or
standard MARL benchmark is authorized.
