# PDSG-COMP-002 local-factor alignment-cost preregistration

Date: 2026-09-06

## Question

Does a materially sparse scheduling statistic have compute cost controlled by
the declared local policy-dependency degree rather than the total number of
agents?

This audit follows the failed C5 gate of PDSG-COMP-001.  It does not relax or
rerun C5.  The estimator is changed: only owner-local critic factors and local
donor policies participate in its forward and autograd graph.  All non-neighbor
actors are excluded from the scheduling calculation.

## Scope

The audit uses deterministic random tensors and CPU autograd.  It contains no
environment trajectory, return, learning outcome, PDSG-001 seed, GPU, or HPC4
operation.  It establishes neither true-gradient sparsity nor learning gain.

The total number of registered actors varies from 16 to 256 while local degree
varies from two to eight.  The timed region contains only the owner-local
factor.  An untimed structural probe verifies that a non-neighbor actor block
is absent from the alignment derivative graph.

## Claims tested

The intended complexity is linear in local factor size and independent of
global agent count at fixed local degree.  It is not degree-independent.  L5
tests global-count invariance; L6 permits subquadratic but nonconstant degree
growth.  L3 and L4 bound the overall cost relative to an ordinary local actor
gradient.

All gates and the stopping rule are frozen in
`sparse_policy_dependency_hvp_manifest_20260906.json`.  A failed primary gate
forbids clean reproduction and learning experiments for this estimator.
