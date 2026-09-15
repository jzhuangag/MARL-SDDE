# PDSG-COMP-001 cross-policy alignment cost preregistration

Date: 2026-09-06

## Purpose

This is an engineering feasibility audit, not an efficacy experiment.  It
measures whether the proposed full-information signed edge statistic can be
formed with one batched cross-policy alignment derivative at acceptable CPU
cost relative to an ordinary owner actor-gradient calculation.

The benchmark does not use an environment, return, trajectory, PDSG-001
outcome, formal seed, GPU, or HPC4.  Passing it cannot establish learning
benefit, estimator accuracy, Markov validity, or ICML readiness.

## Frozen comparison

All distinct actor networks are evaluated on deterministic random tensors.  A
smooth centralized interaction surrogate couples the owner action
distribution to the action distributions of its declared donor neighbors.

The baseline performs the common forward computation and differentiates the
surrogate with respect to the owner actor.  The proposed calculation retains
that owner-gradient graph, forms its alignment with a stopped reference
direction, and differentiates the scalar alignment jointly with respect to all
eligible donor parameter blocks.  The derivative is partitioned by donor, so
no per-edge backward loop is allowed.

Both paths include tensor creation outside the timed region, use CPU float32,
and alternate execution order.  Five warmups precede twenty measured
repetitions for each of three seeds.

## Frozen gates

The machine-readable manifest freezes C1--C6.  C3 and C4 test relative compute
overhead.  C5 tests the claim that one joint reverse derivative does not scale
as one separate backward pass per edge.  C6 is run only if C1--C5 pass.

No threshold may be modified after observing the primary timings.  A failed
gate is an engineering failure for the current estimator interface; it does
not alter the earlier problem-headroom result.

Manifest: `policy_dependency_hvp_cost_manifest_20260906.json`.
