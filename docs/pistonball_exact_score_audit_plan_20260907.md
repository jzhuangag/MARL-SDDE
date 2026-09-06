## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free estimator and complexity audit
- Origin Date: 2026-09-07
- Verification Status: FROZEN BEFORE EXACT-SCORE OUTCOMES
- Version Label: pistonball_exact_score_audit_v1

# Exact sparse counterfactual score audit

The failed one-VJP controller is not retuned. This outcome-blind audit replaces
only its estimator. For owner `i`, it samples a batch from replay completed
strictly before launch, evaluates the current-policy owner gradient, then
evaluates the predicted owner gradient under the null cache and every actual
one-edge-refreshed cache in the causal cone. The resulting alignment enters
the same cache-and-queue Lyapunov index. No future reward, transition, packet,
or evaluation is available to the action.

The exact score removes the finite policy-displacement Taylor approximation
but costs one current gradient plus one reverse evaluation for each of the
`1+Delta` candidate actions. Its arithmetic is therefore linear in local
interaction degree, not total agent count. A separate score RNG prevents the
audit from changing the training replay stream.

Two development-only runs use seed `79008`, SiLU networks, 1,024 launches,
rollout horizon four, maximum delay eight, actor step `0.01`, critic step
`0.0003`, and the same finite terminal cap of 0.5 refresh per launch. The
exact-score run uses unit learning/cache weights solely to expose raw scale;
the no-refresh run measures runtime. Returns are stored but excluded from all
gates and future pilot populations.

The Slurm wall limit is 45 minutes. This exceeds three times the approximately
ten-minute no-refresh development runtime, so the runtime gate can be measured
rather than being confounded with scheduler termination.

The audit passes only if the signed and cache components are nonzero, signed
value favors a refresh in at least 1% of scored launches, the queue is active,
the observed reverse-call count is at most eight (cone degree at most six plus
current and null gradients), and runtime is at most three times no-refresh.
Weights for one final headroom design are fixed by reciprocal median nonzero
absolute component magnitude clipped to `[1e-3,1e8]`. A failed audit stops the
Pistonball policy-cache mainline rather than launching another performance
matrix.
