# Augmented covariance verification and shield geometry

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory/code validation
- Origin Date: 2026-08-31
- Verification Status: ANALYZED
- Version Label: audit_v1

## Independent covariance engine

For deterministic maps x_t=A_t x_(t-1)+B_t xi_t and stationary noise xi_t=lambda xi_(t-1)+u_t, use augmented state (x_t,xi_t). Its transition and innovation injection are

    F_t = [[A_t, lambda B_t], [0, lambda I]],
    G_t = [[B_t], [I]],
    P_t = F_t P_(t-1) F_t^T + (1-lambda^2) G_t Sigma G_t^T.

Initial parameter state is deterministic; initial noise has covariance Sigma and zero cross covariance with that state. These assumptions are explicit. The cross block of P retains parameter/noise dependence, resolving the omission diagnosed in the legacy block propagator. Deterministic target offsets affect means but not these covariances.

`continuous_ar1_covariance_audit.py` implements this recursion independently of all frozen runners. Tests compare against a second construction: express x_t as a linear map C_t of all past noise vectors and compute C_t (R_t(lambda) tensor Sigma) C_t^T. Agreement is checked at every step for lambda in {0,.4,.9}, delay in {0,1,3}, time-varying deterministic maps, and correlated agents. The two-step witness returns .1168, not the legacy .0656. This verifies the covariance primitive, not integration of the exact frozen end-block history convention or correctness of optimized graph weights. Data-dependent maps are outside its contract.

## Nontriviality of the scalar target-interval shield

Use the previous report's simultaneous target coverage event, center h, radius r>=0 and shadow s. Let displacement d=z-s. The exact robust excess upper bound becomes

    U(d) = d^2 + 2*d*(s-h) + 2*r*|d|.

Minimizing separately on d>=0 and d<=0 gives

    min_d U(d) = -[max(|s-h|-r,0)]^2,
    d_star = -sign(s-h)*max(|s-h|-r,0).

Consequently a strict negative bound is possible over all real predictions iff |s-h|>r. If s lies in the interval, every nonzero displacement has U>0. This is also logically necessary: the interval then includes theta=s, at which shadow risk is zero and any different prediction is worse. A graph-constrained action set can only further restrict improvement; being outside the interval alone does not guarantee the donors provide a useful direction.

This is a geometric consequence of the selected robust safety requirement, not an empirical claim that all useful collaboration must satisfy it. A wide confidence interval can make this particular shield permanently conservative even when expected-risk collaboration would help. Do not run an efficacy pilot before assessing this obstruction analytically.

## Low-cost scalar shielding of any proposal

For a proposed p, interpolate from the explicit shadow action: z=s+beta*(p-s), beta in [0,1]. Write v=p-s and c=v*(s-h)+r*|v|. For v!=0, U(beta)=v^2 beta^2+2*c*beta. At tolerance epsilon>=0, the feasible interval is

    0 <= beta <= min(1, (-c+sqrt(c^2+v^2*epsilon))/v^2).

At epsilon=0 it reduces to beta_max=min(1,max(0,-2*c/v^2)). At v=0 the proposal equals shadow and beta may be one. This is O(1) scalar shielding per recipient after a proposal is available, not a low-complexity guarantee for the proposal generator or covariance estimator. For epsilon>0 implement the positive root with a numerically stable branch if used later. This rule is not implemented into the frozen algorithm.

The guarantee is current-target high-probability excess relative to an executable shadow. It does not imply positive gain, unconditional expected-risk no-harm, next-target tracking, or queue stability. A useful final design may require cumulative risk accounting rather than insisting on per-step robust domination; that would need a new proof, not a relaxed retrospective gate.

## Next audit boundary

Integrate deterministic block-end maps with the frozen delay/history convention in a separate evaluator and test means/covariances against direct linear responses. Then examine shield feasibility using public model bounds, without reading formal endpoints for tuning. No efficacy registration yet.

## Execution record

Starting checkout `7e77a14`, clean tree, no Python jobs observed. Command: `.venv/Scripts/python.exe -m pytest experiments/dependence_delay_linear/test_continuous_ar1_covariance_audit.py experiments/dependence_delay_linear/test_markov_baseline_law_audit.py experiments/dependence_delay_linear/test_end_block_debt_audit.py -q`. Final result: **25 passed in 0.83s**. An earlier 24-test run passed before adding the shield geometry test. No stochastic seeds, scientific result directories, frozen modifications, or remote operations. Full-suite result remains the previously recorded run, not claimed rerun here.
