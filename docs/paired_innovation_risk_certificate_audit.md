# Direct paired-risk certificate under known AR(1) mixing

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory/code validation
- Origin Date: 2026-08-31
- Verification Status: ANALYZED
- Version Label: audit_v1

This is a self-contained internal derivation, not a novelty claim for exponential-mixture techniques and not a completed algorithm theorem. Frozen code and experiments remain unchanged.

## Observable innovation estimator removes boundary bias

Assume scalar Y_(k,s)=theta_k+xi_(k,s), target constant during each deterministic block of m>=2 steps. Noise follows continuous AR(1) with **known** lambda in [0,1), independent Gaussian innovations u_t of variance v*(1-lambda^2), and known marginal variance upper bound vbar>=v. Targets and pre-block parameters are measurable before that block; innovations are independent of the full past, including other agents' past. Contemporaneous spatial correlation is allowed.

Use only within-block adjacent pairs, excluding the pair crossing a target change:

    htilde_k = sum_{s=2}^m (Y_(k,s)-lambda*Y_(k,s-1))/((m-1)*(1-lambda))
             = theta_k + sum_{s=2}^m u_(k,s)/((m-1)*(1-lambda)).

Thus htilde_k-theta_k is conditionally centered Gaussian independent of the pre-block past, with variance

    nu = v*(1+lambda)/((m-1)*(1-lambda)) <= nubar.

The first observation is still used in forming the second observation's pair; no extra observations are acquired. This is not a guaranteed variance reduction relative to the ordinary mean; its benefit is conditional centering. Replacing lambda by a plug-in estimate or merely its upper bound does not preserve the centering identity. Unknown-mixing use requires a new error/bias analysis.

## Predictable paired loss cancellation

For the common-gain affine updates L_k,S_k from learner and shadow, d_k=L_k-S_k=(1-eta)^m(x_k-s_k) is pre-block measurable. Define

    g_k=(L_k-theta_k)^2-(S_k-theta_k)^2,
    ghat_k=(L_k-htilde_k)^2-(S_k-htilde_k)^2.

Then E_k=ghat_k-g_k=-2*d_k*(htilde_k-theta_k) is conditionally Gaussian, centered, with variance at most v_k=4*d_k^2*nubar. No independence between L_k and htilde_k is needed: cancellation makes their difference predictable. No inference of predictable post-mix displacement is permitted.

For multiple recipients, construct separate bounds with failure allocation alpha_i whose sum is <=delta. Within-time spatial independence is not needed for that union bound. Each recipient's coefficient must be measurable before all the block innovations used in its statistic.

## All-prefix cumulative certificate

Let S_t=sum_{k<t} E_k and V_t=sum_{k<t} v_k. For any fixed real a, conditional Gaussian exponential moments give a nonnegative supermartingale

    exp(a*S_t - a^2*V_t/2).

Average these over a centered normal density with precision v0>0, fixed before observing data. Completing the square gives

    M_t=sqrt(v0/(V_t+v0))*exp(S_t^2/(2*(V_t+v0))).

Tonelli's theorem and conditional expectation preserve the supermartingale property and M_0=1. To see the crossing bound without assuming optional stopping at an unbounded time: stop at the first crossing of 1/alpha or deterministic N. Nonnegativity and the supermartingale property imply crossing probability by N <=alpha; let N increase. Hence, with probability >=1-alpha, simultaneously for all t,

    |S_t| <= b(V_t),
    b(V)=sqrt((V+v0)*(2*log(1/alpha)+log(1+V/v0))).

In particular sum g_k <= sum ghat_k+b(V_t). At V=0 the actual error is exactly zero; the displayed generic bound need not be tight there. With V_t=O(t), this uncertainty grows as O(sqrt(t log t)), whereas adding a separate fixed-width certificate at every block can accumulate O(t) uncertainty. This rate comparison is conditional on the variance-growth assumption; unbounded adaptive d_k does not automatically satisfy it.

## Why this is not yet a safe collaboration controller

1. The certified quantity is pre-current-mix excess, observed at block end. It does not include the immediate post-mix risk terms identified in the debt audit or the last action's unevaluated consequences.
2. Arbitrary data-dependent post-mix d cannot be inserted into this Gaussian argument. For example if d=e and e is centered Gaussian estimator noise, E=-2e^2 has nonzero negative mean. Same-data selection can manufacture apparent improvement.
3. Certification after a loss is incurred is not a prospective bound preventing cumulative budget overshoot. An action-dependent increment bound/reserve or a delayed-risk contract is still required.
4. Controlling true cumulative loss requires controlling sum ghat, V and the baseline tracking error. No Lyapunov drift minimizer or finite-time tracking theorem follows just from the mixture formula.
5. High-probability certification alone does not bound expected failure-event risk for unbounded Gaussian losses. Additional moment bounds or bounded-output assumptions are needed.

## Decision and bounded next task

Keep this as a useful known-mixing pre-mix risk-monitor lemma. Do not replace the current shield or launch efficacy experiments based on it. Next derive the exact bridge from prior post-mix errors to next-block paired excess, including target shifts and AR boundary dependence. Check whether an observable correction and finite risk reserve can cover the action timing without inventing independent data or changing the reported metric. If that bridge fails, record it before proposing a different training schedule.

## Execution record

Starting checkout `b36d76e`, clean worktree, no Python processes observed. Command `.venv/Scripts/python.exe -m pytest experiments/dependence_delay_linear/test_paired_innovation_certificate_audit.py -q`: **9 passed in 0.34s**. Tests check six exact covariance cases, cancellation of arbitrary initial AR boundary state, the mixture-boundary algebra and an endogenous-coefficient counterexample. They do not verify coverage through simulation or complete an independent proof review. No formal data, scientific trajectories, frozen changes, GPU or HPC4 operations.
