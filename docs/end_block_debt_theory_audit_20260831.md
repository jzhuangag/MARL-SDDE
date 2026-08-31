# End-block debt: exact identities and shield interface

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory/code validation
- Origin Date: 2026-08-31
- Verification Status: ANALYZED
- Version Label: audit_v1

This internal derivation introduces no external bibliographic claims. It is not a completed convergence theorem or a new experiment authorization. Frozen controller, seeds and outcomes are unchanged.

## 1. Implemented probability law

`t071_sampled_graph_controller.py:37–76` samples one stationary Gaussian vector AR(1) chain continuously across blocks. It does **not** reset at block boundaries. Within a block the target is constant; targets can change across blocks. Spatial covariance is `noise_scale * ((1-rho) I + rho 11^T) + 1e-14 I`. The parameter named noise_scale is a variance multiplier here, not a standard deviation.

Thus independence between blocks is valid at temporal correlation zero, not at nonzero correlation. Any analysis treating nonzero-correlation block innovations as independent must be explicitly justified or corrected. This report does not yet audit every earlier exact-moment baseline.

## 2. Paired debt identity: same-data reuse is not automatically biased

Fix recipient i and a block of m observations Y_s=theta+xi_s. Let x and s0 denote its pre-block learner and local-shadow parameters, and eta its common gain. Define a=(1-eta)^m. The implemented affine local updates satisfy

    L = a*x + eta * sum_{s=1}^m (1-eta)^(m-s) Y_s,
    S = a*s0 + eta * sum_{s=1}^m (1-eta)^(m-s) Y_s,
    d = L-S = a*(x-s0).

With Ybar=theta+xibar, the observed excess ghat and true pre-mix excess g obey the exact identity

    ghat = (L-Ybar)^2 - (S-Ybar)^2,
    g = (L-theta)^2 - (S-theta)^2,
    ghat-g = -2*d*xibar.

Although L and S use the same data as Ybar, their difference is measurable before this block. With independent zero-mean block noise, E[ghat-g | past]=0. Therefore blanket claims that same-block reuse necessarily biases this particular paired statistic would be wrong.

For the continuous AR(1) chain xi_s=lambda*xi_(s-1)+innovation_s, take the full pre-block filtration, including xi_0. Then

    E[xibar | past] = c_m(lambda)*xi_0,
    c_m(lambda) = sum_{s=1}^m lambda^s / m,
    E[ghat-g | past] = -2*a*(x-s0)*c_m(lambda)*xi_0.

For an observable smaller filtration, replace xi_0 by its conditional expectation; zero bias still does not follow automatically. The expression vanishes at lambda=0. It is not generally zero at lambda>0 because the learner depends on earlier noise.

## 3. The queue controls a surrogate, not the reported post-mix risk

For Q_(k+1)=[Q_k+ghat_k-epsilon]_+, pathwise telescoping gives

    sum_{k<T} ghat_k <= T*epsilon + Q_T-Q_0.

With V(Q)=Q^2/2, the exact generic upper bound is

    V(Q_(k+1))-V(Q_k)
      <= Q_k*(ghat_k-epsilon) + (ghat_k-epsilon)^2/2.

Neither identity proves Q_T/T -> 0. A drift/stability argument and moment conditions are still required; Gaussian observations do not supply a deterministic bound on increments.

The queue is updated using L and S **before** the current graph choice. Reported risk uses the post-mix parameter M. Its true excess decomposes as

    (M-theta)^2-(S-theta)^2
      = g + 2*(M-L)*(L-theta) + (M-L)^2.

The last two terms are not constrained by the queue update just executed. A last-block harmful mix has no subsequent debt feedback at all. Hence even unbiased pre-mix ghat plus a small observed queue does not establish post-mix terminal no-harm.

The QP's Q*||w-e_i||^2 penalty is not the queue drift term by identity. A bound linking it to the relevant action-dependent future excess is missing; it also needs model scale and target bounds. The existing solver is a convex regularized surrogate, not yet a proved drift-minimizing risk controller.

## 4. A mathematically valid conditional shield interface

Suppose a target confidence set satisfies |theta_i-hat_theta_i|<=r_i on a joint coverage event. For any candidate scalar prediction z_i, and reference s_i, define

    U_i(z_i,s_i) = (z_i-hat_theta_i)^2-(s_i-hat_theta_i)^2
                   + 2*r_i*|z_i-s_i|.

On that event, true excess relative to s_i is at most U_i, **simultaneously for all z_i and s_i**, including data-dependent choices. Proof: subtract the two quadratic differences and apply |theta_i-hat_theta_i|<=r_i. No independent validation trajectory is required for this deterministic implication; obtaining valid adaptive/time-uniform target coverage is still a separate obligation.

For z_i=w_i^T v_i, U_i is convex in w_i. A simplex constraint plus U_i<=epsilon is a convex quadratic/absolute-value constraint, not in general a plain linearly constrained QP. An auxiliary absolute-value variable gives a convex QCQP representation. Its complexity and practical implementation are unverified here.

Reference choice is essential: setting s_i=local post-update gives local-step safety only. Setting s_i=the persistent local shadow targets cumulative collaboration damage; however the shadow may lie outside the donor convex hull, making that constraint infeasible. A prospective safe action set must explicitly include executing the shadow (which gives U_i=0), with its computation and communication accounted. Identity graph alone need not reproduce the shadow.

This interface is conditional and pointwise at the evaluated target, not a guarantee on the next changing target or subsequent Markov learning risk. Nonstationary target coverage, noise calibration, tracking error, and finite-time accumulation remain open. Do not claim a complete shield or launch efficacy experiments until these are resolved.

## 5. Computational scope

The current dense covariance and spectral operations are not O(qd) in general. For n scalar agents, covariance construction costs O(m*n^2); dense eigendecomposition costs O(n^3). Each recipient QP additionally performs an n-by-n eigendecomposition, giving O(n^4) across n recipients with the current implementation, before iterative solves. Dense projected-gradient work is O(K*n^3) across recipients plus projection sorting. Small n=4 success does not establish scalable many-agent complexity.

## Next step

Audit earlier exact-moment static references against continuous cross-block AR(1), then derive a valid target-confidence construction under explicitly known/bounded noise and mixing. Preserve the distinction between a conditional shield lemma, queue stability, and finite-time learning convergence. No GPU needed for these tasks.

## Execution record

Source checkout: `8669f25` before this additive audit. Process check found no Python experiment running. Command: `.venv/Scripts/python.exe -m pytest experiments/dependence_delay_linear/test_end_block_debt_audit.py -q` from the repository root. Result: **5 passed in 0.31s**. These verify identities and counterexamples, not statistical coverage or algorithm effectiveness. No formal data were read for design, no scientific runner executed, no HPC4 access performed.
