# Acquisition value against a budget-wise strong fallback

## Stronger comparator and design decision

Efficient-critic development passed its frozen global-single-q gates, but a
subsequent stronger diagnostic allowed fixed q to depend on the public budget,
overhead, and known mixing configuration. The development ratio then became
1.066058. This is preserved as a diagnostic, not used to rewrite the old gates.
The mechanism is the one-time calibration tax on short actor budgets.

The successor adds a new action: skip acquisition and execute the same strong
public fallback at full budget. Neither q nor a task is removed from the
catalogue. No fresh confirmation outcome is used to design this rule.

## Public fallback

For each publicly known cost/persistence configuration, choose one fixed q
using the declared prior over rho. Spacing maximizes I_N for that q and is
independent of rho. The prior's geometric-risk minimizer is analytically
available and uses no calibration or evaluation outcomes. This is exactly the
public-budget-wise comparator used in the prospective successor.

## Uniform futility theorem

For each paid action a, R_paid(a,rho) = [rho+(1-rho)/q_a]/I_(N_a).
For the unpaid fallback f, R_free(f,rho) has the same affine form with its full
budget horizon. If R_paid(a,0)>=R_free(f,0) and
R_paid(a,1)>=R_free(f,1) for every paid action a, then

min_a R_paid(a,rho) >= R_free(f,rho), for every rho in [0,1].

Proof: each difference is affine, hence its minimum over [0,1] is attained
at an endpoint. All action differences are nonnegative, and taking their
minimum preserves the inequality.

The selector consequently skips a provably futile acquisition without knowing
rho. Its risk and resource use equal the strong fallback in those contexts.
The result holds for any positive total marginal noise scale because that
scale multiplies both risks. The numerical test uses 1e-14 only to tolerate
floating-point roundoff; tests check the interior continuum explicitly.

In other contexts calibration may be useful and is fully charged. The new
study must establish aggregate gain relative to the budget-wise comparator;
the futility test alone is not a universal no-harm theorem for probed contexts.
