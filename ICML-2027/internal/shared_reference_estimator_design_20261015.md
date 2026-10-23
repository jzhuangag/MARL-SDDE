# Prospective shared-reference estimator extension

## Material Passport

Original derivation plus prospective estimator qualification. Registered while
the A1 first-disagreement experiment was running, before reading any of its
scientific return values. No old gate, method, candidate, or outcome is changed.

## Why this extension is necessary

The branch estimator's absolute native sampling bill is known before outcomes:
14,400 transitions per three-candidate score bank. Its variance theorem does
not by itself make per-update probing affordable. The new data path collects
one independently seeded 64-episode *reference-policy* bank (1,600 joint
transitions), reused by all three candidates. Candidate and reference policy
forwards, critic forwards, context reconstruction and downstream validation
are separately accounted. Shared candidates are correlated, and their full
score covariance is retained.

## Exact relative-return identity

Let beta be the frozen reference policy and pi a frozen candidate, both with
full categorical support. On a beta episode let L_t be the cumulative product
of joint action likelihood ratios through time t. Let v_t(s_t) be any bounded
state/history function fixed independently of this reference bank, with
v_H=0. Define

`X = sum_{t=0}^{H-1} (L_t-1) [r_t + v_{t+1}(s_{t+1}) - v_t(s_t)]`.

The reward terms have expectation J(pi)-J(beta). Conditional on the pre-action
history, E_beta[L_t v_t] = E_beta[L_{t-1} v_t]. Thus the potential terms telescope
under both policies, leaving the same initial-state potential and zero
terminal potential; their difference is zero. Hence X is unbiased for actual
undiscounted team-return difference even when v is a badly fitted critic.
It is not a critic-loss objective or a claim that PPO/GAE is unbiased.

Also report three prespecified alternatives: full trajectory IS,
initial-state-centered trajectory IS, and per-decision IS. The primary method
is fixed as the TD control variate before outcomes. None uses clipped or
self-normalized ratios. The exact finite MDP check intentionally uses a wrong
value function. Native sampling and log ratios normalize the same rounded
network probabilities in float64, avoiding a discrepancy between CDF sampling
and likelihood ratios. Equal policies yield exact zero scores.

This is a standard importance-sampling/control-variate construction, not a
novelty claim. It removes the need for counterfactual simulator branches from
the score collection path, but still requires known behavior probabilities,
common support, independent score data, and frozen target candidates.

## Qualification and limitations

The policies are precisely the already prospectively frozen A1 contexts and
candidate hashes, reconstructed without looking at A1 outcomes. Score role
510 is disjoint from roles 110, 210--232, and 310. Reusing the existing
prospectively registered independent validation bank does not tune the score.
It avoids treating additional validation as free training data. The same
policies make a direct estimator comparison possible, but this is not an
additional task/seed confirmation of a learning algorithm.

All four estimators, two tasks, stages and seeds are reported. The fixed primary
must pass both task-level cost-adjusted-variance and positive independent-score
checks; no fallback primary is chosen after outcomes. Original A1 gates remain
binding for A1. A failure of positive margins may concern candidate opportunity
rather than estimator validity, and is diagnosed explicitly.

Low empirical variance is still not a simultaneous confidence certificate.
Unclipped likelihood ratios may have large support or tails. A subsequent
uniform confidence/causal-transfer interface remains mandatory before a
certified online Lyapunov controller or GPU experiment. Frozen policies known
after computing an update do not, by themselves, identify its value *before*
launch; the sunk-computation and launch/receipt issue is not hidden.
