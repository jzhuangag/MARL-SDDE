## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free development design
- Origin Date: 2026-09-07
- Verification Status: OUTCOME-FREE SCORE-SCALE AUDIT
- Version Label: pistonball_score_scale_audit_v1

# Pistonball Lyapunov score-scale audit

The qualified Pistonball learner fixes actor receipt step `0.01` and critic
step `0.0003`.  Before comparing graph controllers, this single-run diagnostic
measures the numerical scale of the three terms that already define the
Lyapunov action index: signed learning-drift change, exact cache-reset energy,
and dual-queue price.  It does not select a method from return and is neither a
pilot nor formal evidence.

The run uses development-only seed `79004`, 2,048 launches, 20 distinct actors,
rollout horizon four, random maximum delay eight, causal-cone radius six, and
a terminal hard cap of 0.5 refresh per launch.  Unlike the historical prefix
cap, the terminal cap permits the virtual queue to price bursts while still
enforcing the exact finite experiment budget.  The dual update is

\[
Q_{p+1}=[Q_p+\nu(c_p-\bar c)]^+,
\qquad \nu=1,
\]

and the corresponding Lyapunov term is `Q_p^2/(2 nu)`.  Thus the action price
remains `Q_p c_p` and the queue step has an explicit drift interpretation.

The legacy weights `V=100000` and `beta=100000` are used only to expose their
scale.  The committed analyzer reconstructs every edge-minus-null index from
its three parts and reports raw components after removing these weights.  It
then applies the frozen outcome-blind calibration rule

\[
V_{\rm rec}=1/\operatorname{median}_{x\ne0}|\Delta D_x|,
\qquad
\beta_{\rm rec}=1/\operatorname{median}_{x\ne0} B_x,
\]

with each value clipped to `[1e-3,1e8]`.  Returns are retained only as tainted
development diagnostics and cannot change this rule.

The next equal-resource controller-headroom matrix is authorized only if all
index identities hold, both raw learning and cache components are finite and
nonzero, and the queue price is active.  Failure changes the score
normalization or estimator interface before any multi-seed efficacy pilot; it
does not permit post-hoc return tuning.
