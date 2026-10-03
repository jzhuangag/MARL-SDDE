## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory--implementation development audit
- Origin Date: 2026-09-08
- Verification Status: CAUSAL PACKET INTERFACE IMPLEMENTED; CONFIRMATION NOT AUTHORIZED
- Version Label: pursuit_delayed_alignment_development_v1

# Pursuit delayed-alignment development audit

## Question

The remaining nonlinear interface question is not whether a neural statistic
can be made nonzero.  It is whether a launch-measurable local representation
can predict the conditional learning value of refreshing one teammate-policy
cache after the selected trajectory packet returns.  The target must preserve
the effect of the refresh on both the Markov trajectory law and the owner
policy-gradient direction.

## Implemented causal target

For owner `i`, launch event `p`, and candidate `a`, let `r_(p,h)` and
`ell_(i,p,h)(a)` be the shared reward and owner log probability along the
`H`-step mixed-policy branch.  The branch packet is

\[
 g_p(a)=\nabla_{\theta_i}
 \left[-\sum_{h=0}^{H-1}\gamma^h r_{p,h}(a)
 \ell_{i,p,h}(a)\right].
\]

The reference `v_(i,p)` is an exponentially weighted combination of owner
packets that arrived strictly before launch `p`.  The bounded delayed response
is

\[
 Y_p(a)=\operatorname{clip}\left(
 \frac{\langle v_{i,p},g_p(a)\rangle}
 {s\lVert v_{i,p}\rVert_2},-1,1\right).
\]

Thus the response is unavailable until receipt, retains trajectory reward
magnitude, and is the same signed alignment that enters the learning-potential
drift.  The earlier cosine prototype was rejected because normalizing by
`||g_p(a)||` cancels reward magnitude in a one-step score-function packet and
can make the response nearly launch-known.

## Exact counterfactual audit

The development implementation copies the underlying Pursuit Markov state at
launch and advances the null and every eligible one-edge refresh using common
actor uniforms and the same environment random state.  It also advances the
actually selected branch through the public parallel environment.  Across all
development runs, selected-branch reward and signed-alignment discrepancies
were zero up to floating-point precision.  These extra branches are an audit
oracle and are not available to the online controller or counted as efficacy
samples.

The candidate graph remains a deterministic degree-four function of current
pursuer positions.  A non-null action copies exactly one actor and charges its
exact parameter bytes.  Feedback is inserted into the reference only at its
declared receipt event.  The current fixed feature map uses 39 local quantities:
owner and donor spatial summaries, cache age, categorical policy total
variation, and the launch reference--score projection.  Candidate formation is
`O(Delta d)` and a shared ridge update is `O(d^2)`.

## Development findings

Three findings change the qualification design.

1. A random multilayer-perceptron policy path with one-step packets produced
identical candidate outcomes in the held-out development rows.  This is a
zero-control-value case, not evidence against the drift theorem.
2. Multi-step packets and an outcome-free local chase policy path create a
genuine edge-effect phase.  With an abrupt public version change and an
eight-step packet, 42.19% of valid launches had nonzero candidate spread and
28.13% had a strict oracle gain over null in a small development check.  The
average gain remained small, so this is mechanism evidence only.
3. Calibrating an interval against individual noisy packet responses is the
wrong target.  Episode-maximum raw-response intervals covered but were wider
than two response standard deviations.  The theorem's `A_p(a)` is the
conditional mean; packet noise belongs in the sub-Gaussian term.  A valid
nonvacuity audit must estimate the conditional mean with independent branch
replicates and assess critic approximation separately from packet noise.

The handcrafted linear head did not pass that conditional-mean prediction
test in the small development split.  No independent confirmation, controller
efficacy experiment, or GPU pilot is authorized by these runs.

A final information-sufficiency check added a fixed eight-dimensional sketch
of the launch reference gradient, the owner score gradient, their coordinatewise
product, candidate policy probabilities, actor identities, and event phase.
With 2,560 representation packets and 2,560 disjoint linear-head packets, the
held-out replicated-mean `R^2` was `-0.1398` overall (`-0.0592` at low
mismatch and `-0.2236` at high mismatch).  This is an improvement over the
39-feature prototype but remains worse than the test mean.  Generic context
regression is therefore stopped; increasing sketch dimension, network depth,
or sample size is not an admissible rescue.

The replacement interface must be critic-derived.  For a pairwise local factor
critic, define

\[
 \widehat Q_i^a(u_i)=\widehat q_i(o_i,u_i)
 +\sum_{j\in\mathcal N_i}
 \sum_{u_j}\pi_{j|i}^a(u_j|o_j)
 \widehat q_{ij}(o_i,o_j,u_i,u_j).
\]

Then the predicted alignment is

\[
 \widehat A_p(a)=\sum_{u_i}
 \langle v_{i,p},\nabla_{\theta_i}\pi_i(u_i|o_i)\rangle
 \widehat Q_i^a(u_i).
\]

All owner-policy directional derivatives are computed once for the five
actions.  Refreshing `j->i` changes only the `ij` factor, so every candidate is
scored in `O(Delta A^2)` arithmetic, which is `O(Delta)` for Pursuit's fixed
five-action space.  This formula preserves the full reference direction and
ties the estimator to a centralized-training critic rather than an arbitrary
context predictor.  It is now the only authorized estimator design.  The
corresponding implementation has since matched exhaustive neighbor-action
enumeration, a centered finite-difference directional derivative, and a fixed
five-reverse-pass complexity audit; the separate theorem--implementation note
records the derivation and the still-open statistical critic gate.

## Frozen next interface, before any efficacy run

The next admissible qualification has four disjoint sources of data:

1. representation episodes fit a local neural factor encoder from selected,
   fully charged completed packets;
2. head episodes freeze that encoder and fit only a finite-dimensional linear
   alignment head;
3. calibration episodes estimate an approximation allowance against
   independently replicated conditional means;
4. untouched test episodes evaluate mean coverage, interval width, signed edge
   ranking, byte accounting, delay causality, and measured runtime.

Encoder and head updates occur only at predictable epoch boundaries.  The
delayed ridge state is reset after an encoder change, matching the piecewise
theorem and its `sqrt(K)` epoch penalty.  Individual packet residual coverage
is reported only as a noise diagnostic; it is not substituted for coverage of
the conditional alignment.

This qualification may use the public heuristic path to locate a nonzero
mismatch phase, but a standard-MARL claim still requires checkpoints generated
by a separately preregistered learner.  A positive heuristic-path result alone
cannot authorize the paper's principal empirical claim.
