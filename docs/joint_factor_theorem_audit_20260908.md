## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: proof and implementation consistency audit
- Origin Date: 2026-09-08
- Verification Status: CONDITIONAL CORE BOUND AND RANDOMIZED-OWNER COROLLARY ALIGNED; SCORE COVERAGE OPEN
- Version Label: joint_factor_theorem_audit_v1

# Audit of the joint graph--packet-weight Lyapunov theorem

## Audit decision

The topology correction requested after the Pursuit support-turnover finding
is valid and is implemented in the right layer. The principal controller and
the principal finite-time bound use

\[
\mathcal L_t^{\rm core}=VF(\theta_t)+Q_t^2/(2\nu).
\]

Consequently a changing candidate graph changes the minimization domain but
does not itself jump the core Lyapunov state. The optional cache-energy
extension is valid only on a fixed edge universe. If weights or support
indicators change, the exact additional jump is

\[
\Xi_t=\frac12\sum_{e\in U}
(\beta_{e,t+1}-\beta_{e,t})\|\theta_{j(e),t}-\chi_{e,t}\|^2.
\]

The executable `topology_motion_increment` matches direct energy differences,
and an explicit regression shows a positive jump even with no policy update
or cache refresh. High graph turnover is therefore not silently cancelled.

## Term-by-term implementation alignment

For each candidate, `choose_core_factor_action_variable_bounds` computes

\[
m(a)=V[A^L(a)-LG(a)M],\qquad C(a)=VLG(a)^2
\]

and exactly minimizes

\[
-m(a)\alpha+C(a)\alpha^2/2+Qc(a)
\]

over the packet-weight interval before scanning the finite edge set. Dense
grid tests agree with the scalar root. The queue recursion associated with
`Q^2/(2 nu)` is `Q^+=[Q+nu(c-bar c)]^+`, which produces the same `Qc` price
used by the implementation. Communication is charged at launch and does not
depend on whether the delayed packet later receives zero weight.

For the cache extension, the code separately verifies the launch cache reset,
donor-update increment, topology-motion increment, and their exact ordered
sum. These quantities do not enter the controlled-kernel confidence radius;
doing so would double count one physical event.

## Corrections made by this audit

The theorem note previously used one symbol `R_p` both before and after
division by `V`. Equation (5) now writes the composite-drift remainder as
`V mathcal R_p`, and the final normalized bound contains `mathcal R_p`. This
is a notation/units correction, not an experimental change.

The audit also states explicitly that a uniform alignment error contributes
`2 bar(alpha) sum epsilon_p` after normalization. Finally, the theorem's left
side is identified as selected-block stationarity at launch states. The added
randomized-owner corollary closes the full-gradient conversion when every
owner has predictable probability at least `pi_min`: conditioning before the
owner draw gives a lower bound of `kappa_min pi_min ||grad F||^2`. Cyclic or
random-permutation ownership at changing iterates needs an epoch-motion lemma
rather than the per-launch probability corollary. This is now proved in
`owner_permutation_epoch_stationarity_20260908.md`: the conversion pays the
measured within-epoch path-motion remainder and is valid for any complete
permutation.

## Exact conditional scope

The proof is complete as a conditional online-optimization interface provided
all of the following are supplied:

1. an injective launch--receipt ledger and eventual drain;
2. block smoothness, clipping, and a launch-measurable receipt-motion bound;
3. one simultaneous lower alignment event over the current finite candidates;
4. a launch-measurable comparator with conditional expected communication at
   most the budget rate;
5. a comparator descent inequality in the same controlled packet model.

Under those conditions, chronological summation accounts for each objective
jump at its receipt and each queue jump at its launch exactly once. The
minimizer can be compared at the original launch because its edge, packet
weight, cost, and score event are launch measurable.

## Remaining theorem kill gates

This audit does not manufacture the remaining missing piece needed for an ICML
main theorem:

- a nonvacuous simultaneous alignment lower bound for the learned continuous
  factor critic under Markov trajectories and controlled cache-dependent
  kernels.

Full-gradient stationarity is available either for i.i.d. owner draws with the
declared positive conditional probabilities or for complete owner permutations
with the separately proved epoch-motion remainder. A future theorem-facing
controller experiment must implement one of these two contracts and, in the
permutation case, log the receipt-update path motion entering that remainder.

The Multiwalker algebra verifies how all candidates are scored in
`O(Delta)`, but it does not yet verify the statistical lower bound. The
standard-benchmark oracle-headroom gate is being run before that estimator is
fit so that a statistically difficult critic is not developed for a problem
with no intrinsic policy-profile value.
