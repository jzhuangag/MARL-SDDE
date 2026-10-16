# TSP-V2 coefficient-certificate theorem--code audit

## Decision

The additive hitting-time theorem is valid conditional on a simultaneous
lower drift event, but the earlier interface did not establish that event for
deep MARL.  In particular, a user-supplied scalar `effective_samples` is not a
mixing proof, and a coefficient measured at the calibration policy is not
automatically valid after the policy moves.  Therefore no C1 development job
is authorized yet.

This is an interface stop, not a reversal of G0-B: G0-B remains evidence that
fully charged short branches contain a participation signal.  It is not a
coefficient certificate and is not performance evidence.

## 1. Bounded disjoint Lyapunov observation

For a fixed public center `r0` and scale `s>0`, a validation trajectory return
`G` is mapped to

\[
  V(G)=\frac12-\frac1\pi\arctan\!\left(\frac{G-r_0}{s}\right)\in(0,1).
\]

The validation environments, seeds, and transitions must be disjoint from
the training trajectory used by the candidate update.  The transform is
monotone, so lower validation risk means higher validation return, but the
raw training reward is never called a certificate.  Every validation worker
transition is charged to the environment budget and every worker--server
transfer is charged to the message budget.

This construction makes the observation bounded; it does not by itself make
successive observations independent or make a local coefficient transferable.

## 2. Mixing correction is a coupling budget, not an invented sample size

Let `Z_j=V_j^before-V_j^after` in `[-1,1]`.  From a trace of length `n`, retain
`m` observations after a registered burn-in and at a registered stride `g`.
If an external argument proves `beta(g) <= beta_bar`, Berbee coupling changes
the selected dependent sample to independent copies with failure probability
at most

\[
  \delta_{cpl}=(m-1)\bar\beta.
\]

For catalogue size `K` and total error `delta`, the remaining per-action
concentration budget is

\[
  \delta_{conc}=\delta/K-\delta_{cpl}.
\]

The plan is invalid when `delta_conc <= 0`.  Otherwise, a one-sided Hoeffding
bound gives the simultaneous radius

\[
 r=\sqrt{2\log(1/\delta_{conc})/m}.
\]

An independently proved initialization or nonstationarity bias is subtracted
separately.  The executable strict path now constructs this plan explicitly
and rejects an inadequate mixing bound.  The legacy `effective_samples`
interface remains available only as a generic theorem input and cannot be
used for C1 qualification.

## 3. Action-to-post-calibration transfer

Let `d_q(theta)` denote the conditional drift of action `q` at policy `theta`.
If a proved bound

\[
 |d_q(\theta)-d_q(\theta_0)|\le L_q\|\theta-\theta_0\|
\]

holds and the deployed epoch enforces
`||theta-theta0|| <= R_q`, then the deployable lower certificate is

\[
 \underline d_q^{deploy}
 =\widehat d_q-r_q-b_q-L_qR_q.
\]

The implementation now exposes and subtracts `L_q R_q`.  It does not estimate
`L_q` from the same outcomes.  For the current nonlinear MAPPO policies,
however, no usable analytic `L_q` bound has yet been established.  An observed
parameter displacement is not a substitute for this missing sensitivity
bound.

Consequently the current deep-MARL interface cannot honestly reuse a local
coefficient for a long post-calibration horizon.  A one-step, fully
recalibrated branch controller would avoid transfer, but its full catalogue
cost is likely to dominate the finite budget and must pass a static headroom
test before it is considered.

## 4. Hitting-time theorem audit

For `X_k=(V_k-v)_+`, if the deployable lower drift is valid uniformly until
the stopping time, the additive-drift expectation bound and its Markov tail
bound are correct.  Maximizing `N_B(q)[d_q]_+` minimizes that registered tail
upper bound because `V_0-v` is common across actions.  This result does not
prove the missing mixing or transfer premises; those are exactly the two
interfaces above.

## 5. Authorization status

CPU unit tests may continue.  A small HPC4 development experiment is
authorized only after one of the following is completed outcome-free:

1. a benchmark-specific analytic sensitivity bound `L_q` plus enforced trust
   region and registered expiry/recalibration rule; or
2. a fully charged one-step branch design whose static oracle headroom remains
   nontrivial after charging every catalogue branch and validation rollout.

Until then, running C1 would measure a heuristic and could not validate the
claimed Lyapunov certificate.  No frozen TSP manuscript or prior G0 artifact
is modified by this audit.

## 6. CPU verification

The TSP-V2 suite passes `20/20` tests, including nine focused additive-drift
tests.  A repository-wide collection in the `ust2` environment is blocked for
four legacy modules because `cvxpy` is absent.  With those four imports
excluded, `2097` tests pass and `7` skip; the remaining eight failures are all
legacy MinAtar smoke tests caused by the absent `minatar` package.  No failure
is in TSP-V2 or in a file changed by this audit.  The base environment is too
old to collect the repository because its NumPy/Python versions do not support
several existing type annotations; it is not used as the repository
regression environment.
