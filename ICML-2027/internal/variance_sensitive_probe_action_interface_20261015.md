# Conditional variance-sensitive probe/action design interface

## Material Passport

Original mathematical interface, not a completed async-MARL algorithm or
performance theorem. Complements the native estimator qualifications without
changing any of their gates. No new experimental outcome is used here.

## A design variable beyond the policy step

For a frozen eligible candidate family, include the reference action a=0 with
relative return mu_0=0. Let c_a be *future* resource cost, not work already paid
to construct or evaluate a candidate. Conditional on current history, collect
m iid shared-reference episodes and form unbiased relative-return estimates.
Let Sigma be the episode-level covariance of those contrast estimates and
S=trace(Sigma). Cost charges are exact. The sampling budget m is chosen before
opening those episodes; the candidate is selected afterward.

For a nonnegative resource price vector Q and learning-value weight V>0,
choose the eligible candidate maximizing `V*mu_hat_a - Q dot c_a`.
Writing e_a=mu_hat_a-mu_a, its conditional true-utility loss to the best
eligible candidate is at most `2V max_a |e_a|` pointwise. Consequently,

`E[utility_loss | history] <= 2V sqrt(S/m)`

and

`E[utility_loss^2 | history] <= 4 V^2 S/m`.

Shared candidates need not be independent: the samples are iid *episodes*,
each returning a vector of contrasts, and the trace is the sum of their
marginal variances. This result needs finite second moments, not a claim
that a realized positive score is a simultaneous lower confidence bound.
It is a standard estimation-to-selection inequality, not a novelty claim.

## Jointly pricing how much to measure and which action to choose

Suppose an independently justified predictable bound S<=Sbar is available,
and a shared-bank episode costs resource vector kappa. Let c_start be the
fully charged fitting/setup/candidate-construction cost. A conditional
drift-plus-penalty upper bound has an estimation-related term

`Q dot (c_start + kappa*m) + 2V sqrt(Sbar/m)`.

Thus the real-valued sampling solution is

`m_star = (V sqrt(Sbar) / (Q dot kappa))^(2/3)`

when `Q dot kappa>0`; choose the resource cap when its denominator is zero.
Compare the floor/ceiling integers and the feasible endpoints; never exceed
the remaining environment, optimizer or message budgets. Then select the
candidate using the measured utility. Lyapunov prices would participate in
both the observation budget m and the candidate decision, rather than merely
certifying a predetermined step-size schedule.

This is useful only if the best candidate's surplus over a no-probe reference
can pay for `Q dot c_start`, the shared-bank bill, and the estimation loss.
Calibrating a score at no charge is not an eligible comparison. A more precise
estimator can lower the bound through S, but a ninefold sampling saving alone
does not imply a higher final learning return.

## What is and is not proved

1. The selector inequality is proved conditionally for frozen, known targets
   and known future action costs.
2. The sampling formula minimizes its stated conditional upper bound under
   a **predictable population variance upper bound**. A sample trace estimate
   is unbiased for S but is not such a bound. Plugging it in is a heuristic
   unless an independent variance-bound argument is supplied.
3. This does not yet prove virtual-queue stability, finite-time stationarity,
   or return dominance against a separate HAPPO learning trajectory. The
   comparator is a best eligible action *at the current state*, not an oracle
   trained along its own different state sequence.
4. A frozen candidate produced after computing a gradient has sunk compute
   cost. This interface alone cannot decide which unknown future packets to
   launch. Launch/receipt timing and action-dependent in-flight value remain
   separate mandatory causal interfaces.
5. Continuous Markov data require a justified bias/dependence extension;
   the qualifications use independent native episodes.

The next theory decision is whether a resource-feasible variance bound and a
useful causal action class can instantiate this interface. It is not valid to
rename a vacuous high-probability certificate as this expected-risk result
and silently preserve a per-step no-harm claim. The guarantee changes and
must be stated explicitly if this design is pursued.
