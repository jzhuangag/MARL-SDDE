# T-067 research decision: endogenous phase control

## Mainline decision

The stationary version of the joint-control claim is too broad.  T-066A
shows substantial cellwise adaptation value relative to strong fixed pairs,
but broad strict separation from both q-only and eta-only policies fails.  The
next ICML-facing question is therefore not "can two hyperparameters be tuned
together?"  It is:

> Can a low-complexity predictable controller track the endogenous transition
> from contraction-dominated learning to correlation-limited noise-floor
> learning by jointly controlling data acquisition and update gain under
> delay and two hard budgets?

This formulation is scientifically stronger and more falsifiable.  It
identifies a state transition that fixed and one-dimensional policies may be
unable to track, rather than claiming novelty from the number of controls.

## Phase variable

The observable paired-block statistics estimate a dimensionless
signal-to-noise state

\[
\chi_t=\frac{\|F(\theta_t)\|^2}
{\operatorname{tr}(\Sigma_t)+\epsilon}.
\]

Large `chi_t` corresponds to contraction-dominated learning: reducing bias is
more important than averaging noise.  Small `chi_t` corresponds to the noise
floor: common/private dependence, participation cost, and gain-induced noise
become decisive.  The transition is endogenous because `F(theta_t)` shrinks
as learning progresses even when the task law is stationary.

The controller does not need to estimate parameter error or `theta_star`.
Disjoint residual cross-products estimate the numerator, paired differences
estimate the denominator, and fingerprints estimate the cross-agent common
fraction.  Confidence bounds, rather than clipped point estimates, must feed
the theorem-facing action.

## Role of the discrete Lyapunov function

For each delay class, the common certificate

\[
V_t=z_t^\top P_D z_t,
\qquad z_t=(e_t,\ldots,e_{t-D}),
\]

serves three purposes:

1. it certifies every online gain in a continuous safe interval;
2. it converts the observable mean-field and noise bounds into a robust
   one-block drift score;
3. together with residual-budget queues/shields, it prices the future learning
   opportunity consumed by participation and sensing.

It does not select `q` by itself.  Its drift inequality supplies the public
coefficients of the exact T-064 joint minimizer.  SDDE is unnecessary for all
three purposes and is not in the primary proof dependency graph.

## Required separation theorem

Construct a stationary delayed affine Markov family whose learning trajectory
crosses two separated `chi` phases.  Under identical message/environment
budgets and sensing charges, prove:

\[
R(\pi_{q,\eta}^{\rm joint})
\le R(\pi_q)-\Delta_q,
\qquad
R(\pi_{q,\eta}^{\rm joint})
\le R(\pi_\eta)-\Delta_\eta,
\]

for positive explicit gaps, where `pi_q` may adapt participation with its gain
schedule restricted to the registered one-dimensional class and `pi_eta` may
adapt gain with its participation schedule similarly restricted.  Comparator
classes must be strong enough that the result is not created by fixing a bad
constant.

The proof should first use an exact two-phase scalar construction, then lift
to strongly monotone vector SA with the common delayed certificate.  A claim
against only fixed q or fixed eta is insufficient.

## Outcome-free static gate before sampling

A new experiment identifier may be preregistered only after an exact
finite-risk schedule scan is frozen.  It must:

- use a single stationary task law; the phase change comes from shrinking
  mean-field signal, not a hidden exogenous regime switch;
- charge a fresh paired sensor at every allowed decision boundary;
- allocate both resources per block before observing downstream outcomes;
- compare the best two-block joint schedule with strong fixed schedules,
  adaptive-q schedules over all registered gain schedules, and adaptive-eta
  schedules over all registered participation schedules;
- show at least 10% aggregate finite-risk improvement over the strong fixed
  schedule, strict cell improvement in at least 60%, and strict separation
  from both one-dimensional schedule classes in at least 30%;
- pass common-certificate, action diversity, dual-budget, taint, and clean-
  reproduction gates.

These thresholds are prospective design requirements, not reinterpretations
of T-066A.  T-066A remains failed on S5.

## Paper-facing experiment ladder

If the exact dynamic gate passes:

1. run a sampled affine Markov-TD CPU pilot with actual residual/fingerprint
   blocks and pathwise budgets;
2. freeze new confirmation seeds and test on standard stochastic fixed-policy
   tasks with public features, using best task-by-budget fixed schedules,
   q-only, eta-only, correlation-ignorant adaptive gain/batch, and a
   clairvoyant schedule oracle;
3. report terminal prediction risk, normalized learning AUC, CVaR90, improved-
   cell breadth, phase-switch timing, budget utilization, sensing opportunity
   cost, and controller overhead;
4. use GPU only for a later learned-representation external-validity test, not
   for rescuing a failed theorem-aligned CPU experiment.

## ICML status

The current evidence is not yet an ICML-ready paper: there is no dynamic
separation theorem, confidence-to-control excess-risk bound, or positive
standard-RL joint-controller result.  The discrete certificate, exact online
optimizer, positive T-065A mechanism pilot, and T-066A phase diagnosis form a
credible research program.  ICML viability now depends on the dynamic phase
gates above, not on adding SDDE terminology or selecting a favorable static
benchmark.
