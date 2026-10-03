# EXP-010A preregistration: multistate certificate transfer

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Objective

Test whether the proved decorrelated delayed mean-square certificate and its
low-complexity scalar controller transfer from the two-regime mechanism audit
to a genuinely vector-valued TD problem.  This experiment does not estimate
mixing online.  It gives the controller an exact finite-state mixing
certificate so that model transfer is not confounded with the finite-sample
estimation penalty isolated by EXP-009A--D.

The experiment asks:

1. is the theorem nonvacuous on a seven-state, four-feature TD problem?
2. does the certified action change participation with cross-agent
   correlation and change thinning/step size with temporal persistence and
   delay?
3. does joint certified selection improve on both certified endpoint
   participation rules under a fixed communication-and-observation budget?

## Frozen MRP family

Use the seven-state reward and four stationary-orthonormal features from
EXP-007A.  Replace its transition matrix by the circulant family

\[
P_\kappa=\kappa I+(1-\kappa)P_{\rm fast},
\]

where \(P_{\rm fast}\) moves clockwise or counter-clockwise with probability
0.475 each and restarts uniformly with probability 0.05.  The stationary
distribution is exactly uniform for every
\(\kappa\in\{0,0.9,0.98\}\).

The TD Jacobian for transition pair \((s,s')\) is

\[
H(s,s')=\phi(s)\{\phi(s)-\gamma\phi(s')\}^{\mathsf T},
\qquad \gamma=0.9.
\]

For each \(\kappa\), recompute the exact projected TD fixed point, mean
Jacobian, strong-monotonicity constant, Jacobian norm bound, pair-chain
transition matrix, and stationary TD-noise second moment.

## Dependence and delays

- cross-agent pair sharing \(\rho\in\{0,0.9\}\);
- candidate participation \(q\in\{1,2,4,8,16,32\}\);
- maximum delay label \(D\in\{0,8\}\);
- deterministic fastest-first delay profile from EXP-007A, constructed for
  32 agents and truncated to the participating prefix;
- total resource budget \(B=128000\);
- update cost \(4+q+b\), where \(b\) is the separation between retained
  transition pairs.

Every agent marginal has the exact stationary transition-pair law.  A common
pair chain and independent idiosyncratic pair chains are advanced by the same
registered separation \(b\).  Independent masks choose the common pair with
probability \(\sqrt{\rho}\), so two agents use the common source together with
probability \(\rho\).  Accidental equality between independent categorical
draws is not counted as common-source sharing.

## Exact certificate and controller

Let \(R_\kappa\) be the 49-state transition matrix of consecutive transition
pairs.  Compute

\[
\delta_{\rm pair}(b)
=\max_z\|R_\kappa^b(z,\cdot)-\pi_{\rm pair}\|_{\rm TV}
\]

exactly.  The joint common-plus-idiosyncratic source is certified by the
tensorization upper bound

\[
\delta_q(b)=\min\{1,(q+1)\delta_{\rm pair}(b)\}.
\]

For each candidate \(q\), search the smallest gaps associated with four
frozen fractions \(\{0.1,0.25,0.5,0.75\}\) of the admissible mixing margin
\(\mu/(2L)\).  At each candidate use the rate-optimal scalar step inside

\[
\sqrt{1-2\eta\mu_\delta+\eta^2K_\delta}
 \eta^2L^2\tau_{\rm rms}<1.
\]

Choose the action minimizing the theorem-inspired finite-budget surrogate

\[
c_{\rm sharp}^{\lfloor U/(2D_{\rm act}+1)\rfloor}
\frac{\eta^2\Omega_q}{1-c_{\rm sharp}},
\]

multiplied by the exact initial squared error in its transient term.  The
homogeneous contraction in this expression is certified.  Its additive
residual is a selection surrogate for TD: the simple residual theorem also
requires conditional centering and orthogonality, which finite-gap Markov TD
does not automatically satisfy.  No finite-sample TD error coverage claim is
made from this surrogate.  Here
\(U=\lfloor B/(4+q+b)\rfloor\), \(D_{\rm act}\) is the largest delay in the
participating prefix, and \(\Omega_q\) is the exact same-time stationary
TD-noise second moment under pair sharing.

Compare:

1. joint certified selection over all \(q\);
2. the same certified optimization restricted to \(q=1\);
3. the same certified optimization restricted to \(q=32\).

The exact 49-state eigencalculation is an offline audit oracle.  It is not
part of the proposed online algorithm.

## Simulation and statistics

- 32 fresh paired seeds beginning at 20261030;
- all three policies in a scenario use the same common and idiosyncratic
  random streams;
- endpoint is squared parameter error after the charged budget;
- no seed or finite trajectory is removed;
- paired bootstrap with 2,000 replications for joint versus each endpoint
  baseline in every scenario;
- report all selected actions, constants, per-seed errors, numerical
  invariants, and hashes.

## Preregistered gates

All numerical gates and at least four of the five scientific gates must pass.

### Numerical gates

1. every transition/pair-chain row sums to one, the uniform state
   distribution and exact pair distribution are stationary, and all
   monotonicity margins used by the controller are positive;
2. every selected action strictly satisfies the sharp certificate, every
   charged execution stays within budget, and all 1,152 policy runs are
   finite;
3. an independently iterated matrix-power calculation of pair-chain total
   variation at every selected gap agrees with the direct exact calculation
   to \(10^{-12}\).

### Scientific gates

1. **nonvacuity:** every selected action performs at least 50 server updates
   and uses a positive step;
2. **correlation response:** for each matched \((\kappa,D)\), selected
   \(q(\rho=0.9)\le q(\rho=0)\), with a strict decrease in at least four of
   six cells;
3. **mixing response:** median selected gap at \(\kappa=0.98\) is at least
   four times that at \(\kappa=0\);
4. **delay response:** at fixed \((\kappa,\rho)\), the selected \(D=8\)
   step is no larger than its \(D=0\) counterpart in at least five of six
   cells;
5. **endpoint value:** joint selection has lower mean error than both
   endpoint rules in at least eight of the twelve scenarios and is never more
   than 25% worse than the better endpoint.

The overall transfer passes only when all three numerical gates and at least
four scientific gates pass.  Thresholds, seeds, candidate actions, MRP,
resource model, and comparisons will not be changed after the primary run.

## Decision rule

- A pass promotes the certified homogeneous vector-TD stability rule and the
  empirically validated TD controller as the main linear experiment, while
  keeping online mixing estimation and affine Markov finite-time risk as
  separate uncertainty/proof obligations.
- If safety/nonvacuity passes but participation or endpoint value fails,
  retain the theorem and correlation-limited speedup result, and remove
  adaptive-control optimality from the main claim.
- Any numerical failure blocks use of the current theorem implementation.

## Pre-execution clarification

Before the primary simulation, implementation tests exposed two wording
ambiguities.  First, the additive finite-budget expression is a controller
selection surrogate, not a proved TD-risk upper bound without the extra
conditional-centering/orthogonality assumption already stated in the proof
program.  Second, the \(10^{-12}\) TV gate is an independent deterministic
matrix-power check, not a Monte Carlo estimate.  These clarifications change
no MRP, action, seed, threshold, controller, or scientific gate.
