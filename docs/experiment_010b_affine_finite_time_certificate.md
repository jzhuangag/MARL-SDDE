# EXP-010B preregistration: affine Markov-TD finite-time certificate

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Objective

Test the newly proved affine extension of the predictably decorrelated delayed
mean-square theorem.  EXP-010A used a homogeneous contraction certificate but
selected actions with a stationary-noise surrogate.  EXP-010B changes only
that selection rule: every candidate is now scored by the finite-time upper
bound in Theorem 4, which allows conditionally biased and
Jacobian-correlated TD innovations.

This experiment asks whether the rigorous affine bound is:

1. numerically valid and nonvacuous on vector TD;
2. responsive to cross-agent correlation, temporal mixing, and delay;
3. empirically conservative on fresh paths;
4. informative enough to choose participation under a charged budget.

## Frozen model and resource design

Use the exact EXP-010A seven-state, four-feature MRP family without alteration:

- laziness \(\kappa\in\{0,0.9,0.98\}\);
- common-source co-use \(\rho\in\{0,0.9\}\);
- maximum delay label \(D\in\{0,8\}\);
- \(q\in\{1,2,4,8,16,32\}\);
- target mixing-margin fractions \(\{0.1,0.25,0.5,0.75\}\);
- resource budget \(B=128000\);
- cost per update \(4+q+b\);
- fastest-first deterministic delay profile.

The exact 49-state pair-chain TV calculation and the tensorized
\((q+1)\delta_{\rm pair}(b)\) joint bound are unchanged.

## Affine action certificate

For every candidate \((q,b)\), compute

\[
\begin{aligned}
a_\delta(\eta)
&=1-\eta\mu_\delta
  +\eta^2(2K_q+4L^2\delta),\\
\beta_\delta(\eta)
&=\frac{4\eta G^2\delta^2}{\mu_\delta}
  +\eta^2(2\Omega_q+4G^2\delta),\\
h_\tau(\eta)
&=2\eta^4L^4\tau_{\rm rms}^2,\\
g_\tau(\eta)
&=2\eta^4L^2G^2\tau_{\rm rms}^2.
\end{aligned}
\]

Use the contraction-minimizing Young parameter
\(\lambda_\star=\sqrt{h_\tau/a_\delta}\), with its continuous zero-delay
limit, to obtain \(c_{\rm aff}\) and \(d_{\rm aff}\).  The registered
finite-time score is

\[
\mathcal B_T
=R_\star+c_{\rm aff}^{\,n}(R_0-R_\star)_+,
\qquad
R_\star=\frac{d_{\rm aff}}{1-c_{\rm aff}},
\]

where \(n=\lfloor U/(2D_{\rm act}+1)\rfloor\).  Optimize the scalar step over
the first connected interval \(c_{\rm aff}(\eta)<1\).  Select the minimum
proved bound over all \(q,b,\eta\), and separately under \(q=1\) and \(q=32\)
restrictions.

The no-update bound is \(R_0=\|\theta^\star\|^2\).  A selected positive action
is called theorem-useful only if \(\mathcal B_T<R_0\).

## Simulation

- 32 new paired seeds beginning at 20261130;
- unit-time common and idiosyncratic chains shared across the three policies;
- exact charged thinning and messages;
- squared parameter error at the budget endpoint;
- no discarded seed, policy, or finite trajectory;
- 2,000 paired bootstrap replications;
- one-sided 99% bootstrap upper confidence limit for each policy/scenario
  mean, used only as an empirical calibration diagnostic.

## Preregistered gates

All three numerical gates and at least five of the following six scientific
gates are required for an overall pass.

### Numerical gates

1. **Certificate validity:** every selected positive action has
   \(a_\delta>0\), \(0<c_{\rm aff}<1\), finite forcing/residual/bound, and
   positive effective monotonicity.
2. **Execution validity:** all 1,152 policy runs are finite, none diverges,
   and every charged execution stays within budget.
3. **Algebraic reproduction:** direct evaluation of
   \((\sqrt{a_\delta}+\sqrt{h_\tau})^2\) agrees with the Young-expanded
   contraction to \(10^{-12}\) in every delayed candidate; zero-delay
   candidates agree with their continuous limit to \(10^{-12}\).

### Scientific gates

1. **Finite-time nonvacuity:** the joint selected bound beats no update in at
   least 8/12 scenarios, and every selected joint action completes at least 50
   updates.
2. **Correlation response:** \(q(0.9)\le q(0)\) in all six matched cells, with
   strict reduction in at least four.
3. **Mixing response:** the median selected gap at \(\kappa=0.98\) is at least
   four times its \(\kappa=0\) value.
4. **Delay response:** the selected delayed step is no larger than the
   zero-delay step in at least five of six matched cells.
5. **Empirical upper calibration:** the one-sided 99% bootstrap upper mean
   error does not exceed the proved finite-time bound in any of the 12 joint
   scenarios.
6. **Bound informativeness:** in at least 9/12 joint scenarios, the proved
   bound is no more than \(10^3\) times the observed mean error.

No gate, seed, candidate, threshold, model constant, or resource cost will be
changed after the primary run.

## Decision rule

- A pass closes the finite-gap affine Markov-TD theorem/experiment loop and
  promotes the proved finite-time bound into the main algorithm.
- If validity and calibration pass but nonvacuity/informativeness fail, retain
  Theorem 4 as a correctness result but do not use it as the practical
  controller objective.
- Any numerical or empirical-calibration failure triggers a proof or
  implementation audit before further experiments.

