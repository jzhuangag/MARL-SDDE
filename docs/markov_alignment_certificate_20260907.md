## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theorem derivation and executable proof audit
- Origin Date: 2026-09-07
- Verification Status: FIXED-EVENT FINITE-STATE THEOREM PROVED; PURSUIT ESTIMATOR OPEN
- Version Label: markov_alignment_certificate_v1

# Markov alignment certificates for causal policy freshness

## Why this interface is needed

The joint controller in `joint_factor_lyapunov_theorem_20260907.md` requires a
launch-measurable lower bound

\[
A_p^L(a)\leq
\mathbb E[\langle \nabla_iF(\theta_{b(p)}),g_p(a)\rangle
\mid\mathcal F_{b(p)}].
\]

Calling a correlated-block root-mean-square estimate a confidence certificate
does not establish this inequality.  The score map must be fixed before its
holdout path is generated, Markov transients must be retained, and a cache
action can change the trajectory kernel.  This note supplies an exact
finite-state interface and records the remaining nonlinear gap.

## Fixed-event setup

At one certificate event, let `P_0` be the reference finite-state Markov
kernel and `P_a` the kernel induced by candidate cache action `a`.  Let
`f_a(z)` be a scalar alignment score fixed before a completed reference path
is generated.  Write

\[
R_a=\max_z f_a(z)-\min_z f_a(z).
\]

The reference score samples are `f_a(Z_0),...,f_a(Z_{m-1})`; the chain may
start from an arbitrary state.  A future candidate packet contains `H`
Markov transitions and starts from the observed launch state.  Any declared
critic, factorization, clipping, or off-policy discrepancy is collected in a
nonnegative additive term `epsilon_app(a)`.

For a scalar score `f` and kernel `P`, define its stationary mean
`pi_P f` and the centered Poisson solution

\[
h_P-P h_P=f-\pi_P f,
\qquad \pi_P h_P=0,
\]

with span `s_P(f)=max h_P-min h_P`.  These constants are exactly computable in
a finite model.  If the Dobrushin coefficient is at most `rho<1`, then

\[
s_P(f)\leq \frac{\operatorname{osc}(f)}{1-\rho}.
\tag{1}
\]

## Theorem 1: fixed-event holdout lower bound

Let `A` be the number of candidate actions and let `delta_p` be the failure
probability assigned to this event.  With probability at least
`1-delta_p`, simultaneously for all candidates,

\[
\begin{aligned}
\mathbb E_a\!\left[\frac1H\sum_{t=0}^{H-1}f_a(Y_t)
\middle|Y_0=y\right]
\geq{}& \bar f_{a,m}
-s_{0,a}\left[\frac1m+
\sqrt{\frac{\log(A/\delta_p)}{2m}}\right]\\
&-\frac{s_{a,a}}{H}
-R_a\,\operatorname{TV}(\pi_0,\pi_a)
-\epsilon_{\rm app}(a),
\end{aligned}
\tag{2}
\]

where `s_0,a` is the Poisson span of `f_a` under `P_0` and `s_a,a` is its
span under `P_a`.  The bound holds for every launch state `y`; stationarity of
the observed path is not assumed.

### Proof

The Poisson equation gives

\[
\sum_{t=0}^{m-1}[f_a(Z_t)-\pi_0f_a]
=h_0(Z_0)-h_0(Z_m)
+\sum_{t=0}^{m-1}[h_0(Z_{t+1})-P_0h_0(Z_t)].
\tag{3}
\]

The last sum is a martingale whose increment has conditional range at most
`s_0,a`; the boundary is at most `s_0,a`.  Conditional Hoeffding--Azuma and a
union bound over the finite action set therefore lower-bound `pi_0 f_a` by
the first line of (2).  Applying the Poisson identity in expectation to the
future `P_a` block gives

\[
\mathbb E_a[H^{-1}\sum_{t<H}f_a(Y_t)\mid Y_0=y]
\geq \pi_af_a-s_{a,a}/H.
\tag{4}
\]

Finally,

\[
|\pi_af_a-\pi_0f_a|
\leq R_a\operatorname{TV}(\pi_a,\pi_0).
\tag{5}
\]

Subtracting the declared approximation error proves (2).  No iid
substitution is used.

## Corollary 1: public contraction and policy-shift bounds

Suppose `rho_0,rho_a<1` upper-bound the Dobrushin coefficients and

\[
\sup_z\operatorname{TV}(P_0(z,\cdot),P_a(z,\cdot))\leq e_a.
\]

Stationary perturbation gives

\[
\operatorname{TV}(\pi_0,\pi_a)
\leq \min\{1,e_a/(1-\rho_a)\}.
\tag{6}
\]

Equations (1), (2), and (6) produce a fully explicit conservative bound.  In
a Markov game, data processing permits `e_a` to be upper-bounded by the
supremum total-variation change of the one-edge joint policy.  Thus the cache
action's occupancy shift is charged rather than silently ignored.

For infinitely many events, use

\[
\delta_p=\frac{6\delta}{\pi^2(p+1)^2}.
\tag{7}
\]

The probabilities sum to `delta`, so ordinary conditional union bounds make
all event statements simultaneous.  Equation (7) does not convert an
arbitrary fixed-event estimator into a time-uniform confidence sequence.

## Exact state-conditional model value

When `P_a` and `f_a` are known, the controller should not discard the launch
state.  The exact alignment is

\[
A_{p,H}(a,z)=\frac1H\sum_{t=0}^{H-1}e_z^\top P_a^t f_a.
\tag{8}
\]

This value can change sign with `z`; it is the correct interface for the
factor-switch separation.  Its implementation is
`finite_horizon_markov_alignment`.  Substituting (8), minus any valid
approximation radius, into the joint Lyapunov index preserves state-adaptive
graph value.

## Theorem 2: learned local finite-state model

The stationary limitation can be removed for a finite local state space.
For one candidate action, let `N_xy` be its transition counts and let `N_x`
be the corresponding row count.  Let `K_x` bounded score observations with
conditional mean `f_a(x)` have empirical mean `fhat_a(x)` and range at most
`R`.  Action selection and state visits may be adaptive, but every transition
and score observation must have the declared state--action conditional mean.

For any bounded variable with `n` observations, finite family size `M`, and
event error `delta_e`, define

\[
r(n,R,M,\delta_e)
=R\sqrt{\frac{\log(2Mn(n+1)/\delta_e)}{2n}}.
\tag{9}
\]

This follows by applying conditional Hoeffding at every count and allocating
failure probability proportional to `1/[n(n+1)]`.  Half of `delta_p` is
assigned to the `|A|S` score means and half to the `|A|S^2` transition
coordinates.  Hence, simultaneously over actions, states, successors, and
all observed positive counts,

\[
f_a(x)\geq \widehat f_a(x)-r(K_x,R,|A|S,\delta_p/2)-e_f,
\tag{10}
\]

and the true current transition row belongs to

\[
\left\{q:\|q-\widehat P_a(x,\cdot)\|_1
\leq \min\left(2,
S r(N_x,1,|A|S^2,\delta_p/2)+e_{P,x}\right)\right\}.
\tag{11}
\]

Here `e_f` is a uniform score-approximation error and `e_P,x` is an explicit
L1 drift allowance from the data-collection kernel to the current candidate
kernel.  Neither may be fitted on the same holdout outcomes without a separate
uniform-complexity argument.

Starting from `V_0=0`, perform rectangular robust dynamic programming,

\[
V_{h+1}(x)=f_a^L(x)+
\min_{q\in\mathcal P_{a,x}}q^\top V_h,
\qquad h=0,\ldots,H-1.
\tag{12}
\]

On the simultaneous confidence event, backward induction gives

\[
H^{-1}V_H(z)
\leq A_{p,H}(a,z).
\tag{13}
\]

Thus (13) is a valid state-conditional choice for `A_p^L(a)` in the joint
Lyapunov controller.  The inner minimization in (12) is solved exactly by
moving at most half the L1 radius from the largest-value states to the
smallest-value states.  The cost per candidate is `O(HS^2)` with this simple
implementation and remains local rather than scaling with the total number
of agents.

### Proof

Conditional Hoeffding applies to the bounded martingale-difference sequence
formed by departures from every visited state--action pair; global iid state
sampling is not required.  Unioning the count-indexed bounds proves (10) and
(11).  If `V_h` lower-bounds the remaining `h`-step return, the true row lies
in `P_a,x`, so its conditional continuation value is no smaller than the
minimum in (12).  Adding the immediate score lower bound proves the induction;
division by `H` proves (13).

If any required state has zero score or transition count, the executable
certificate returns `-infinity`.  This correctly makes the packet-weight
shield reject an uncertified action.  It also exposes rather than solves the
exploration problem: learning all action models may require explicitly
charged exploration, a structural generalization model, or a prior dataset.

## What is and is not closed

The following items are now exact:

1. the finite-state Poisson constants and arbitrary-start Markov
   concentration;
2. the target-packet transient term;
3. the stationary-law shift caused by a cache action;
4. a summable launch-event failure allocation;
5. the exact state-conditional value when the local Markov model is known;
6. a count-uniform state-conditional tabular lower bound with robust dynamic
   programming when score and transition feedback for a candidate are
   available.

The stationary bridge in (2) can nevertheless be vacuous in the most
interesting switching games.  For example, the two edge scores in the exact
factor-switch construction have zero stationary mean even though the observed
state selects a strictly beneficial edge.  Consequently, (2) is a safe
fallback certificate, not by itself the final adaptive estimator.

Theorem 2 closes the state-conditioned interface for an explicit tabular local
factor.  It does not automatically certify a neural Pursuit critic.  That
extension must either provide a state-conditional approximation radius or
label the critic as an empirical estimator.  Reusing the same path to fit and
certify an unrestricted neural score remains unauthorized.

For Pursuit, the structural factor audit alone does not supply either radius.
No efficacy pilot or GPU benchmark is authorized by this result.

## Executable verification

`markov_alignment_certificate.py` implements (2), (6)--(13).  Tests
verify the canonical two-state Poisson span, exact state dependence, the
stationary perturbation inequality, summability, fail-closed input handling,
and fixed-event coverage by complete path enumeration.  They also compare the
simplex minimizer with a dense grid, verify robust dynamic programming against
a grid of admissible kernels, check contraction of the count-uniform radius,
and require zero-count actions to fail closed.  The Dobrushin form is checked
to be no less conservative than the exact-span form on a finite example.
