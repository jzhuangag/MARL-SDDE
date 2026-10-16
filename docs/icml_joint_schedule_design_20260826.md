# ICML 2027 joint participation--gain schedule design

> **Status note (2026-08-26):** this offline schedule-library proposal is
> retained as design provenance and as a comparator. The recommended mainline
> is now the fully online joint controller in
> `docs/icml_online_joint_controller_design_20260826.md`, where offline work
> certifies a safe action region but does not select the executed `(q,eta)`.

## Decision

The current reward-free fingerprint controller is a strong and experimentally
supported special case, but a paper whose only adaptive output is one static
participation count `q` has a narrow algorithmic surface. The recommended
ICML mainline is therefore:

> use independent, fully charged trajectory fingerprints to choose a
> Lyapunov-certified finite-horizon training schedule that jointly controls
> participation and gain under delayed correlated Markov data.

The schedule, rather than an unrestricted online optimizer, is selected from
a small preregistered library. This preserves low online complexity and makes
the complete decision auditable.

No acceptance outcome can be guaranteed. The purpose of this redesign is to
remove the most obvious novelty and mechanism objections without claiming that
the existing q-only experiments already validate the extension.

## Core discrete model

For the fixed-policy TD / affine stochastic-approximation head, write the
delayed error recursion as

\[
e_{t+1}=e_t-\eta_t A e_{t-D}+\eta_t\bar\xi_{t+1}^{(q_t)}+r_t,
\]

where `r_t` collects the controlled Markov-bias terms. Under the registered
common/private trajectory-switch coupling,

\[
\operatorname{Cov}(\bar\xi_t^{(q)})
=g(q,\rho)\Sigma_t,
\qquad
g(q,\rho)=\rho+\frac{1-\rho}{q}.
\]

One synchronized round costs `h+q_t` message units and `q_t` actor
transitions. After a fully charged independent probe, a schedule is feasible
only if its learning rounds plus the delay pipeline satisfy both residual
budgets.

## The opposing mechanisms

### Participation: variance reduction versus usable horizon

- Increasing `q` suppresses only the private-noise term `(1-rho)/q`; the
  common component `rho` remains.
- Increasing `q` consumes more messages and actor transitions per round, so it
  reduces the number of contractions that fit in a fixed budget.

Thus large `q` is useful when dependence is low and variance is the dominant
error, but wasteful when dependence is high or the remaining horizon is the
dominant constraint.

### Gain: contraction versus noise and delay stability

- A larger `eta` removes initial bias faster.
- The same larger `eta` amplifies injected noise and shrinks the stability
  margin of the delayed recursion.

Thus an aggressive safe gain is useful in the early bias-dominated phase,
while a smaller gain is useful in the late variance-dominated phase. Delay
makes the aggressive region smaller.

These mechanisms interact. Early training may prefer small `q` and a larger
safe `eta` because extra rounds are valuable; late training may prefer a
smaller `eta` and, only when `rho` is low, a larger `q` to suppress the noise
floor.

## Recommended decision variable

A minimal nontrivial schedule is

\[
\pi=(q_{\rm burn},\eta_{\rm burn},T_{\rm switch},
      q_{\rm avg},\eta_{\rm avg}).
\]

The existing controller is recovered by setting the two q values equal, the
two gains equal, and removing the switch. A static joint pair `(q,eta)` should
be retained as an ablation, not used as the headline algorithm.

## What the Lyapunov function does

Lift the delayed state to

\[
z_t=(e_t,e_{t-1},\ldots,e_{t-D}).
\]

For every candidate gain, let `C_eta,D` be its deterministic lifted companion
matrix. Offline, search for a common positive-definite matrix `P` satisfying

\[
C_{\eta,D}^{\mathsf T}PC_{\eta,D}
\preceq (1-\alpha_\eta)P
\]

for all gains retained in the library. The quadratic function

\[
V_t=z_t^{\mathsf T}Pz_t
\]

is therefore a stability and finite-risk certificate, not an online tuning
variable. Its conditional drift has the schematic form

\[
\mathbb E[V_{t+1}-V_t\mid\mathcal F_t]
\le -\alpha_{\eta_t}V_t
 +c_P\eta_t^2 g(q_t,\rho)\sigma^2
 +\varepsilon_{\rm mix}(t).
\]

Summing the drift over a feasible candidate schedule yields a certified
finite-budget risk curve `U_pi(rho)`. Under the common/private coupling this
curve is affine in `rho` for a deterministic schedule once the Markov impulse
table is fixed.

The online fingerprint supplies an interval `I_rho`. The controller performs
the robust table lookup

\[
\widehat\pi\in\arg\min_{\pi\in\Pi}
\max_{\rho\in I_\rho} U_\pi(\rho).
\]

It does not solve an LMI, invert a Hessian, or estimate a covariance matrix
online.

## Role of the SDDE

The round-time diffusion interpretation is

\[
de(t)=-\eta(t)Ae(t-\tau)dt
 +\eta(t)\sqrt{g(q(t),\rho)\Sigma}\,dW(t).
\]

A Lyapunov--Krasovskii functional such as

\[
\mathcal V(t)=e(t)^{\mathsf T}Pe(t)
+\int_{t-\tau}^{t}e(s)^{\mathsf T}Re(s)\,ds
\]

explains how delay limits the admissible gain and how participation changes
the diffusion intensity. It is useful for constructing the gain library and
the phase geometry. The primary convergence theorem must nevertheless be the
executed discrete delayed Markov recursion. The SDDE becomes a theorem-level
corollary only after a proved interpolation/weak-limit statement; otherwise it
is an interpretation and design layer.

## Current versus proposed gain selection

The current MinAtar implementation does not use a finite gain catalogue. For
each task it computes

\[
\eta=\min\{\eta_{\max},c/\|\widehat A\|_2\},
\]

with `eta_max=0.05` and `c=0.20`, using an independent reference-moment
artifact, and checks the lifted spectral radius for every registered delay.
All three current tasks use `eta=0.05`. Therefore T-061A/T-063A/T-063B test
correlation-adaptive participation at a fixed pre-screened gain.

The proposed offline library construction is:

1. freeze a logarithmic gain grid and a small q catalogue;
2. use independent calibration moments or public analytic bounds to build the
   lifted matrices;
3. reject gains that do not share the required Lyapunov stability certificate
   over all registered delays;
4. enumerate only one-switch schedules and impose exact integer message and
   actor-transition budgets;
5. precompute each schedule's finite-risk coefficients and a strong fixed
   schedule baseline;
6. freeze the library, hashes, selector, seeds, and gates before generating
   controller outcomes.

The correlation observation remains reward-free. The full schedule library
should be called reward-free only if its calibration also avoids rewards;
otherwise the precise claim is reward-free adaptation signal.

## Required theorem upgrade

1. An exact or upper finite-horizon risk recursion for deterministic,
   piecewise-constant `(q_t,eta_t)` schedules with Markov lag covariance and
   delay.
2. A common-Lyapunov or certified-switching theorem establishing stability for
   every retained schedule.
3. A robust-selection excess-risk bound driven by the fingerprint confidence
   interval and the modulus of the schedule risk curves.
4. A lower bound or phase result showing when joint scheduling has strictly
   positive value over the best static `(q,eta)` action after probe cost.

## Evidence boundary and next gate

T-063A and T-063B provide reproducible positive evidence for the q-only special
case: about 17% aggregate improvement over the strong fixed-q comparator,
taskwise and delay-wise gains, and close true-rho oracle performance. Both
remain qualified formal failures because the preregistered bootstrap lower
bound for strict improved-cell breadth is below 0.60.

They do not validate gain adaptation or a two-stage schedule. Before a new
controller experiment, use CPU-only work to:

1. prove and test the scheduled finite-risk recursion and common certificate;
2. preregister a disjoint-selection fixed `(q,eta)` and one-switch schedule
   phase scan;
3. require a meaningful oracle ceiling over the best static pair, not merely
   over a weak fixed-q baseline;
4. proceed to a new formal experiment only if that prospective value gate
   passes.

GPU is unnecessary for the theorem audit and fixed-encoder CPU scan. A GPU is
only justified later for learned-representation external validity after the
joint-schedule mechanism clears its CPU gates.
