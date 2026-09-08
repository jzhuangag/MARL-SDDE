## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theorem--implementation interface audit
- Origin Date: 2026-09-08
- Verification Status: ALGEBRA AND COMPLEXITY VERIFIED; STATISTICAL CRITIC GATE OPEN
- Version Label: pursuit_compatible_factor_interface_v1

# Compatible pair-factor alignment for Pursuit

## Purpose and boundary

The nonlinear controller must score a null cache profile and every eligible
one-edge teammate refresh without discarding the launch-time learning
direction or paying one actor reverse pass per edge.  Generic regression from
handcrafted launch features failed the conditional-mean test.  The replacement
uses the centralized factor critic already permitted by CTDE.

This note closes only the algebraic and computational interface.  It does not
claim that a learned factor critic is calibrated, that the Pursuit controller
improves return, or that a GPU experiment is authorized.

## Factor model and signed learning value

At a launch for owner `i`, let `u_i` be its discrete action and let the local
candidate universe contain donors `N_i`.  The centralized critic approximation
is

\[
 \widehat Q_i^{a}(u_i)
 =\widehat q_i(o_i,u_i)
 +\sum_{j\in\mathcal N_i}\sum_{u_j}
 p_{j}^{a}(u_j\mid o_j)
 \widehat q_{ij}(o_i,o_j,u_i,u_j),
\]

where `p_j^a` is the recipient's cached donor distribution except that action
`a=j->i` replaces donor `j` by its current distribution.  The null action
changes no cache.  All quantities are evaluated from launch-measurable policy
versions and a critic trained only on packets completed before the declared
critic epoch.

Let `v_(i,p)` be the launch-time reference direction.  Define once for every
owner action

\[
 d_i(u_i)=\left\langle v_{i,p},
 \nabla_{\theta_i}\pi_i(u_i\mid o_i)\right\rangle .
\]

Then the critic-derived conditional alignment is

\[
 \widehat A_p(a)=\sum_{u_i}d_i(u_i)\widehat Q_i^a(u_i).
\]

For the null profile set

\[
 z_i^{0}=\widehat q_i+
 \sum_{j\in\mathcal N_i}\widehat Q_{ij}p_j^{\rm cache}.
\]

One refresh has the exact rank-one candidate difference

\[
 z_i^{j}=z_i^0+widehat Q_{ij}
 (p_j^{\rm current}-p_j^{\rm cache}),\qquad
 \widehat A_p(j\!\to\!i)=d_i^\top z_i^j .
\]

This is not a Taylor approximation in the policy cache: it is an exact
evaluation of the declared pair-factor critic under the two categorical donor
distributions.  Approximation enters only through the critic factorization and
estimation error.

## Exact relationship to the Lyapunov action

For every candidate `a`, a simultaneous lower confidence value has the form

\[
 \underline A_p(a)=\widehat A_p(a)-\epsilon_{p}^{\rm stat}(a)
 -\epsilon_{p}^{\rm factor}(a).
\]

Substitution into the topology-robust core index gives

\[
 \widehat\alpha_p(a)=\Pi_{[0,\bar\alpha]}
 \frac{V[\underline A_p(a)-L_iG_p(a)M_p]}
 {VL_iG_p(a)^2},
\]

\[
 (a_p,\alpha_p)\in\arg\min_{a,\alpha}
 \left\{-V[\underline A_p(a)-L_iG_p(a)M_p]\alpha
 +\frac{VL_iG_p(a)^2}{2}\alpha^2+Q_pc_p(a)\right\}.
\]

Thus the factor critic does not independently choose a graph.  It supplies the
signed conditional learning term to the same Lyapunov minimization that jointly
chooses the directed refresh edge and delayed packet weight.  Communication
price, predicted descent, receipt motion, and packet curvature remain in one
decision rule.

If

\[
 \sup_a |A_p(a)-\widehat A_p(a)|
 \le \epsilon_p^{\rm stat}+\epsilon_p^{\rm factor},
\]

the existing finite-action optimistic-action lemma contributes at most twice
this uniform error to the paired launch--receipt regret term.  No cancellation
over graph changes is used.  The optional persistent cache-energy extension
still requires a fixed edge universe or the explicit topology-motion
remainder.

## Complexity

For `A` owner actions, computing all `d_i(u_i)` uses exactly `A` reverse-mode
calls, once per launch and independently of local degree.  Forming the null
profile and all one-edge differences uses

\[
 O(A\,|\theta_i|+\Delta_i A^2)
\]

arithmetic and `O(Delta_i A^2)` factor storage.  Pursuit has `A=5` and a
degree-four causal cone, so the candidate scan is `O(Delta_i)` after the five
owner probability-direction products.  A dense game is not claimed to have
degree-independent cost.

The implementation regression oracle explicitly enumerates all neighbor joint
actions.  The factor scan matches that exponential enumeration to numerical
precision, while its recorded reverse-pass count remains five at degrees one,
two, and four.  A centered finite-difference test independently confirms that
the returned alignment is the directional derivative of the factor objective.

## Statistical obligation before an efficacy experiment

The next gate is not another feature search.  It is to learn the pair factors
from fully charged, completed Pursuit packets with a predictable sample split:

1. fit the factor representation on representation episodes;
2. freeze it and fit the declared factor heads on disjoint head episodes;
3. estimate `epsilon_stat` and `epsilon_factor` from independent calibration
   episodes with replicated conditional-mean counterfactuals used only for the
   offline audit;
4. evaluate simultaneous mean coverage, signed candidate ranking, interval
   width, byte charging, delayed causality, and measured runtime on untouched
   test episodes.

The online method never observes counterfactual branches.  A positive offline
interface gate must show that the certified edge ordering is informative and
nonvacuous; coverage alone is insufficient.  Only then may an equal-resource
strong-online oracle-headroom study be frozen, followed by a separately
preregistered Pursuit learning pilot.

## Current decision

The compatible pair-factor formula is the only active nonlinear estimator
interface.  Its algebra and local-degree complexity pass.  Critic calibration,
equal-resource headroom, controller efficacy, independent confirmation, and
standard-MARL GPU evidence remain open.  No GPU or HPC4 run is authorized by
this result.
