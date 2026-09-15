# Conformal certificate for a state-dependent policy-influence cone

Date: 2026-09-06

Status: finite-sample marginal tube certificate and policy-shift correction
proved.  No benchmark-return experiment is authorized here.

## 1. Observable residual

At launch, let `x` contain the centralized training state needed by a public
trajectory predictor.  For a rollout horizon `H`, the predictor outputs a
space--time tube `T_r(x,H)` expanded by radius `r`.  Let

\[
R(\tau,x)=\inf\{r:\tau\subseteq T_r(x,H)\}
\tag{1}
\]

be the smallest expansion containing the realized interaction path.  For
Pistonball, the center line uses launch ball position and horizontal velocity;
the residual is the maximum ball-path deviation in piston-width units, and a
declared contact radius is added after calibration.  The residual uses states,
not rewards or post hoc controller performance.

## 2. Split-conformal radius

Let `R_1,...,R_n,R_{n+1}` be exchangeable residuals under a fixed reference
policy and launch-state stratum.  For target miscoverage `alpha`, define

\[
k=\lceil(n+1)(1-\alpha)\rceil.
\tag{2}
\]

If `k<=n`, let `q_alpha` be the `k`th order statistic of the calibration
residuals.  Uniformity of the new residual's rank gives

\[
\mathbb P\{R_{n+1}\le q_\alpha\}\ge \frac{k}{n+1}\ge1-\alpha.
\tag{3}
\]

If `k=n+1`, the implementation returns infinity.  It never substitutes the
sample maximum and calls it a finite-sample certificate.  The guarantee is
marginal over a new exchangeable rollout; it is not a simultaneous pathwise
guarantee for every future policy update.

## 3. Correction for policy drift

Let `P_0` be the reference trajectory law and `P_theta` the law at launch.
For any tube event `E`, total variation gives

\[
P_\theta(E^c)\le P_0(E^c)+\|P_\theta-P_0\|_{TV}.
\tag{4}
\]

Suppose the executed stochastic policies are transformed Gaussians with the
same standard deviation `sigma` and the joint pre-transform mean changes by at
most `epsilon_theta` in Euclidean norm at every visited state.  The chain rule
for trajectory relative entropy and invariance under the `tanh` bijection give

\[
\mathrm{KL}(P_\theta\|P_0)
\le \frac{H\epsilon_\theta^2}{2\sigma^2}.
\tag{5}
\]

Pinsker's inequality therefore yields

\[
\delta_{\rm cone}(\theta)
\le
\min\left\{1,\alpha+\frac{\sqrt H\epsilon_\theta}{2\sigma}\right\}.
\tag{6}
\]

Equation (6) is conservative but observable when the server records policy
versions and the actor has a certified output-Lipschitz constant.  If the
right-hand side is one, the cone certificate is vacuous and the scheduler
must fall back to the full prospective support; it may not claim safety from
the reference calibration set.

## 4. Entry into the Lyapunov bound

The causal-cone coupling lemma gives, for owner `i`,

\[
\varepsilon_{i,k}^{cone}
=4B_RB_{Z,i}\delta_{cone}(\theta^k).
\tag{7}
\]

This term is included once in the noisy drift error or the comparator
remainder.  Increasing the tube radius changes the empirical calibration
miscoverage and communication cost in opposite directions.  The dispatch
decision minimizes the certified drift-plus-penalty bound over nested tube
shells; because shells are nested, it can add positive net-benefit shells in
order rather than solve a dense graph program.

## 5. Scope and next empirical gate

The theorem applies directly to fixed-policy exchangeable calibration.  During
learning it additionally requires the policy-shift bound in (5), or a new
calibration epoch.  It does not justify reusing an arbitrary old replay buffer
after large policy changes.

Before a controller pilot, a fresh CPU experiment must independently freeze:

1. launch-state strata and stochastic reference policies;
2. calibration and holdout trajectory seeds;
3. target miscoverage and the conformal rank;
4. contact expansion and edge-cost accounting;
5. holdout coverage, nonvacuity, state adaptivity, and reproduction gates.

No reward comparison, GPU job, or controller tuning belongs to that
certificate experiment.
