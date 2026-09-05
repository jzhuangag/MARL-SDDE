# Cross-agent gradient transport proof program

Status: conditional event-time theorem derived; Markov-to-wall-clock and strong-comparator separation remain open.

## 1. Episodic cooperative Markov-game model

Consider an identical-interest episodic Markov game with joint factorized policy

\[
\pi_\theta(a\mid o)=\prod_{i=1}^n\pi_{i,\theta_i}(a_i\mid o_i),
\qquad
\theta=(\theta_1,\ldots,\theta_n)\in\mathbb R^d,
\]

and common discounted return `J(theta)`.
Training uses independent reset environment replicas.
Execution is decentralized; only the centralized trainer performs the operations below.

At proposal birth, worker `p` receives snapshot `theta^{b_p}` and a fixed-horizon joint rollout batch.
It eventually returns a stochastic block gradient for one actor.
Completion time is conditionally independent of the rollout values; fixed-horizon or padded rollouts satisfy this interface.
Let

\[
v_{p,k}=\theta_k-\theta^{b_p}
\]

be the observed joint displacement when proposal `p` completes.
The retained birth batch supplies both `hat g_p^b` and a Hessian-vector product `hat H_{p,:}^b v_{p,k}`.

## 2. Assumptions

All assumptions are local to a public trust region `Theta_0` containing every accepted line segment.

**Assumption A (objective regularity).**

(a) `J` is differentiable and upper bounded by `J_star` on `Theta_0`.

(b) Every block is `L_i`-smooth:

\[
J(\theta+U_i u)
\ge J(\theta)+\langle\nabla_iJ(\theta),u\rangle
-\frac{L_i}{2}\|u\|^2.
\]

(c) The block Hessian row is `rho_i`-Lipschitz:

\[
\|\nabla^2_{i,:}J(\theta')-\nabla^2_{i,:}J(\theta)\|_{\rm op}
\le\rho_i\|\theta'-\theta\|.
\]

(d) Block gradients are bounded by `G`.

**Assumption B (proposal confidence event).**

For the first `K` completed proposals, simultaneously,

\[
\|\widehat g_i^b-\nabla_iJ(\theta^b)\|\le r_i^g,
\]

and, for the predictable displacement used by the proposal,

\[
\|(\widehat H_{i,:}^b-\nabla^2_{i,:}J(\theta^b))v\|
\le r_i^H\|v\|.
\]

The event is established in Section 7 for independent bounded episodic batches.

**Assumption C (asynchronous population).**

(a) At most `P` proposals are in flight.

(b) Every policy block completes at least once in every `D` completion events.

(c) The path of an in-flight proposal is at most `E`, and every public step is at most `alpha_bar`.

(d) Applying or discarding a completion immediately births its replacement proposal at the current policy.

These are theorem conditions, not conclusions silently inferred from arbitrary stragglers.

## 3. Transport coverage

Define

\[
\widetilde g_{p,k}
=\widehat g_p^b+\widehat H_{p,:}^b v_{p,k}
\]

and

Let `ell_{p,k}` be the cumulative accepted joint-policy path length since proposal `p` was born.
Because `||v_{p,k}||<=ell_{p,k}`, use the monotone executable radius

\[
R_{p,k}
=r_p^g+r_p^H\ell_{p,k}
+\frac{\rho_p}{2}\ell_{p,k}^2.
\tag{1}
\]

### Lemma 1 (transport radius)

Under Assumptions A and B,

\[
\|\widetilde g_{p,k}-\nabla_pJ(\theta_k)\|
\le R_{p,k}.
\tag{2}
\]

### Proof

Taylor's integral formula gives

\[
\nabla_pJ(\theta^b+v)
=\nabla_pJ(\theta^b)+\nabla^2_{p,:}J(\theta^b)v+\varepsilon_p(v),
\]

where

\[
\|\varepsilon_p(v)\|
\le\int_0^1\rho_p t\|v\|^2dt
=\frac{\rho_p}{2}\|v\|^2.
\]

Add and subtract the population gradient and population Hessian-vector product.
The triangle inequality, Assumption B and `||v||<=ell` yield (2).

The corresponding untransported radius is `r_p^g+L_p^{joint}||v||`.
Thus exact Hessian-vector transport changes the path term from first to second order.

## 4. One-event gain and future-radius envelope

For a completed block `i`, set `s_i=||tilde g_i||`.
If the server applies `u_i=alpha tilde g_i`, block smoothness and (2) give

\[
J(\theta+U_i u_i)-J(\theta)
\ge
\alpha(s_i^2-R_i s_i)-\frac{L_i}{2}\alpha^2s_i^2
=G_i(\alpha).
\tag{3}
\]

For another in-flight proposal `p`, let `Z_p=R_p(ell_p)`.
Adding a candidate displacement of length `alpha s_i` to its cumulative path and applying (1) gives

\[
R_p(v_p+U_i u_i)-R_p(v_p)
\le
(r_p^H+\rho_p\ell_p)\alpha s_i
+\frac{\rho_p}{2}\alpha^2s_i^2
=a_{p,i}(\alpha).
\tag{4}
\]

This bound retains direction-independent validity while charging only the second-order transport remainder when `r_p^H=0`.

## 5. Composite Lyapunov drift and executable step

For positive weights `w_p`, define

\[
\mathcal L_k
=V(J^\star-J(\theta_k))
+\frac12\sum_{p\in\mathcal F_k}w_pZ_{p,k}^2.
\tag{5}
\]

Let proposal `i` complete at event `k`.
Its replacement enters with base radius `r_{i,k+1}^g`; write

\[
B_{i,k}=\frac12w_i(r_{i,k+1}^g)^2.
\]

Combining (3)-(5) gives

\[
\mathcal L_{k+1}-\mathcal L_k
\le B_{i,k}-\frac12w_iZ_{i,k}^2+F_{i,k}(\alpha),
\tag{6}
\]

where

\[
F_{i,k}(\alpha)
=-VG_i(\alpha)
+\frac12\sum_{p\ne i}w_p
\big[(Z_{p,k}+a_{p,i}(\alpha))^2-Z_{p,k}^2\big].
\tag{7}
\]

The executable step is

\[
\alpha_k\in\arg\min_{0\le\alpha\le\bar\alpha_i}F_{i,k}(\alpha).
\tag{8}
\]

### Lemma 2 (continuous low-complexity update)

The function in (7) is convex on the nonnegative real line.
If its derivative at zero is nonnegative, (8) returns zero.
If its derivative at the public cap is nonpositive, (8) returns the cap.
Otherwise its unique stationary point is found by scalar bisection.

### Proof

Write `a_{p,i}(alpha)=c_p alpha+d_p alpha^2`, with `c_p,d_p>=0`.
The negative certified gain is a convex quadratic.
Each remaining term is one half the difference between `(Z_p+c_p alpha+d_p alpha^2)^2` and `Z_p^2`.
Its second derivative is

\[
w_p\big[(c_p+2d_p\alpha)^2+2d_p(Z_p+c_p\alpha+d_p\alpha^2)\big]\ge0.
\]

Therefore the derivative of (7) is monotone.
Computing its coefficients costs `O(P)` and bisection costs `O(log(1/epsilon))` scalar iterations.

## 6. Conditional event-time performance theorem

Define the marginal transported residual

\[
q_{i,k}
=\left[
s_{i,k}-R_{i,k}
-\frac1V\sum_{p\ne i}w_pZ_{p,k}(r_p^H+\rho_p\ell_{p,k})
\right]_+.
\tag{9}
\]

The derivative of (7) at zero is `-V s_{i,k}q_{i,k}`.

Assume in addition that the second derivative of (7) is at most

\[
M_0s_{i,k}^2
\]

on `[0,alpha_bar]`, for a public finite `M_0`, and that `alpha_bar>=V/M_0`.
To make this condition explicit, define

\[
r_{\max}=r_{g,\max}+r_{H,\max}E+\frac{\rho_{\max}}2E^2,
\]

\[
h_{\max}=r_{H,\max}+\rho_{\max}E,
\qquad
a_{\max}=h_{\max}\bar\alpha G
+\frac{\rho_{\max}}2\bar\alpha^2G^2.
\]

If `w_p<=w_max` and at most `P-1` other proposals are in flight, the public choice

\[
M_0
=VL_{\max}
+(P-1)w_{\max}
\left[
(h_{\max}+\rho_{\max}\bar\alpha G)^2
+\rho_{\max}(r_{\max}+a_{\max})
\right]
\tag{9a}
\]

satisfies `F''(alpha)<=M_0s^2`.
Because `M_0>=VL_max`, the required cap `V/M_0` is no larger than `1/L_max` and is therefore nonvacuous on a standard block-smoothness trust interval.

### Theorem 1 (transport-residual and completed-radius bound)

On the simultaneous confidence event, the update (8) satisfies

\[
\sum_{k=0}^{K-1}
\left[
\frac12w_{I_k}Z_{I_k,k}^2
+\frac{V^2}{2M_0}q_{I_k,k}^2
\right]
\le
\mathcal L_0+\sum_{k=0}^{K-1}B_{I_k,k}.
\tag{10}
\]

### Proof

If `q=0`, convex minimization gives `F(alpha_k)<=F(0)=0`.
If `q>0`, smooth convexity of `F` and the cap condition allow the comparison step

\[
\alpha^{\rm cmp}=\frac{Vq}{M_0s}\le\frac{V}{M_0}\le\bar\alpha.
\]

The quadratic upper model at zero gives

\[
F(\alpha_k)
\le F(\alpha^{\rm cmp})
\le-\frac{V^2q^2}{2M_0}.
\]

Substitute this inequality in (6), telescope, and use `mathcal L_K>=0`.

Equation (10) is a real finite-time Lyapunov guarantee.
It is stronger than observing that the drift formula is negative at one event, but it is not yet the final joint-gradient theorem.

### Corollary 1 (conditional full joint-gradient stationarity)

Let `w_min=min_p w_p>0` and

\[
H_w=\max_p w_p(r_p^H+\rho_pE).
\]

Assume each in-flight proposal present before event `K` completes by event `K+D`; equivalently, evaluate the bound on the `D`-event completion closure of the horizon.
Monotonicity of (1) and the completion-gap condition give

\[
\sum_{k=0}^{K-1}\sum_{p\in\mathcal F_k}Z_{p,k}^2
\le
D\sum_{k=0}^{K+D-1}Z_{I_k,k}^2.
\tag{11}
\]

Let `A_{K+D}=mathcal L_0+sum_{k=0}^{K+D-1}B_{I_k,k}`.
From (9), transport coverage and `(a+b+c)^2<=3(a^2+b^2+c^2)`,

\[
\sum_{k=0}^{K-1}
\|\nabla_{I_k}J(\theta_k)\|^2
\le
\left[
\frac{6M_0}{V^2}
+\frac{24}{w_{\min}}
+\frac{6PH_w^2D}{V^2w_{\min}}
\right]A_{K+D}.
\tag{12}
\]

The drift derivative is the sum of `-V s q` and nonnegative curvature.
Since the certified-gain part contributes at least `V L_min s^2` to the second derivative, its root satisfies

\[
\delta_k:=\alpha_ks_{I_k,k}
\le q_{I_k,k}/L_{\min}.
\]

Thus

\[
\sum_{k=0}^{K-1}\delta_k^2
\le\frac{2M_0}{V^2L_{\min}^2}A_{K+D}.
\tag{13}
\]

Finally assume joint block-gradient cross sensitivity

\[
\|\nabla_iJ(\theta')-\nabla_iJ(\theta)\|
\le\Gamma\|\theta'-\theta\|.
\]

Reuse each block's most recent completed-gradient bound for at most `D` events and charge intervening path energy by Cauchy--Schwarz.
Then

\[
\frac1K\sum_{k=0}^{K-1}\|\nabla J(\theta_k)\|^2
\le
\frac{C_{\rm event}A_{K+D}}{K}
+\frac{4n\Gamma^2D^2M_0A_{K+D}}
{KV^2L_{\min}^2}
+\frac{2nDG^2}{K},
\tag{14}
\]

where

\[
C_{\rm event}
=2D\left[
\frac{6M_0}{V^2}
+\frac{24}{w_{\min}}
+\frac{6PH_w^2D}{V^2w_{\min}}
\right].
\]

With exact reset gradients, exact Hessian-vector products and zero initial transport debt, every `B_k` is zero and (14) is `O(1/K)` for fixed bounded constants.
With sampled trajectories, `K^{-1}sum B_k` remains an explicit estimator floor.
This corollary is conditional on the bounded completion closure and does not yet convert event count to random wall-clock time.

## 7. Bounded episodic Markov estimator interface

For a fixed-horizon trajectory `tau`, define the block score

\[
S_i(\tau)=\sum_{t=0}^{H-1}\nabla_{\theta_i}
\log\pi_{i,\theta_i}(A_{i,t}\mid O_{i,t}).
\]

If the discounted return is bounded by `R_bar`, the score norm by `S_bar`, and the log-policy Hessian norm by `T_bar`, the REINFORCE trajectory estimators satisfy

\[
\|\widehat g_i(\tau)\|\le R_{\rm bar}S_{\rm bar},
\]

and the joint Hessian row applied to a predictable unit vector is bounded by a public constant formed from `R_bar`, `S_bar` and `T_bar`.
For `m` independent reset trajectories, coordinate Hoeffding plus a union bound over the first `K` completions gives radii of order

\[
r_i^g
=B_g\sqrt{\frac{2d_i\log(2d_i nK/\delta)}{m}},
\qquad
r_i^H
=B_H\sqrt{\frac{2d_i\log(2d_i nK/\delta)}{m}}.
\tag{15}
\]

The second statement in (15) conditions on the displacement vector, which must be predictable relative to the retained birth batch.
Independent replicas and outcome-independent completion times provide that measurability.
If completion time depends on trajectory content, or if batches are shared across interacting proposals, (15) is not valid without a martingale replacement.

Within-trajectory Markov dependence is part of one bounded trajectory estimator; it is not counted as independent samples.
Continuing trajectories require a separate mixing-time or spectral-gap concentration result and are outside Theorem 1.

## 8. Remaining proof obligations

The following items block any efficacy preregistration.

1. **Wall-clock theorem.** Convert event count using a verified Markov completion process and include the extra Hessian-vector computation in service time.
2. **Strong separation.** Prove a family in which transport beats barrier, raw stale, discard/refresh and first-order delay compensation under the same compute budget.
3. **Neural estimator boundary.** The practical critic and any diagonal Hessian approximation require their own bias radii; they cannot inherit (15) by naming them estimators.

Until these three obligations close, Theorem 1 and Corollary 1 are a conditional optimization interface rather than a complete ICML main theorem.
