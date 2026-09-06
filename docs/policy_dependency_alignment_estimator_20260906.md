# Sparse signed-alignment estimator for asynchronous MARL

Date: 2026-09-06

Status: estimator definition and expectation-level error theorem.  This is not
an empirical nonvacuity result and does not authorize a controller pilot.

## 1. Estimator used by the launch decision

Immediately before dispatch, the server uses completed data only.  Let
`a_i=grad_i F(theta)` be the current owner gradient and let `g_i^0` be the
mean gradient under the worker's current mixed-version cache.  For a candidate
local edge `j->i`, define the known version displacement

\[
\delta_{ji}=v_j-\chi_{j\to i}.
\tag{1}
\]

The local first-order candidate is

\[
\widehat g_{ij}=\widehat g_i^0+\widehat J_{ij}\delta_{ji},
\qquad
\widehat J_{ij}\approx
\nabla_{\theta_j}G_i(\theta_i,\chi_{-i\to i}).
\tag{2}
\]

The Jacobian-vector product is obtained by differentiating the owner-gradient
alignment through only the declared local critic factors.  Non-neighbor actor
blocks are absent from the forward/autograd graph.  No additional environment
trajectory is collected per edge.

The no-refresh action uses `hat g_i^0`.  Exact cache reset and queue terms are
observed, so only the optimized learning-drift term requires statistical
estimation.

## 2. Markov sample-mean error

For a stationary vector statistic `X_t`, assume

\[
\sum_\ell\operatorname{Var}(X_t^{(\ell)})\le\nu^2,
\qquad
|\operatorname{Cov}(X_t^{(\ell)},X_{t+h}^{(\ell)})|
\le\operatorname{Var}(X_t^{(\ell)})\rho^h,
\tag{3}
\]

with a declared `0<=rho<1`.  For a block of length `m`, define

\[
\kappa_m(\rho)=1+2\sum_{h=1}^{m-1}
\left(1-\frac hm\right)\rho^h.
\tag{4}
\]

Directly expanding the covariance of the sample mean gives

\[
\left(\mathbb E\|\bar X-\mathbb EX\|^2\right)^{1/2}
\le \nu\sqrt{\frac{\kappa_m(\rho)}m}.
\tag{5}
\]

A nonstationary start, critic approximation, or clipped importance weight
adds a named mean-bias norm to (5).  These effects are not hidden inside an
independence assumption.  Unrestricted unknown mixing is outside the theorem;
the implementation needs either a public upper bound on `rho`, regeneration,
or a separately certified bound away from one.

## 3. Candidate-gradient error

Let `r_0` and `r_J` be RMS bounds for the base gradient and local Jacobian
estimates.  If the local Jacobian is `M_ij`-Lipschitz along (1), Taylor's
theorem and the sparse true-gradient tail give

\[
r_{g,ij}
\le r_0+r_J\|\delta_{ji}\|
+\frac{M_{ij}}2\|\delta_{ji}\|^2
+\varepsilon_{i,k}^{sp}+b_{ij}^{off}.
\tag{6}
\]

Every term in (6) has a distinct meaning: sampling error, JVP sampling error,
Taylor remainder, omitted policy-dependency tail, and off-policy bias.  A term
already present in the pending-history radius is not added again.

## 4. Error of the optimized drift value

For current gradient `a`, candidate `g`, curvature `Lambda`, and
`0<=alpha<=alpha_max`, let

\[
\phi(a,g)=\min_\alpha
\left\{-\alpha\langle a,g\rangle
+\frac{\Lambda\alpha^2}{2}\|g\|^2\right\}.
\tag{7}
\]

Suppose `||a||<=A`, `||g||<=G`, and the RMS estimator errors are `r_a` and
`r_g`.  Cauchy--Schwarz gives

\[
\mathbb E|\langle\widehat a,\widehat g\rangle-
\langle a,g\rangle|
\le Ar_g+Gr_a+r_ar_g,
\tag{8}
\]

and

\[
\mathbb E|\|\widehat g\|^2-\|g\|^2|
\le r_g(2G+r_g).
\tag{9}
\]

Because `|min f-min g|<=sup|f-g|`, one valid action-score error is

\[
e_{ij}\le
\bar\alpha(Ar_g+Gr_a+r_ar_g)
+\frac{\Lambda\bar\alpha^2}{2}r_g(2G+r_g).
\tag{10}
\]

For a finite neighborhood, `e_k` in the main theorem can use any proved
uniform bound; the conservative expectation-only choice
`sum_(j in N_i union {0}) e_ij` follows from `max<=sum`.  A sharper shared-data
maximal inequality is useful but is not needed for correctness.

## 5. Adaptivity at packet receipt

Choosing a scalar step from a realized gradient and applying that same random
gradient creates selection correlation.  The first theorem-facing
implementation therefore uses two conditionally independent groups of
rollouts born under the same mixed policy:

1. the control group estimates the return-time score and selects the step;
2. the update group supplies the gradient that is applied.

Both groups are counted in the environment and wall-clock ledgers.  In an
episodic benchmark they must have distinct environment resets; consecutive
halves of one continuing trajectory are not called independent.  A future
martingale proof may remove this split, but the experiment cannot assume that
improvement for free.

## 6. Computational scope

For dependency degree `Delta`, the estimator stores and contracts `Delta`
local JVP blocks, so its structural arithmetic and derivative support are
`O(Delta)` times the local actor/factor cost.  The PDSG-COMP-002 timing audit
did not certify monotone wall-clock scaling: its median and maximum aggregate
HVP ratios were 1.7441 and 2.0277, while two strict micro-timing gates failed.
The paper may state the structural support result and report measured overhead;
it may not claim degree-independent runtime.

## 7. Next gate

Before any learning controller is run, an independently committed CPU audit
must test whether (10) is nonvacuous after charging the control/update split.
It must vary block length, Markov persistence, degree, displacement, Taylor
curvature, and sparse tail.  Passing requires both score-error-to-margin
separation and correct directional edge ranking on a frozen synthetic model.
Failure modifies or rejects the estimator; it does not alter the previously
observed oracle problem headroom.
