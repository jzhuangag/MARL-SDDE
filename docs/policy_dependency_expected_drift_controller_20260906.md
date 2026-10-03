# Expected-drift controller for asynchronous policy synchronization

Date: 2026-09-06

Status: primary algorithmic direction and deterministic theorem skeleton.  No
new efficacy experiment or seed set is authorized by this document.

## 1. Why the primary guarantee is cumulative

Requiring every stochastic refresh decision to be harmless with high
probability forces a simultaneous lower confidence bound on a difference of
policy-gradient alignments.  Earlier fully charged audits in this repository
showed that analogous per-packet certificates can remain inactive even in
optimistic regimes.  Reusing that design would repeat a known failure.

The primary claim is instead stochastic-optimization based: the controller
minimizes a predictable noisy estimate of drift plus communication price, and
the theorem pays the cumulative conditional mean estimation error.  A single
refresh may be suboptimal.  The average policy progress and long-run message
budget are controlled.

## 2. True and estimated launch scores

At dispatch event `k`, let `A_k` contain no refresh and at most one eligible
incoming refresh edge.  For action `a` and step `alpha`, let

\[
\psi_k(a,\alpha)
=-\alpha A_{k,a}
+\frac{L_i\alpha^2}{2}N_{k,a}
-R_{k,a}
+Q_kc_{k,a}
+C_{k,a}^{\rm hist},
\tag{1}
\]

where `A_(k,a)` is the true gradient alignment, `N_(k,a)` upper-bounds the
candidate squared-gradient norm, `R_(k,a)` is the delayed cache-debt reduction,
and `C_(k,a)^hist` pays future source motion and Markov packet bias.  Cache,
queue, and message terms are observed; gradient and history terms are
estimated from data completed before dispatch.

Let `hat(psi)_k` be the executable score and define

\[
E_k=\sup_{a\in\mathcal A_k,\ 0\le\alpha\le\bar\alpha}
|\widehat\psi_k(a,\alpha)-\psi_k(a,\alpha)|.
\tag{2}
\]

The controller chooses

\[
(\widehat a_k,\widehat\alpha_k)
\in\arg\min_{a,\alpha}\widehat\psi_k(a,\alpha).
\tag{3}
\]

For a fixed edge, the estimated first-order alignment and norm coefficient
retain the scalar closed-form step.  The finite edge comparison then costs
`O(deg(i))`.

## 3. Selection lemma

### Lemma 1 (noisy drift minimization)

For every realization on which (2) is finite,

\[
\psi_k(\widehat a_k,\widehat\alpha_k)
-\inf_{a,\alpha}\psi_k(a,\alpha)
\le 2E_k.
\tag{4}
\]

Consequently, if
`E[E_k | F_k] <= epsilon_k`, then the conditional expected drift regret is at
most `2 epsilon_k`.

### Proof

Let `(a_k^*,alpha_k^*)` minimize the true score.  Add and subtract the two
estimated scores and use the optimality in (3):

\[
\begin{aligned}
\psi_k(\widehat a_k,\widehat\alpha_k)
&\le\widehat\psi_k(\widehat a_k,\widehat\alpha_k)+E_k\\
&\le\widehat\psi_k(a_k^*,\alpha_k^*)+E_k\\
&\le\psi_k(a_k^*,\alpha_k^*)+2E_k.
\end{aligned}
\]

Taking conditional expectation proves the second statement.  No independence
between candidate scores is required.

If only the alignment and squared-norm coefficients are estimated, with
conditional expected uniform errors `r_A,k` and `r_N,k`, then

\[
\epsilon_k
\le\bar\alpha r_{A,k}
+\frac{L_i\bar\alpha^2}{2}r_{N,k}
+r_{\rm hist,k}.
\tag{5}
\]

This identifies the statistical target required from the Markov packet
analysis without demanding a per-event lower confidence certificate.

## 4. Pathwise message accounting

For

\[
Q_{k+1}=[Q_k+c_k-\bar c]^+,
\tag{6}
\]

iteration gives

\[
\frac1T\sum_{k<T}c_k
\le\bar c+\frac{Q_T-Q_0}{T}.
\tag{7}
\]

Suppose every estimated learning/history score excluding `Q_k c` is bounded
in absolute value by `M`, no refresh has zero cost, and every nonzero edge has
cost at least `c_min`.  Whenever

\[
Q_k>\frac{2M}{c_{\min}},
\tag{8}
\]

no positive-cost action can beat no refresh.  Hence

\[
Q_k\le
\max\left\{Q_0,\frac{2M}{c_{\min}}+c_{\max}\right\}
\tag{9}
\]

under the exact minimizer and bounded costs.  Equations (7)--(9) give a
pathwise asymptotic message-budget guarantee without a stochastic Slater
argument.  Approximate minimization adds its declared score tolerance to the
threshold.

## 5. Intended finite-time learning theorem

Let the event-time composite energy contain the expected Markov-potential loss,
pairwise cache debt, a nonnegative pending-packet history term, and `Q_k^2/2`.
The target discrete theorem has the form

\[
\sum_{k<T}\alpha_k
\mathbb E\|\nabla F(\theta_k)\|^2
\le
C_0+C_{\rm noise}\sum_{k<T}\alpha_k^2
+C_{\rm delay}\sum_{k<T}\alpha_k^2D_k^2
+2\sum_{k<T}\epsilon_k.
\tag{10}
\]

The cache and pending-history terms must be chosen so their shifts cancel the
same staleness cross terms appearing in the learning drift.  Equation (10) is
not yet proved for the policy-refresh pipeline.  In particular, the frozen
delayed-snapshot runner does not establish it.

If `alpha_k` is of order `T^(-1/2)` and the average `epsilon_k` decays at the
matching scale, (10) yields an `O(T^(-1/2))` stationarity rate plus explicit
delay and Markov-estimation terms.  A potential-game Nash-gap corollary needs
separately stated occupancy and policy-interiority conditions; stationarity is
not automatically a Nash guarantee.

## 6. Nontrivial adaptation condition

Let `Delta_oracle` be the equal-budget dynamic drift advantage over the strong
static/online envelope and let

\[
\overline\epsilon_T=T^{-1}\sum_{k<T}\epsilon_k.
\]

The signed controller has provable room to improve only when

\[
\Delta_{\rm oracle}
>2\overline\epsilon_T
+\Delta_{\rm delay}
+\Delta_{\rm compute}.
\tag{11}
\]

PDSG-001 estimates the left side in an exact quadratic delayed-snapshot model.
It does not measure the three right-side terms.  The next CPU feasibility gate
must estimate those terms without using PDSG-001 outcomes to choose its
thresholds.

## 7. Immediate work program

1. Specify the four-event dispatch/delivery/rollout/return filtration and prove
   the pending-history identity.
2. Derive `epsilon_k` for reset episodic trajectories first; continuing-chain
   mixing is a later extension.
3. Implement one-reverse-pass alignment prediction and measure its cost and
   Taylor error on frozen neural batches without running a learning outcome.
4. Freeze a nonvacuity gate for (11).
5. Only if that gate passes, preregister a successor CPU learning experiment.

This sequence keeps the research question fixed while allowing the estimator
or history functional to be rejected before a costly benchmark.
