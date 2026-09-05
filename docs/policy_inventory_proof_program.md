# Policy-inventory proof program

Status: exact one-step geometry and conditional event-time drift are closed;
the Markov trajectory interface, full-gradient conversion, wall-clock theorem
and separation remain proof obligations.  This document is not a completed
paper theorem.

## 1. Event model

There are `n` distinct actor blocks in an identical-interest Markov game with
factorized policy `pi_theta=product_i pi_{i,theta_i}` and common return
`J(theta)`.  A finite set `F_k` of independent-reset joint rollout batches is
in flight immediately before completion event `k`.  Batch `p` was born under
the recorded behavior snapshot `theta^{b_p}`.  Completion times are
conditionally independent of trajectory values.

At event `k`, batch `i_k` completes.  An importance-corrected estimator
`gtilde_k` for block `i_k` is formed from that batch.  The server either rejects
it or applies `theta_{i_k} <- theta_{i_k}+alpha_k gtilde_k`; it then launches a
replacement joint batch at the new snapshot.  Thus the replacement has zero
inventory divergence at birth.  At most `P` batches are outstanding.

## 2. Risk interface

For each outstanding batch define

\[
z_{p,k}=\log\mathbb E_{\mu_p}[W_{p,k}^2],
\qquad
\Phi(z)=e^z-1,
\tag{1}
\]

where `W_{p,k}` is the trajectory likelihood ratio from its behavior joint
policy to the current target joint policy.  Then `Phi(z)` is its chi-square
importance-weight variance inflation.

For the fixed-variance factorized Gaussian one-step game,

\[
z_{p,k}=\sum_j
\frac{\|\theta_{k,j}-\theta^{b_p}_j\|^2}{\sigma_j^2}
\tag{2}
\]

exactly.  A block-`i` step along `g` gives

\[
z_{p,k}(\alpha)=z_{p,k}
+\ell_{p,k}\alpha+c_k\alpha^2,
\quad
\ell_{p,k}=\frac{2\langle
\theta_{k,i}-\theta^{b_p}_i,g\rangle}{\sigma_i^2},
\quad
c_k=\frac{\|g\|^2}{\sigma_i^2}.
\tag{3}
\]

For a general finite-horizon Markov game, the proof must replace (2) by an
observable upper process `zbar_{p,k}` satisfying both `z<=zbar` and a
predictable post-step envelope.  A sum of empirical per-agent KLs is not
automatically a Renyi-second-moment bound and may not be substituted without a
ratio or concentration assumption.

If an unweighted trajectory-gradient contribution has norm at most `C` and
the batch contains `B` independent episodes, importance weighting yields

\[
\mathbb E\|\widetilde g-\nabla_iJ(\theta)\|^2
\le C^2e^z/B.
\tag{4}
\]

Equation (4) alone is **not** sufficient for the adaptive realized-direction
drift below: the step is selected after observing the same batch, so an RMS
number cannot be substituted as a pathwise norm-error radius.  The drift
theorem is conditional on a simultaneous event
`||gtilde-nabla_i J||<=r(z)`.  Establishing that event requires bounded/clipped
ratios with explicit bias or a robust mean result under declared tail
conditions.

## 3. Physical Lyapunov drift

Let

\[
\mathcal L_k=V(J^\star-J(\theta_k))
+\sum_{p\in\mathcal F_k}w_p\Phi(z_{p,k}).
\tag{5}
\]

On the simultaneous norm-error event, let `s_k=||gtilde_k||`, certified radius
`r_k`, block smoothness `L_i`, and

\[
G_k(\alpha)=\alpha(s_k^2-r_ks_k)
-\frac{L_i}{2}\alpha^2s_k^2,
\tag{6}
\]

the one-event drift is bounded by

\[
\mathcal L_{k+1}-\mathcal L_k
\le -w_{i_k}\Phi(z_{i_k,k})+F_k(\alpha),
\tag{7}
\]

where

\[
F_k(\alpha)=-VG_k(\alpha)
+\sum_{p\ne i_k}w_p
\{\Phi[z_{p,k}(\alpha)]-\Phi(z_{p,k})\}.
\tag{8}
\]

The algorithm minimizes (8) on a public trust interval.  Each exponential of
the convex quadratic (3) is convex, so `F_k` is scalar convex.  The derivative
is monotone and the exact minimizer is obtained with endpoint checks and
bisection in `O(P+log(1/epsilon))` scalar work after the policy-gradient
computation.

Since `alpha=0` is feasible and `F_k(0)=0`, on the confidence event,

\[
\mathcal L_{k+1}-\mathcal L_k
\le -w_{i_k}\Phi(z_{i_k,k}).
\tag{9}
\]

Therefore, for any finite `K` on the simultaneous event,

\[
\sum_{k=0}^{K-1}w_{i_k}\Phi(z_{i_k,k})
\le \mathcal L_0.
\tag{10}
\]

This is the first useful closed statement: the total second-moment inflation
of consumed paid rollouts is financed by the initial optimization potential
and initial inventory.  It does not by itself prove nonzero learning progress;
an algorithm that always rejects also satisfies (10).

## 4. Nontrivial-progress interface

The derivative at zero is

\[
F'_k(0)=-V(s_k^2-r_ks_k)
+\sum_{p\ne i_k}w_pe^{z_{p,k}}\ell_{p,k}.
\tag{11}
\]

Define the inventory-priced marginal signal

\[
q_k=\left[
s_k-r_k-
\frac{1}{Vs_k}
\sum_{p\ne i_k}w_pe^{z_{p,k}}\ell_{p,k}
\right]_+
\tag{12}
\]

for `s_k>0`, and zero otherwise.  Then `F'_k(0)=-Vs_kq_k` whenever the
positive part is active.  If a public bound

\[
F''_k(\alpha)\le M_0s_k^2
\tag{13}
\]

holds on the trust interval and the cap contains `Vq_k/(M_0s_k)`, scalar
smoothness gives

\[
F_k(\alpha_k)\le -\frac{V^2}{2M_0}q_k^2.
\tag{14}
\]

Combining (7) and (14) yields the conditional finite-event bound

\[
\sum_{k<K}\left[
w_{i_k}\Phi(z_{i_k,k})
+\frac{V^2}{2M_0}q_k^2
\right]
\le \mathcal L_0.
\tag{15}
\]

The repository algebra verifies the exact risk geometry, positivity of the
curvature and the scalar minimizer.  A paper proof still needs an explicit
`M_0` from bounded `P`, `z`, displacements, gradient norms, policy variance and
trust cap.

## 5. What remains for a valid MARL stationarity theorem

Equation (15) controls an inventory-priced signal, not automatically the full
joint gradient.  Conversion requires all of the following stated assumptions
or new lemmas:

1. every actor block completes in at most `D` completion events;
2. the importance-gradient RMS/bias interface covers the implemented
   estimator under Markov trajectories;
3. likelihood-ratio risk is capped or robustly estimated so `M_0` is finite;
4. the signed inventory price in (12) is related to completed-risk energy,
   rather than bounded by a constant that makes the theorem vacuous;
5. wall-clock service has an ergodic lower completion rate and includes learner
   and actor costs;
6. a nonempty regime makes `q_k>0`, showing the protection term does not force
   permanent rejection.

A target corollary would have the transparent form

\[
\frac1K\sum_{k<K}\mathbb E\|\nabla J(\theta_k)\|^2
\le O\!\left(\frac{D\mathcal L_0}{K}\right)
+O\!\left(\frac{e^{z_{\max}}}{B}\right)
+O\!\left(\frac{P^2}{V^2}\right),
\tag{16}
\]

followed by a service-rate conversion from events to wall-clock time.  The
constants and exponents in (16) are a desired interface, not yet a theorem.

## 6. Required separation

The positive story cannot rest only on convergence.  The required family has
two hidden workload phases with the same policy objective:

- low inventory load, where the base optimizer is nearly optimal;
- bursty high load, where a base-size update exponentially inflates the second
  moment of many outstanding joint rollouts.

Every fixed step or fixed version-age threshold must pay one of two costs:
slower learning in the low phase or spoiled gradients in the high phase.  The
observable inventory state in (1)--(3) must let the same causal controller
approach the phase oracle without knowing phase transition times.  This must be
shown first by an outcome-free CPU ceiling and then by disjoint stochastic
seeds.  If that separation has less than material wall-clock value after all
rollout/log-probability costs, the candidate stops before a GPU benchmark.
