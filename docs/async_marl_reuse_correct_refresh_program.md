# Reuse, Correct, or Refresh? A unified asynchronous-MARL program

## One research question

In centralized training with decentralized execution, heterogeneous actors
finish trajectories under different versions of a factorized joint policy.
When a packet reaches the learner, there are three meaningful actions:

1. **reuse** it without correcting teammate drift, accepting bias;
2. **correct** selected teammate likelihood factors, accepting multiplicative
   importance-weight variance;
3. **refresh** it under the current joint policy, spending actor interactions
   and serialized learner time.

The paper question is how to make this choice causally and online under
long-run interaction, communication, and wall-clock budgets while retaining a
finite-time learning guarantee.  This is not a collection of heuristics: all
three actions lie on one certified bias--variance--freshness frontier.

The earlier freshness-only program is a strict special case with correction
disabled.  Its equal-cost MPE failure shows why the correction axis is
necessary; it is not positive evidence for this new program.

## Factorized tempered correction

Let a completed trajectory have behavior joint policy

\[
\mu(a\mid o)=\prod_{j=1}^m\mu_j(a_j\mid o_j)
\]

and let the learner's current policy be
\(\pi=\prod_j\pi_j\).  For teammate factor \(j\), choose a continuous
tempering strength \(\alpha_j\in[0,1]\).  Zero ignores that factor and one
uses its full likelihood-ratio correction.  Intermediate values follow the
normalized geometric path between \(\mu_j\) and \(\pi_j\).

Suppose an observable sensitivity certificate \(d_j\) bounds the change in
the owner update caused by leaving factor \(j\) uncorrected, and an observable
log-ratio proxy \(v_j\) bounds its contribution to the corrected estimator's
log second moment.  Then

\[
 B(\alpha)\leq \left[\sum_j d_j(1-\alpha_j)\right]^2,
 \qquad
 V(\alpha)\leq \frac{\sigma^2}{n_{\rm eff}}
 \exp\!\left(\sum_jv_j\alpha_j^2\right).
\]

For \(S=\sum_jv_j\) and
\(c(S)=(e^S-1)/S\), convexity of the exponential gives the valid secant bound

\[
e^s\leq 1+c(S)s,\qquad 0\leq s\leq S.
\]

Thus the causal correction decision solves the box QP

\[
\min_{0\leq\alpha\leq1}
 \left[d^\top(\mathbf1-\alpha)\right]^2
 +\frac{\sigma^2}{n_{\rm eff}}c(S)
   \sum_jv_j\alpha_j^2
 +p^\top\alpha,
\]

up to the additive base sampling variance.  The prices \(p_j\) come from
Lyapunov queues for correction metadata or compute.  The Hessian is diagonal
plus rank one.  If \(x=d^\top(\mathbf1-\alpha)\), the KKT solution is

\[
\alpha_j(x)=\left[
\frac{d_jx-p_j/2}{\kappa v_j}
\right]_{[0,1]},\qquad
x+\sum_jd_j\alpha_j(x)=\sum_jd_j,
\]

where \(\kappa=\sigma^2c(S)/n_{\rm eff}\).  The final equation is monotone
and solved by scalar bisection.  Complexity is
\(O(m\log(1/\varepsilon))\), with no subset scan, Hessian inverse, or
preconditioner.

## Lyapunov drift is structural, not decorative

For resource \(r\), let

\[
Z_{r,k+1}=[Z_{r,k}+c_{r,k}(u_k,\alpha_k)-\bar c_r]_+.
\]

At each packet arrival the learner compares the priced QP optimum with a
fully charged fresh estimator.  It minimizes the usual drift-plus-certified-
risk expression

\[
V\,\overline R_k(u_k,\alpha_k)
+\sum_r Z_{r,k}c_{r,k}(u_k,\alpha_k).
\]

Queue stability yields the registered time-average resource constraints; the
drift bound yields an \(O(1/V)\) risk gap to the best stationary causal policy,
plus an \(O(V/K)\) finite-horizon debt term.  This is the precise role of
Lyapunov theory: it converts statistical correction/refresh choices into an
online constrained stochastic-optimization algorithm.

## Learning consequence to prove

For an \(L\)-smooth Markov potential \(\Phi\) and update
\(\theta_{k+1}=\theta_k+\eta\widehat g_k\), a certified conditional MSE
\(\mathbb E_k\|\widehat g_k-\nabla\Phi(\theta_k)\|^2\leq R_k\) gives, for a
sufficiently small step size,

\[
\frac1K\sum_{k<K}\mathbb E\|\nabla\Phi(\theta_k)\|^2
\leq
O\!\left(\frac{\Phi_{\max}-\Phi_0}{\eta K}\right)
+O\!\left(\frac1K\sum_{k<K}\mathbb E R_k\right).
\]

Under a stated gradient-dominance condition this becomes a finite-time Nash-
gap result.  Without that condition the honest conclusion is convergence to a
stationarity neighborhood.  Delay enters through the observed policy-ratio
and sensitivity certificates rather than an arbitrary worst-case delay cap.

## Two survival gates before a new experiment identifier

1. **Performance-bound gate.**  Prove the normalized geometric-path bias
   bound, second-moment bound, predictable certificate construction, Lyapunov
   guarantee, and potential-learning consequence without treating reused
   samples as independent or on-policy.
2. **CPU headroom gate.**  On frozen, outcome-free stochastic regimes, the
   adaptive reuse/correct/refresh oracle and causal controller must beat no
   correction, full correction, fixed clipping/tempering, fixed staleness
   cutoffs, and fixed refresh schedules under identical costs.  The gain must
   persist after Markov effective-sample-size charging.

Only if both gates pass will the project preregister a standard MARL
actor--critic benchmark.  That later benchmark will require a GPU; no GPU is
currently authorized.

## Scope discipline

- CTDE is the training setting; execution remains decentralized and does not
  require the controller or inter-agent communication.
- Markov-game trajectories, not iid gradients, are the target data model.
- Discrete-time Lyapunov drift is the primary proof tool.  An SDDE can be used
  later only as a continuous-time interpretation of asynchronous queue/policy
  dynamics, not as the convergence proof.
- Novelty cannot be claimed for importance sampling, V-trace/MA-Trace,
  synchronization, clipping, or virtual queues individually.  The candidate
  contribution is their certified continuous multi-agent
  reuse--correction--refresh frontier and its joint resource/performance
  theorem.
