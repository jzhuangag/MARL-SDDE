# T-030 full-risk theory program after EXP-019A

## Decision inherited from EXP-019A

The variance-only fixed-q selector is permanently stopped.  Its prospective
Blackjack pilot improved normalized MSVE-AUC by only 0.01665%, about 300 times
smaller than the frozen 5% gate, and the exact reproduction is byte-identical.
Increasing seeds cannot repair this scientific-effect failure.  No Asterix or
GPU experiment is authorized.

The only admissible next main line is **Transient Before Variance: Bias-,
Correlation-, and Delay-Limited Participation from Multi-Agent Markov Data**.
It must select a scalar step and participation jointly, or derive a single
offline switch from low participation during contraction to larger
participation near the noise floor.  It may not use a preconditioner, Hessian,
covariance matrix, online probe, posterior, or outcome feedback.

## Exact iid base recurrence

For a stationary transition sample `z=(S,R,S')`, define

\[
J(z)=\phi(S)(\phi(S)-\gamma\phi(S'))^\top,
\quad
\xi(z)=\phi(S)\{R+\gamma\phi(S')^\top w^*-\phi(S)^\top w^*\}.
\]

Let `A=E[J]`, `E[xi]=0`, `Q=E[xi xi^T]`, and
`Jop(X)=E[J X J^T]`.  Under the registered common-versus-independent batch
coupling, set

\[
v_q(\rho)=\rho+(1-\rho)/q.
\]

For an iid stationary batch independent of the current error, the delayed TD
error satisfies

\[
e_{t+1}=e_t-\alpha\bar J_t e_{t-D}+\alpha\bar\xi_t,
\qquad
m_{t+1}=m_t-\alpha A m_{t-D}.
\]

Writing `M_{a,b}=E[e_a e_b^T]` and
`C(m)=E[J m xi^T]`, the new diagonal block is exactly

\[
\begin{aligned}
M_{t+1,t+1}={}&M_{t,t}
-\alpha\{A M_{t-D,t}+M_{t,t-D}A^\top\}\\
&+\alpha^2\{v_q\,\mathcal J(M_{t-D,t-D})
 +(1-v_q)A M_{t-D,t-D}A^\top\}\\
&-\alpha^2v_q\{C(m_{t-D})+C(m_{t-D})^\top\}
 +\alpha^2v_q Q.
\end{aligned}
\]

For every older lifted block,

\[
M_{t+1,k}=M_{t,k}-\alpha A M_{t-D,k}.
\]

These equations retain transient bias, batch variance, cross-agent
correlation, scalar step size, and discrete delay.  They also show why a
variance-only score cannot be a finite-time learning bound.

## Markov route already available, and its practical limitation

The displayed recurrence is exact only when the current batch is independent
of the lifted error history.  EXP-019A uses thinned Markov streams, so replacing
conditional products such as

\[
E[\bar J_t e_{t-D}e_t^\top]
\]

by `A M_{t-D,t}` is not exact.  A rigorous T-030 theorem must close one of the
following without changing the registered marginal task law:

1. a state-conditioned Poisson/martingale recurrence that retains the
   dependence between the Markov sample and all delayed error blocks; or
2. a proved mixing perturbation bound with explicit constants that is added
   to, rather than substituted for, the iid full-risk recurrence.

The repository's audited Theorem 4 already supplies route 2 for the
predictably decorrelated affine algorithm: it retains finite-time bias,
innovation forcing, total-variation mixing error, scalar step size, and
delay.  Therefore a new Poisson proof is not required merely to obtain a
valid conservative selector.  T-030 must first instantiate that theorem on
Blackjack and audit whether its Euclidean-norm constants are informative.

The lifted delay state has `D+1` parameter blocks.  Merely constructing the
full dense covariance is not an acceptable algorithm: it scales as
`O((D+1)^2 d^2)` memory and would contradict the low-complexity objective.
The proof may use this object analytically, but the executable selector must
reduce to scalar certified constants and cost at most
`O(|Q||A|)` offline and `O(1)` online.

## Required risk and possible one-switch rule

The target certificate must retain all terms

\[
\mathcal R_c(q,\alpha)=
C_0r(\alpha,D)^{2N_c(q)}
+\mathcal V(\alpha,\lambda,b,D)v_q(\rho)
+\mathcal M_{\lambda,b}+\mathcal D_{\alpha,D}.
\]

The fixed method jointly minimizes this certificate over the finite q
catalogue and a scalar stability-screened step catalogue.  If the same bound
has a unique bias-variance crossing, T-030 may additionally derive one public
switch time

```text
q_t = 1 before T_switch; q_t = q_variance after T_switch.
```

This is an offline schedule, not an online controller.

## Stop/proceed gates before another sampled experiment

All gates are mandatory:

1. delayed mean and second-moment recurrences are proved in the stated Markov
   model, including the sample/error dependence term;
2. the scalar stability interval depends explicitly on D, and the discrete
   proof is separate from the SDDE interpretation;
3. bias, variance, mixing, and delay residuals all remain visible;
4. Blackjack constants come from its exact kernel and reward law, never by
   fitting EXP-019A outcomes;
5. on a frozen budget grid spanning at least one order of magnitude, the
   analytic exact/upper-bound ceiling has aggregate AUC improvement at least
   5% and terminal ratio at most 0.98;
6. at least 60% of prospective active cells improve by at least 2%, inactive
   aggregate ratio is at most 1.02, and at least three q values are optimal;
7. theorem ranking matches the exact recursion ranking in at least 80% of
   auditable cells;
8. zero initialization passes independently; warm starts cannot rescue it;
9. gains cover low/high correlation, every registered delay, and more than
   one isolated budget ray;
10. offline selection is at most `O(|Q||A|)` and online overhead is `O(1)`.

Until all ten pass, no EXP-020A, formal seeds, Asterix preregistration, GPU
prompt, or HPC4 job is allowed.  If the analytic ceiling fails the practical
effect gates, the participation-algorithm ICML line stops and the defensible
output becomes a narrower theory paper centered on the correlation law,
affine delayed convergence, opportunity-cost lower bound, and unknown-mixing
impossibility.

## Current theorem status

- exact iid D=0/delayed lifted algebra: specified above; its common/private
  batch coefficients are exhaustively checked in a scalar finite-support base
  case by `test_t030_iid_full_risk.py`;
- full Markov exact-moment conditioning: open, but not required for the
  existing conservative Theorem 4 route;
- low-memory scalar Theorem 4 certificate: proved previously and now requires
  a Blackjack nonvacuity audit;
- bias-aware exact Blackjack analytic ceiling: still open beyond the
  conservative certificate;
- new sampled CPU/GPU experiments: prohibited.

The first conservative instantiation is now complete.  Even after setting
mixing error, innovation forcing, and delay to zero and combining the most
favorable curvature with the largest update count, the Euclidean Theorem 4
bound can decrease by at most 0.0105612%.  It therefore fails the 5%
nonvacuity gate.  The next admissible proof target is a stationary-weighted
MSVE certificate; the current Euclidean certificate cannot be used as the
practical selector.

This is a genuine theory blocker, not a request for more compute.
