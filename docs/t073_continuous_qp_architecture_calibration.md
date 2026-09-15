# T-073 continuous-QP collaboration architecture calibration

T-073 replaces T-072's seven recipient actions by a continuous probability
simplex. For recipient (i), the observable first-half fingerprint supplies a
mean vector, a PSD cross-agent covariance-of-mean proxy, and a conservative
effective sample size. The controller solves

\[
\min_{p\in\Delta^n}
  (p^\top x-\widehat\mu_i)^2
  +p^\top\widehat\Sigma_{\rm mean}p
  +nQ_i\lVert p-e_i\rVert^2.
\]

The three terms respectively control personalized bias, correlation-dependent
variance, and Lyapunov safety debt. The complete vector (p) jointly selects
donors and mixing strengths; there is no finite donor/alpha scan. Projected
gradient uses at most 50 iterations, each (O(n^2)), and simplex projection
costs (O(n\log n)). The disjoint second half validates and learns exactly as
in T-072, so every transition remains a learning transition.

This calibration is explicitly outcome-informed: covariance weights 0.25 and
1 were checked on four old T-071A seeds and both were positive; the canonical
unit coefficient was frozen rather than selecting the slightly better value.
The full run reuses all old seeds and can only authorize a separately committed
new-seed preregistration. Q1--Q9 are descriptive architecture criteria, not
paper claims or significance tests. No GPU, HPC4, nonlinear, or formal run is
authorized.
