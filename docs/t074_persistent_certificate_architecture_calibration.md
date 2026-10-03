# T-074 persistent mixing-certificate architecture calibration

T-074 retains T-073's continuous covariance-aware simplex QP and replaces its
five-sample mixing estimate by a predictable persistent certificate. Completed
blocks contribute only within-block centered scatter and lag products, so
target switches do not masquerade as noise correlation. At a decision, the
current prefix is added temporarily; future validation samples never enter the
proposal.

The certificate uses

\[
\rho_k^+=\min\{0.95,\widehat\rho_k+r_k(0.1)\},\qquad
m_{\rm eff,k}=\max\left\{1,
5\frac{1-\rho_k^+}{1+\rho_k^+}\right\},
\]

and supplies \(\widehat\Sigma/m_{\rm eff,k}\) to the QP. The cap states a
separated class \(\lambda\leq0.95\); it is not an unrestricted unknown-mixing
claim. The present radius is an algorithmic prototype whose validity for the
Gaussian Markov subclass must be audited before any new-seed preregistration.

All observations remain learning transitions and every fingerprint message is
charged. Constants were checked on four old seeds, so the full 32-seed run is
design calibration only. P1--P11 include high-temporal and low-scale safety,
certificate direction, resource accounting, and measured QP complexity.
Passing permits only a separate theory/identifiability audit, not a pilot.
