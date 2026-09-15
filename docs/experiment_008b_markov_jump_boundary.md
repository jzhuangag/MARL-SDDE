# EXP-008B preregistration: exact Markov-jump boundary

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp008b_preregistration_v1

## Question

Does temporal Markov persistence change the exact mean-square stability
boundary after same-time agent count, conditional cross-agent correlation, and
heterogeneous delay have been controlled explicitly?

EXP-008B is a deterministic theorem audit. It is not a Monte Carlo performance
experiment and it is not used to tune a controller.

## Frozen Markov-modulated TD model

The TD feature vectors and discount are

\[
\phi(0)=2^{-1/2}(1,1)^\top,\qquad
\phi(1)=2^{-1/2}(1,-1)^\top,\qquad \gamma=0.9.
\]

The four possible Jacobians are

\[
H_{ss'}=\phi(s)\{\phi(s)-\gamma\phi(s')\}^{\top}.
\]

An exogenous regime \(Z_k\in\{0,1\}\) has transition matrix

\[
P_p=\begin{bmatrix}p&1-p\\1-p&p\end{bmatrix}.
\]

Conditional on \(Z_k=z\), each agent's transition-pair index has the following
registered distribution, in the order \(00,01,10,11\):

\[
w_0=(0.05,0.85,0.05,0.05),\qquad
w_1=(0.05,0.05,0.85,0.05).
\]

Thus the stationary same-time distribution is fixed while \(p\) changes only
temporal persistence. Conditional cross-agent dependence uses the registered
pair-sharing construction: every agent selects a common conditional draw with
probability \(\sqrt{\rho}\), otherwise an independent conditional draw.
Consequently each one-agent conditional law is invariant to \(\rho\), while
two different agents share the same draw with probability \(\rho\).

The homogeneous delayed update is

\[
e_{k+1}=e_k-\frac{\eta}{q}\sum_{i=1}^{q}
H_{i,k}e_{k-\tau_i}.
\]

## Registered cells

- Agent counts: \(q\in\{1,2,3\}\).
- Delay profiles:
  - \(D=0\): all delays are zero;
  - \(D=2\): \((2)\), \((0,2)\), and \((0,1,2)\) for \(q=1,2,3\).
- Regime persistence: \(p\in\{0.5,0.9,0.98\}\).
- Conditional pair-sharing correlation: \(\rho\in\{0,0.9\}\).
- Total exact cells: 36.

The mode-conditioned covariance operator is

\[
[\mathfrak L_\eta(P)]^b
=\sum_a [P_p]_{ab}\,
\mathbb E[M_{a,k}P^aM_{a,k}^{\top}\mid Z_k=a].
\]

Its exact first positive stability boundary is the endpoint of the connected
small-step interval satisfying \(\rho(\mathfrak L_\eta)<1\).

Three comparisons are frozen:

1. the exact Markov-jump mean-square boundary;
2. the boundary after replacing the regime sequence by i.i.d. stationary
   regimes while retaining the identical one-step distribution;
3. the existing same-time scalar rule using the exact stationary aggregate
   curvature but no temporal-mixing inflation.

The exact mode-conditioned first-moment boundary is also recorded so that
mean and mean-square failure cannot be conflated.

## Preregistered numerical checks

1. **Independent construction.** A directly enumerated conditional
   \(M\otimes M\) operator and the moment-form operator must differ by at most
   \(10^{-11}\) in a registered \(q=2,D=2\) cell.
2. **Boundary validity.** All 36 boundaries must be finite and positive; the
   radius below each boundary must be below one, the radius above it must be
   above one, and all reported eigen-residuals must be at most \(10^{-7}\).
3. **I.i.d. reduction.** At \(p=0.5\), the Markov-jump and stationary-i.i.d.
   boundaries must agree to relative error at most \(10^{-7}\) in all 12
   cells.
4. **One-agent correlation invariance.** For \(q=1\), changing \(\rho\) must
   change neither exact boundary nor the scalar step by more than relative
   \(10^{-10}\).

Failure of any numerical check invalidates the implementation and prevents
scientific interpretation.

## Preregistered scientific gates

These are reported separately rather than combined into a single pass count.

1. **Temporal persistence is active.** At \(p=0.98\), the exact
   Markov-to-i.i.d. boundary ratio is at most 0.8 in at least 6 of 12
   \((q,D,\rho)\) cells.
2. **The i.i.d. scalar rule is Markov-safe.** Its exact Markov spectral radius
   is below one in all 36 cells. Any failure withdraws the uninflated scalar
   rule as a Markov theorem candidate and triggers a mixing-aware replacement.
3. **Agent count remains mechanistic.** For every \((p,D)\), the \(q=3\) to
   \(q=1\) boundary gain under \(\rho=0\) is no smaller than the corresponding
   gain under \(\rho=0.9\), up to numerical tolerance.
4. **Correlation-limited saturation is visible.** In at least four of the six
   \((p,D)\) slices, the \(q=3\) gain under \(\rho=0.9\) is at most 1.10.
5. **Delay and persistence interact.** In at least four of six \((q,\rho)\)
   slices, the \(D=2\) to \(D=0\) boundary ratio at \(p=0.98\) is no larger
   than the corresponding ratio at \(p=0.5\).
6. **Mean stability is insufficient.** In at least 24 of 36 cells, the exact
   mean-square boundary is at most 80% of the exact Markov first-moment
   boundary.

## Decision rule

- If the i.i.d. scalar rule fails exact Markov safety while persistence is
  active, the paper's algorithm must estimate a temporal inflation or
  contraction margin; same-time correlation estimation alone is insufficient.
- If the scalar rule remains safe but becomes substantially more conservative
  as persistence rises, the online method may retain the rule and estimate
  mixing only to recover efficiency.
- If persistence is inactive in this frozen model, no temporal-adaptive claim
  is promoted from EXP-008B; a different model may be used only in a newly
  preregistered experiment and must not overwrite this result.
- Agent-count claims are retained only if the registered \(q\)-gates pass.
