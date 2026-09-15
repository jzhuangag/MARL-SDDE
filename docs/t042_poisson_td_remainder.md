# T-042 Poisson decomposition for delayed multiplicative Markov TD

## Decision

T-042 closes the exact algebraic decomposition that was missing after T-036
and T-037.  It does **not** yet close the full finite-time stability theorem:
the iterate-increment remainder must still be absorbed into a Lyapunov bound
without erasing the participation phase.  This distinction is theorem-facing
and prevents an exact identity from being overstated as a convergence rate.

## Setting

Let \(Z_t\) be an exogenous ergodic finite-state Markov chain with transition
matrix \(P\) and stationary law \(\pi\).  Let

\[
F(z)=A(z)-\bar A,\qquad \sum_z\pi(z)F(z)=0,
\]

where \(F(z)\) may be matrix-valued.  The delayed TD error contains the
sample--iterate-coupled term \(F(Z_t)e_{t-D}\); it is not replaced by an
independent innovation.

On a finite chain, the canonical Poisson solution is

\[
H=(I-P+\mathbf 1\pi^\top)^{-1}F,
\qquad H-PH=F,\qquad \pi^\top H=0.
\]

## Theorem 1: exact predictable decomposition

For every horizon \(T\) and every \(\mathcal F_t\)-measurable vector \(y_t\),
define

\[
M_{t+1}=H(Z_{t+1})-(PH)(Z_t).
\]

Then \(\mathbb E[M_{t+1}\mid\mathcal F_t]=0\), and pathwise

\[
\begin{aligned}
\sum_{t=0}^{T-1}F(Z_t)y_t
={}&\sum_{t=0}^{T-1}M_{t+1}y_t
 +H(Z_0)y_0-H(Z_T)y_{T-1}\\
&+\sum_{t=1}^{T-1}H(Z_t)(y_t-y_{t-1}).
\end{aligned}
\]

In delayed TD, \(y_t=e_{t-D}\) is predictable before \(Z_{t+1}\) is drawn.
The three right-hand components are respectively a martingale transform,
boundary/coboundary terms, and the delayed iterate-increment remainder.

### Proof

The Poisson equation gives

\[
F(Z_t)=H(Z_t)-H(Z_{t+1})+M_{t+1}.
\]

Multiply by \(y_t\), sum, and apply discrete summation by parts to
\(\sum_t[H(Z_t)-H(Z_{t+1})]y_t\).  Conditional centering of \(M_{t+1}\)
follows directly from the Markov property.  No independence between
\(F(Z_t)\) and \(y_t\) is used.

## Theorem 2: exact finite-state moment constant

For matrix-valued \(H\), define

\[
V_H(z)=\sum_bP_{zb}
 [H(b)-(PH)(z)]^\top[H(b)-(PH)(z)],
\qquad
v_H=\max_z\lambda_{\max}(V_H(z)).
\]

Martingale orthogonality yields, for every predictable \(y_t\),

\[
\mathbb E\left\|\sum_{t=0}^{T-1}M_{t+1}y_t\right\|^2
=\sum_{t=0}^{T-1}\mathbb E[y_t^\top V_H(Z_t)y_t]
\le v_H\sum_{t=0}^{T-1}\mathbb E\|y_t\|^2.
\]

Writing \(h_H=\max_z\|H(z)\|_{\rm op}\), the remaining pathwise norm is at
most

\[
h_H\left(\|y_0\|+\|y_{T-1}\|
 +\sum_{t=1}^{T-1}\|y_t-y_{t-1}\|\right).
\]

Both \(v_H\) and \(h_H\) are computable from \((P,F)\); no unspecified
``mixing factor'' is fitted after seeing outcomes.

## Corollary: terminal impulse-response decomposition

For deterministic left weights \(W_t\), the same argument gives

\[
\begin{aligned}
\sum_{t=0}^{T-1}W_tF(Z_t)y_t
={}&\sum_{t=0}^{T-1}W_tM_{t+1}y_t
 +W_0H(Z_0)y_0-W_{T-1}H(Z_T)y_{T-1}\\
&+\sum_{t=1}^{T-1}(W_t-W_{t-1})H(Z_t)y_t\\
&+\sum_{t=1}^{T-1}W_{t-1}H(Z_t)(y_t-y_{t-1}).
\end{aligned}
\]

This is the form needed for terminal TD risk: \(W_t\) is the deterministic
lifted impulse response from update \(t\) to the terminal iterate.  It exposes
separately the impulse-response increment and delayed-iterate increment, so a
future proof cannot hide either term inside an unspecified constant.  Its
martingale term obeys the corresponding bound with
\(v_H\sum_t\|W_t\|_{\rm op}^2\mathbb E\|y_t\|^2\).

## Consequence and exact remaining obligation

For

\[
e_{t+1}=e_t-\eta\bar A e_{t-D}+\eta\xi(Z_t)
          -\eta F(Z_t)e_{t-D},
\]

Theorem 1 converts the last term into an MDS part plus a boundary of order
\(\eta h_H\) and an increment term containing
\(e_{t-D}-e_{t-1-D}\).  Each increment itself contains a factor \(\eta\),
but summing its norm naively over \(T\) gives a potentially vacuous
\(O(\eta T)\) loss.  The next lemma must use the delayed Lyapunov contraction
to absorb this term (typically blockwise or by a small-gain argument) while
retaining the \(q,\rho,D\), mixing, and dual-budget dependence.

Accordingly:

- the Poisson identity, martingale isometry, and finite-state constants are
  proved exactly;
- the earlier invalid iid substitution is no longer needed;
- a general nonasymptotic multiplicative-TD risk theorem is still open until
  the increment term is nonvacuously absorbed;
- T-037 remains the exact main theorem if that absorption destroys the sharp
  speedup/saturation/reversal phase.

## Verification

CPU tests check the Poisson residual and canonical centering, exact pathwise
decomposition for delays 0, 1, and 3, conditional martingale centering,
martingale isometry by complete path enumeration with history-dependent
predictable iterates, the reported variance-operator bound, and the weighted
terminal impulse-response identity.
