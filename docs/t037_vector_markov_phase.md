# T-037 dimension-free vector Markov phase theorem

## Scope and claim boundary

T-037 proves an exact finite-horizon risk identity for delayed vector linear
stochastic approximation with additive stationary Markov innovations.  The
identity is dimension-free with respect to the Markov state space: it needs
only the vector lag-covariance sequence and a lifted delay matrix.  It covers
finite-state, continuous-state, and non-Markov stationary generators having
the stated second moments.

It does **not** replace the sample matrix in general Markov TD by its mean.
For updates containing the multiplicative term
\((A(Z_t)-\bar A)e_{t-D}\), T-036 remains the exact finite-state result and a
Poisson/martingale remainder theorem is still required.  This boundary keeps
the present theorem exact rather than calling a variance proxy a TD proof.

## Model

Let \(e_t\in\mathbb R^d\) follow

\[
e_{t+1}=e_t-\eta A e_{t-D}+\eta\xi_t,
\tag{1}
\]

where the initial history \(e_{-D},\ldots,e_0\) is deterministic, \(A\) is
fixed, and \(\xi_t\) is zero-mean second-order stationary with

\[
K_k=\mathbb E[\xi_{t+k}\xi_t^\top],\qquad K_{-k}=K_k^\top.
\]

For \(q\) agent streams and predictable aggregation weights fixed over the
evaluated horizon, \(\xi_t=\sum_iw_i\xi_{i,t}\) and hence

\[
K_k(w)=\sum_{i,j}w_iw_jK_k^{ij}.
\tag{2}
\]

Thus cross-agent dependence enters through an identity, not an effective
sample-size assumption.

## Exact theorem

Stack \(x_t=(e_t^\top,e_{t-1}^\top,\ldots,e_{t-D}^\top)^\top\).  Let \(G_D\)
be the companion matrix for (1), \(J=[I_d,0,\ldots,0]\), and \(B=J^\top\).
For \(H_{T,s}=JG_D^{T-1-s}B\), direct iteration gives

\[
e_T=JG_D^Tx_0+\eta\sum_{s=0}^{T-1}H_{T,s}\xi_s.
\tag{3}
\]

Therefore, for every positive-semidefinite risk matrix \(Q\),

\[
\begin{aligned}
\mathbb E\|e_T\|_Q^2
&=\|JG_D^Tx_0\|_Q^2\\
&\quad+\eta^2\sum_{s,r=0}^{T-1}
 \operatorname{tr}\!\left(QH_{T,s}K_{s-r}H_{T,r}^\top\right).
\tag{4}
\end{aligned}
\]

Equation (4) is exact at every finite horizon.  It retains temporal
correlation, cross-agent covariance, vector geometry, delay, transient bias,
and the finite number of resource-feasible updates.

### Proof

The lifted recursion is \(x_{t+1}=G_Dx_t+\eta B\xi_t\).  Iterating it gives
(3).  The innovation sum has zero mean.  Expanding its outer product and
using stationarity yields

\[
\operatorname{Cov}(e_T)=\eta^2\sum_{s,r}H_{T,s}K_{s-r}H_{T,r}^\top.
\]

The identity
\(\mathbb E[e_T^\top Qe_T]=\mathbb E[e_T]^\top Q\mathbb E[e_T]
+\operatorname{tr}(Q\operatorname{Cov}(e_T))\) proves (4).  No independence
between times or agents is invoked.

## Finite-horizon dimension-free bound

Using \(|\operatorname{tr}(UV)|\leq\|U\|_{\rm op}\|V\|_*\), the stochastic
term in (4) obeys

\[
R_{\rm noise}(T)\leq\eta^2\sum_{s,r=0}^{T-1}
 \|Q^{1/2}H_{T,s}\|_{\rm op}
 \|K_{s-r}\|_*
 \|Q^{1/2}H_{T,r}\|_{\rm op}.
\tag{5}
\]

This bound has no dependence on the number of Markov modes.  If
\(\|Q^{1/2}JG_D^\ell B\|_{\rm op}\leq C_Dr_D^\ell\), \(r_D<1\), then

\[
R_{\rm noise}(T)\leq
\frac{\eta^2C_D^2}{1-r_D^2}
\sum_{k\in\mathbb Z}r_D^{|k|}\|K_k\|_*.
\tag{6}
\]

The weighted covariance series is finite whenever the lag covariances are
absolutely summable.  Delay affects both \(C_D,r_D\) and the feasible horizon;
it is not hidden in a generic constant.

## Correlation and dual-budget corollary

For a separable equicorrelated AR(1) family,

\[
K_k(q)=\sigma^2\left(\rho+\frac{1-\rho}{q}\right)
       \lambda^{|k|}\Sigma.
\tag{7}
\]

Thus (4) recovers
\(n_{\rm eff}=q/[1+(q-1)\rho]\) exactly at equal horizons.  With per-update
message cost \(c_m(q)\), stride \(b\), message budget \(B_m\), environment
budget \(B_e\), and the registered delay charge,

\[
T(q,b,D)=\min\left\{
\left\lfloor\frac{B_m}{c_m(q)}\right\rfloor,
\left\lfloor\frac{[B_e-D]_+}{b}\right\rfloor
\right\}.
\tag{8}
\]

Substituting (7)--(8) into (4) gives an exact, computable phase classifier:
larger \(q\) can yield speedup, correlation saturation, or finite-resource
reversal because it changes both \(K_k\) and \(T\).

## Verification and consequence

The implementation is independently checked against the T-035 scalar formula
for delays 0, 1, and 3; against coordinatewise closed forms for diagonal
vector iid recursions; and against (5).  Budget tests separately charge
message cost, stride, and delay.

T-037 closes the dimension-free additive part of P2/P7 and supplies the exact
object for a prospective CPU phase-map preregistration.  It does not close
the multiplicative Markov-TD remainder, predictable within-horizon action
changes, the matching minimax lower bound, AC-9c, or the quantitative SDDE
bridge.  Those remain hard gates before the main ICML claim is authorized.
