# T-035 exact scalar phase theorem

## Scope

T-035 establishes the exact scalar Gaussian base case of the T-034 main
theorem.  It includes cross-agent correlation, temporal AR(1) mixing,
parameter delay, and a resource-determined finite update horizon.  It creates
no scientific trajectory and does not claim the general affine Markov or
adaptive-policy result.

## Model

For strong-monotonicity coefficient \(\mu>0\), step size \(\eta>0\), and
integer delay \(D\ge0\), consider

\[
e_{t+1}=e_t-\eta\mu e_{t-D}+\eta\bar\xi_t,
\qquad e_{-D}=\cdots=e_0.
\]

Each of \(q\) agents has stationary innovation variance \(\sigma^2\), common
AR coefficient \(\lambda\in[0,1)\), and equicorrelation \(\rho\).  Hence

\[
\operatorname{Cov}(\bar\xi_s,\bar\xi_t)
=\sigma^2v_q(\rho)\lambda^{|s-t|},
\qquad
v_q(\rho)=\rho+\frac{1-\rho}{q}.
\]

A resource budget \(B\) and per-update cost \(c(q)\) permit
\(N_q=\lfloor B/c(q)\rfloor\) updates.

## Proposition 1: exact finite-horizon risk

Let \(F_D\) be the delay companion matrix with first row
\(e_1^\top-\eta\mu e_{D+1}^\top\), and let
\(h_j=e_1^\top F_D^j e_1\).  With constant initial history
\(z_0=e_0{\bf1}_{D+1}\),

\[
\begin{aligned}
\mathbb E[e_{N_q}^2]
={}&(e_1^\top F_D^{N_q}z_0)^2\\
&+\eta^2\sigma^2v_q(\rho)
\sum_{s=0}^{N_q-1}\sum_{r=0}^{N_q-1}
h_{N_q-1-s}h_{N_q-1-r}\lambda^{|s-r|}.
\end{aligned}
\]

### Proof

The companion form gives
\(z_{t+1}=F_Dz_t+\eta e_1\bar\xi_t\).  Iterating it to \(N_q\)
separates the deterministic initial-history term from a linear combination of
zero-mean innovations.  Squaring and substituting their stationary AR(1)
cross-time covariance yields the displayed double sum.  No independence
between times is used.

For \(D=0\) and \(\lambda=0\), this reduces to

\[
(1-\eta\mu)^{2N_q}e_0^2
+\eta^2\sigma^2v_q(\rho)
\frac{1-(1-\eta\mu)^{2N_q}}
 {1-(1-\eta\mu)^2}.
\]

## Corollary 1: dependence-adjusted effective agents

At a fixed update horizon and delay, the complete stochastic term scales with
\(v_q(\rho)\).  Thus

\[
n_{\rm eff}(q,\rho)=v_q(\rho)^{-1}
=\frac{q}{1+(q-1)\rho}.
\]

It equals \(q\) for independent streams and one for perfectly correlated
streams, proving exact speedup and saturation in this base class.

## Corollary 2: budget-induced reversal

The noise-dominated resource proxy with affine cost
\(c(q)=c_0+c_1q\) is proportional to

\[
\left(\rho+\frac{1-\rho}{q}\right)(c_0+c_1q).
\]

For \(0<\rho<1\), its continuous minimizer is

\[
q^*=\sqrt{\frac{(1-\rho)c_0}{\rho c_1}}.
\]

The exact proposition additionally retains the contraction term through
\(N_q\).  Consequently, at short horizons a larger independent batch may
have lower innovation variance and higher total error because it performs too
few contraction steps.  The registered unit test constructs this reversal
without changing the marginal innovation law.

## What remains open

The next proof step must replace the scalar AR covariance by the long-run
matrix covariance of affine Markov temporal-difference innovations, retain
sample/error dependence through a Poisson-equation martingale decomposition,
and establish matching predictable-policy lower bounds.  T-035 is not used to
authorize a new sampled experiment or an SDDE claim.
