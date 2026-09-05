# T-048 scheduled PR risk and probe-cost oracle theorem

## Scope

T-048 closes the three proof obligations needed before an outcome-free
EXP-021A static design: exact tail Polyak--Ruppert risk for a deterministic
time-varying schedule, a mixing-corrected correlation certificate, and a
finite-budget oracle inequality that charges every probe resource. The result
applies to the T-047 common/private additive Markov model. Schedule selection
is based on an independent probe stream; it is not allowed to reuse the
learning innovations whose risk is evaluated.

## Theorem 1: exact scheduled PR risk

Let \(C\) be the stable delayed companion matrix, let \(S\) select the current
iterate, and let \(J=S^\top\) inject an innovation. For a deterministic
prefix-participation schedule \(q_{0:N-1}\), define

\[
 \bar e_{b:N}=\frac1{N-b}\sum_{t=b+1}^N e_t
\]

and, for innovation time \(s\),

\[
 H_s^{(b,N)}=\frac{\eta}{N-b}
 \sum_{t=\max\{b+1,s+1\}}^N SC^{t-1-s}J .
\]

Under the assumptions of T-047,

\[
 \operatorname{Cov}(\bar e_{b:N})
 =\sum_{s,r=0}^{N-1}
 \left[\rho+\frac{1-\rho}{\max(q_s,q_r)}\right]
 H_s^{(b,N)}K_{s-r}H_r^{(b,N)\top}.                 \tag{1}
\]

The mean is \((N-b)^{-1}\sum_{t=b+1}^NSC^tx_0\). Therefore every fixed
schedule has exact risk \(R_\pi(\rho)=a_\pi+b_\pi\rho\).

### Proof

Iterating the lifted recursion and averaging gives the stated mean and the
linear innovation impulse \(H_s^{(b,N)}\). The T-047 prefix-overlap identity
gives

\[
 \operatorname{Cov}(\bar\xi_s,\bar\xi_r)
 =\left[\rho+(1-\rho)/\max(q_s,q_r)\right]K_{s-r}.
\]

Substitution into the covariance of the linear readout proves (1). Its
coefficient is affine in \(\rho\), so the quadratic risk is affine as well.
No Gaussian or independent-in-time replacement is used.

## Assumption 2: observable bounded probe block

At a frozen parameter, the server collects \(n\) fully charged probe rounds
with \(q_p\ge2\) and physical stride \(b_p\). Let \(Y_{\ell i}\in[-1,1]\)
be a public scalar projection of agent \(i\)'s stochastic update. Assume:

(a) the joint probe process is stationary with common mean \(\mu\), marginal
variance \(\nu\ge\nu_{\min}>0\), and pair covariance \(\rho\nu\);

(b) its absolute-regularity coefficients obey the certified envelope

\[
 \beta(k)\le C_\beta\lambda^k,\quad 0\le\lambda<1;
\]

(c) the beta coefficient uses the normalization for which the total-variation
coupling error of \(n\) stride-\(b_p\) observations is at most
\((n-1)C_\beta\lambda^{b_p}\).

The bounded projection, clipping rule, \(C_\beta\), \(\lambda\), and
\(\nu_{\min}\) are public preregistered quantities. This is a certified-mixing
theorem, not an unknown-mixing guarantee.

## Theorem 2: mixing-corrected correlation interval

For each probe time define

\[
 A_\ell=\frac1{q_p}\sum_iY_{\ell i},\quad
 B_\ell=\frac1{q_p}\sum_iY_{\ell i}^2,\quad
 G_\ell=\frac{(\sum_iY_{\ell i})^2-\sum_iY_{\ell i}^2}
 {q_p(q_p-1)}.
\]

Their expectations are \(\mu\), \(\mu^2+\nu\), and
\(\mu^2+\rho\nu\). Let

\[
 m_\beta=(n-1)C_\beta\lambda^{b_p},\quad
 a=\alpha/3-m_\beta.
\]

If \(a>0\), set

\[
 r_n=\sqrt{\frac2n\log\frac2a}.                         \tag{2}
\]

Construct radius-\(r_n\) intervals for the three expectations, propagate the
interval for \(\mu\) through the square, and divide the resulting nonnegative
covariance interval by the positive variance interval. The clipped result
\(I_n=[\rho_L,\rho_U]\subseteq[0,1]\) obeys

\[
 \Pr\{\rho\notin I_n\}\le\alpha.                        \tag{3}
\]

If \(a\le0\), or if interval arithmetic is incompatible with the structural
variance lower bound, the procedure returns \([0,1]\), never a fabricated
certificate. If only the variance lower endpoint is zero, the upper
correlation endpoint remains one. For preregistered checkpoints \(n_j\),
assigning error levels \(\alpha_j\) with
\(\sum_j\alpha_j\le\alpha\) makes (3) simultaneous at every checkpoint.

### Proof

Each of \(A_\ell,B_\ell,G_\ell\) lies in an interval of length at most two.
Couple the stride-spaced block to independent copies. Hoeffding's inequality
and the coupling error give, for each summary mean,

\[
 \Pr\{|\widehat M-M|>r\}
 \le2e^{-nr^2/2}+m_\beta.
\]

Equation (2) makes the right side \(\alpha/3\). A union bound covers all three
moments. On this event, interval arithmetic contains \(\mu^2\), \(\nu\), and
\(\rho\nu\); division by the positive variance endpoint and clipping preserve
coverage. A second union bound proves the registered-checkpoint statement.

## Theorem 3: probe-cost oracle inequality

Let \(\Pi_B\) be the full-budget schedule library and \(\Pi_{B'}\) the library
after subtracting all probe messages, probe environment steps, and delay.
Both libraries include their exact affine PR-risk tables. Given
\(I=[\rho_L,\rho_U]\), select

\[
 \widehat\pi\in\arg\min_{\pi\in\Pi_{B'}}
 \max_{r\in I}R_\pi(B',r).                              \tag{4}
\]

On the event \(\rho\in I\),

\[
 R_{\widehat\pi}(B',\rho)-
 \min_{\pi\in\Pi_B}R_\pi(B,\rho)
 \le \mathcal E(B,B',I),                                \tag{5}
\]

where the exact computable certificate is

\[
 \mathcal E(B,B',I)=
 \max_{r\in\{\rho_L,\rho_U\}}
 \max_{\pi\in\Pi_B}
 \{R_{\widehat\pi}(B',r)-R_\pi(B,r)\}.                  \tag{6}
\]

Thus (6) includes the opportunity cost of both resources and delay rather
than comparing only post-probe horizons. If the excess on the complement of
the coverage event is bounded by \(R_{\max}\), then

\[
 \mathbb E R_{\widehat\pi}(B',\rho)-R_B^*(\rho)
 \le \mathcal E(B,B',I)
 +\alpha[R_{\max}-\mathcal E(B,B',I)]_+.                \tag{7}
\]

For the post-probe oracle alone, if
\(L=\max_{\pi\in\Pi_{B'}}|b_\pi|\), robust minimization also gives the generic
bound \(L(\rho_U-\rho_L)\).

### Proof

For fixed selected \(\widehat\pi\), its risk minus the full-budget lower
envelope equals the maximum over \(\pi\in\Pi_B\) of finitely many affine
functions. This is convex in \(\rho\), so its maximum on \(I\) occurs at an
endpoint, proving (5)--(6). On the complement of the coverage event, the
excess is at most \(R_{\max}\), which proves (7) by conditioning. The generic
post-probe bound follows by comparing (4) with the true post-probe oracle and
using the Lipschitz constant of its affine risk.

## Complexity and remaining boundary

Computing the three probe summaries costs \(O(nq_p)\) arithmetic and constant
state per registered checkpoint. Online schedule selection is \(O(|\Pi|)\);
learning-gradient aggregation remains \(O(qd)\). There is no matrix inversion
or preconditioner online.

T-048 matches the necessary
\(\Delta_\rho^{-2}\log(1/\alpha)\) identification dependence on a separated
correlation class. The elementary coupling proof additionally requires a
stride large enough that
\((n-1)C_\beta\lambda^{b_p}<\alpha/3\), which incurs a logarithmic mixing
overhead. A sharp constant-factor match to the controlled-belief lower bound
remains open and may not be claimed. The next admissible step is an
outcome-free CPU static design; no sampled or GPU experiment is authorized by
this theorem alone.
