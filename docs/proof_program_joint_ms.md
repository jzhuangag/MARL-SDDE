# Proof program: correlation-limited mean-square stability

## Material Passport

- Origin Skill: academic-research-suite / publishable-academic-writing
- Origin Mode: theorem development
- Origin Date: 2026-07-30
- Verification Status: PARTIALLY_VERIFIED
- Version Label: joint_ms_proof_v1

## 1. Model and notation

Let \(H_{i,k}\in\mathbb R^{d\times d}\) denote the random linearization of
agent \(i\)'s stochastic update at time \(k\), and let

\[
\bar H_{q,k}=\frac1q\sum_{i=1}^{q}H_{i,k}.
\]

For temporal-difference learning,

\[
H_{i,k}
=
\phi(S_{i,k})
[\phi(S_{i,k})-\gamma\phi(S'_{i,k})]^\top .
\]

The mean operator is \(A=\mathbb E[H_{i,k}]\).  Define

\[
B_{\rm d}=\mathbb E[H_{i,k}^\top H_{i,k}],
\qquad
B_{\rm o}=\mathbb E[H_{i,k}^\top H_{j,k}],\quad i\ne j.
\]

The following assumptions are used only for the two proved propositions.

**Assumption 1 (same-time exchangeability).**  For every \(k\), (a) the
collection \((H_{1,k},\ldots,H_{q,k})\) is exchangeable, (b) its second moments
are finite, and (c) \(\mathbb E[H_{i,k}]=A\).

**Assumption 2 (fresh no-delay recursion).**  The error follows
\(e_{k+1}=(I-\eta\bar H_{q,k})e_k\), and
\(\bar H_{q,k}\) is independent of \(e_k\).

Assumption 2 is deliberately not imposed on delayed Markov TD.  Removing it is
the main proof obligation in Section 5.

## 2. Exact aggregate-Jacobian identity

**Proposition 1 (exchangeable Jacobian second moment).**  Under Assumption 1,

\[
Q_q
\triangleq
\mathbb E[\bar H_{q,k}^\top\bar H_{q,k}]
=
B_{\rm o}+\frac{B_{\rm d}-B_{\rm o}}{q}.
\]

If each agent uses a shared transition-pair draw with pairwise sharing
probability \(\rho\), and otherwise uses an independent draw with the same
mean \(A\), then

\[
B_{\rm o}=\rho B_{\rm d}+(1-\rho)A^\top A
\]

and hence

\[
Q_q
=
\left[\rho+\frac{1-\rho}{q}\right]B_{\rm d}
+\left[1-\rho-\frac{1-\rho}{q}\right]A^\top A.
\]

*Proof.*  Expanding the product gives

\[
Q_q
=
\frac1{q^2}\sum_{i=1}^{q}\mathbb E[H_i^\top H_i]
+\frac1{q^2}\sum_{i\ne j}\mathbb E[H_i^\top H_j].
\]

Exchangeability makes the \(q\) diagonal terms equal to \(B_{\rm d}\) and the
\(q(q-1)\) ordered off-diagonal terms equal to \(B_{\rm o}\), proving the first
identity.  In the pair-sharing model, both agents use the same draw with
probability \(\rho\), which contributes \(B_{\rm d}\).  In every other case
their draws are independent with mean \(A\), which contributes \(A^\top A\).
Substitution gives the second identity. \(\square\)

This proposition is the first explicit agent-count result.  Since

\[
B_{\rm d}-A^\top A
=
\mathbb E[(H-A)^\top(H-A)]\succeq0,
\]

\(Q_q\) decreases in the Loewner order with \(q\) when \(\rho<1\), but is
constant in \(q\) when \(\rho=1\).  Thus participation improves the
random-Jacobian stability margin only through the noncommon component.

## 3. Exact no-delay mean-square contraction

Let

\[
\mu=\lambda_{\min}\!\left(\frac{A+A^\top}{2}\right)>0,
\qquad
K_q=\lambda_{\max}(Q_q).
\]

**Proposition 2 (correlation-limited no-delay contraction).**  Under
Assumptions 1--2, every step size satisfying

\[
0<\eta<\frac{2\mu}{K_q}
\]

obeys

\[
\mathbb E[\|e_{k+1}\|^2\mid e_k]
\le
\left(1-2\mu\eta+K_q\eta^2\right)\|e_k\|^2.
\]

Consequently,

\[
\mathbb E\|e_k\|^2
\le
\left(1-2\mu\eta+K_q\eta^2\right)^k\|e_0\|^2.
\]

*Proof.*  Conditional independence gives

\[
\begin{aligned}
\mathbb E[\|e_{k+1}\|^2\mid e_k]
&=
e_k^\top
\mathbb E[(I-\eta\bar H_q)^\top(I-\eta\bar H_q)]
e_k\\
&=
e_k^\top
[I-\eta(A+A^\top)+\eta^2Q_q]e_k.
\end{aligned}
\]

The definitions of \(\mu\) and \(K_q\) yield the stated upper bound.  Its
contraction decrement is
\(\eta(2\mu-\eta K_q)>0\), and iteration proves the final claim.
\(\square\)

This result converts the exact participation identity into a stability law:
nominal linear speedup is unavailable when common correlation prevents
\(K_q\) from decreasing.

## 4. Additive-noise corollary

Suppose

\[
e_{k+1}
=(I-\eta\bar H_{q,k})e_k+\eta\zeta_{q,k},
\]

where the cross term has conditional expectation zero and
\(\mathbb E[\|\zeta_{q,k}\|^2\mid e_k]\le\sigma_q^2\).  With
\(r_q=1-2\mu\eta+K_q\eta^2<1\), Proposition 2 gives

\[
\mathbb E\|e_k\|^2
\le
r_q^k\|e_0\|^2
+\eta^2\sigma_q^2\frac{1-r_q^k}{1-r_q}.
\]

In particular,

\[
\limsup_{k\to\infty}\mathbb E\|e_k\|^2
\le
\frac{\eta\sigma_q^2}{2\mu-\eta K_q}.
\]

The multiplicative quantity \(K_q\) controls the stable step, while the
additive quantity \(\sigma_q^2\) controls the residual floor.  Combining them
into one variance parameter would erase the mechanism exposed by EXP-007B--D.

## 5. Exact heterogeneous-delay mean-square theorem

The next result removes the common-delay restriction without imposing a norm
bound on each realization.  It retains temporal independence so that the
lifted state and the current random Jacobians are independent.

Let \(D=\max_i\tau_i\), define

\[
X_k=
[e_k^\top,e_{k-1}^\top,\ldots,e_{k-D}^\top]^\top
\in\mathbb R^{d(D+1)},
\]

and let \(S_\ell\in\mathbb R^{d\times d(D+1)}\) select the
\(\ell\)-th block of \(X_k\).  Let
\(J=S_0^\top\), and define the deterministic shift matrix

\[
C=
JS_0+\sum_{\ell=1}^{D}S_\ell^\top S_{\ell-1}.
\]

The lifted recursion is

\[
X_{k+1}=M_k(\eta)X_k,
\qquad
M_k(\eta)
=
C-\frac{\eta}{q}\sum_{i=1}^{q}
JH_{i,k}S_{\tau_i}.
\]

**Assumption 3 (fresh heterogeneous-delay matrices).**  The vectors
\((H_{1,k},\ldots,H_{q,k})\) are (a) independent and identically distributed
over \(k\) and (b) independent of \(X_0\).  Their same-time law satisfies
Assumption 1.  The deterministic delays satisfy
\(\tau_i\in\{0,\ldots,D\}\).

For a symmetric matrix \(P\), define the positive linear operator

\[
\mathcal L_\eta(P)
=
\mathbb E[M_k(\eta)P M_k(\eta)^\top].
\]

**Theorem 1 (exact heterogeneous-delay mean-square stability).**  Under
Assumptions 1 and 3, the following statements are equivalent:

1. for every square-integrable initial lifted state \(X_0\),
   \(\mathbb E\|X_k\|^2\to0\);
2. the spectral radius of \(\mathcal L_\eta\) is strictly below one;
3. \(\rho(\mathbb E[M_k(\eta)\otimes M_k(\eta)])<1\).

When these conditions hold, for every
\(\varepsilon\in(0,1-\rho(\mathcal L_\eta))\), there is a finite
\(C_\varepsilon\) such that

\[
\mathbb E\|X_k\|^2
\le
C_\varepsilon
[\rho(\mathcal L_\eta)+\varepsilon]^k
\mathbb E\|X_0\|^2.
\]

*Proof.*  Let \(P_k=\mathbb E[X_kX_k^\top]\).  Temporal independence in
Assumption 3 gives

\[
P_{k+1}
=
\mathbb E[M_kP_kM_k^\top]
=
\mathcal L_\eta(P_k),
\]

and hence \(P_k=\mathcal L_\eta^k(P_0)\).  Vectorization yields

\[
\operatorname{vec}(P_{k+1})
=
\mathbb E[M_k\otimes M_k]\operatorname{vec}(P_k),
\]

so statements 2 and 3 are equivalent.  If the spectral radius is below one,
the finite-dimensional spectral-radius formula implies the displayed
geometric bound, and
\(\mathbb E\|X_k\|^2=\operatorname{tr}(P_k)\to0\).

Conversely, \(\mathcal L_\eta\) preserves the cone of positive semidefinite
matrices.  The Perron--Frobenius theorem for a cone-preserving
finite-dimensional linear map gives a nonzero \(P_\star\succeq0\) satisfying
\(\mathcal L_\eta(P_\star)=\rho(\mathcal L_\eta)P_\star\).  Choose a
zero-mean square-integrable \(X_0\) with covariance \(P_\star\).  If
\(\rho(\mathcal L_\eta)\ge1\), then
\(\operatorname{tr}(P_k)=\rho(\mathcal L_\eta)^k
\operatorname{tr}(P_\star)\) cannot converge to zero, contradicting
statement 1. \(\square\)

Theorem 1 is an exact stability characterization.  It does not use the
parallel-sum approximation and does not replace random Jacobians by their
mean.

For computation, set

\[
\mathcal R_{\rm d}(Y)=\mathbb E[H_iYH_i^\top],
\qquad
\mathcal R_{\rm o}(Y)=\mathbb E[H_iYH_j^\top],\quad i\ne j.
\]

Let \(n_\ell=|\{i:\tau_i=\ell\}|\) and
\(\bar G=\sum_{\ell=0}^{D}n_\ell JAS_\ell\).  Direct expansion gives

\[
\begin{aligned}
\mathcal L_\eta(P)
={}&
CPC^\top
-\frac{\eta}{q}
(\bar GPC^\top+CP\bar G^\top)\\
&+\frac{\eta^2}{q^2}J
\left[
\sum_{\ell=0}^{D}
n_\ell\mathcal R_{\rm d}(P_{\ell\ell})
+\sum_{\ell=0}^{D}
n_\ell(n_\ell-1)\mathcal R_{\rm o}(P_{\ell\ell})\right.\\
&\left.\quad+
\sum_{\ell\ne m}
n_\ell n_m\mathcal R_{\rm o}(P_{\ell m})
\right]J^\top,
\end{aligned}
\]

where \(P_{\ell m}=S_\ell P S_m^\top\).  Under the registered pair-sharing
model,

\[
\mathcal R_{\rm o}(Y)
=
\rho\mathcal R_{\rm d}(Y)
+(1-\rho)AYA^\top.
\]

This representation applies \(\mathcal L_\eta\) without materializing the
\(d^2(D+1)^2\)-dimensional Kronecker matrix.  Agent count and the complete
delay histogram appear explicitly through \(n_\ell\), rather than only through
the maximum delay.

## 6. Exact finite-state Markov jump theorem

The finite-state Markov case also admits an exact lifted characterization,
although its direct mode space may be too large for an online algorithm.
Let \(Z_k\) be an exogenous Markov chain on
\(\mathcal Z=\{1,\ldots,m\}\) with transition probabilities \(p_{ab}\).
Conditional on \(Z_k=a,Z_{k+1}=b\), let the lifted update matrix be
\(M_{ab,k}\).

**Assumption 4 (exogenous Markov modes).**  (a) The transition law of \(Z_k\)
does not depend on \(X_k\); (b) conditional on \(Z_k=a,Z_{k+1}=b\),
\(M_{ab,k}\) is independent of \(X_k\); and (c) all conditional second moments
of \(M_{ab,k}\) are finite.

For \(P=(P^1,\ldots,P^m)\), define the block positive operator

\[
[\mathfrak L_\eta(P)]^b
=
\sum_{a=1}^{m}p_{ab}
\mathbb E[M_{ab,k}P^aM_{ab,k}^\top\mid a,b].
\]

**Theorem 2 (exact Markov jump mean-square stability).**  Under Assumption 4,
the homogeneous lifted recursion is mean-square stable for every
square-integrable initial state and initial mode distribution if and only if

\[
\rho(\mathfrak L_\eta)<1.
\]

Equivalently, construct the block matrix whose \((b,a)\)-block is

\[
p_{ab}\,
\mathbb E[M_{ab,k}\otimes M_{ab,k}\mid a,b].
\]

The recursion is mean-square stable if and only if the spectral radius of this
block matrix is below one.

*Proof.*  Define the unnormalized mode-conditioned covariance

\[
P_k^a
=
\mathbb E[X_kX_k^\top\mathbf 1\{Z_k=a\}].
\]

The Markov and conditional-independence assumptions give
\(P_{k+1}=\mathfrak L_\eta(P_k)\).  Vectorizing each block gives the stated
block matrix.  Sufficiency follows from its spectral radius being below one.
For necessity, \(\mathfrak L_\eta\) preserves the product cone of \(m\)
positive semidefinite matrices.  The cone-preserving Perron--Frobenius theorem
provides a nonzero product-cone eigenvector at
\(\rho(\mathfrak L_\eta)\); an initial distribution with these unnormalized
mode-conditioned covariances cannot converge in mean square when the radius
is at least one. \(\square\)

Theorem 2 is exact for finite joint Markov modes.  Its direct dimension can
grow exponentially with agent count, so it is a proof benchmark rather than
the proposed controller.

## 7. General unthinned mixing theorem: remaining obligations

For heterogeneous delays, the homogeneous recursion is

\[
e_{k+1}
=
e_k-\frac{\eta}{q}\sum_{i=1}^{q}
H_{i,k}e_{k-\tau_{i,k}}.
\]

The target Lyapunov--Krasovskii functional is

\[
\mathcal V_k
=
\|e_k\|_P^2
+\sum_{\ell=1}^{D}c_\ell
\|e_{k+1-\ell}-e_{k-\ell}\|^2.
\]

A valid main theorem must establish, rather than assume, a recursion of the
form

\[
\mathbb E[\mathcal V_{k+1}]
\le
\left[
1-c_\mu\mu\eta
+c_KK_{\rm mix}(q)\eta^2
+c_DL_H^2\Psi(\tau)\eta^2
\right]\mathbb E[\mathcal V_k]
+c_\zeta\eta^2\Omega_{\rm add}(q).
\]

The remaining obligations are:

1. **Markov dependence.**  Conditional on the iterate history,
   \(\mathbb E[H_{i,k}]\) need not equal \(A\).  A Poisson-equation or
   block-mixing argument must control this bias without declaring the sample
   Jacobian independent of \(e_k\).
2. **Temporal multiplicative noise.**  Replace the same-time \(K_q\) by an
   explicit \(K_{\rm mix}(q)\), or prove a mixing inflation
   \(K_{\rm mix}(q)\le\chi_{\rm mix}K_q\).  A Green--Kubo covariance for
   additive noise alone is insufficient.
3. **Heterogeneous staleness.**  Use
   \(e_k-e_{k-\tau_i}=\sum_{r=k-\tau_i}^{k-1}(e_{r+1}-e_r)\) inside the
   Lyapunov functional.  Bounding every delay by \(D\) is acceptable for the
   base theorem; an average-delay corollary must retain the registered delay
   weights.
4. **Cross terms.**  Young inequalities must be parameterized before choosing
   \(c_\ell\).  Choosing them after the desired step-size formula would make
   the parallel-sum rule circular.
5. **Predictable adaptation.**  Any online choice of \(q_k\) or \(\eta_k\)
   must be measurable with respect to past probe data.  Samples used to select
   the current action cannot also be treated as fresh evaluation noise.

The theorem-quality sufficient condition should have additive inverse margins,
for example

\[
\eta^{-1}
\ge
c_D\eta_{\rm delay}^{-1}
+c_KK_{\rm mix}(q)/\mu
+c_{\rm mix}L_H^2\tau_{\rm mix}/\mu,
\]

with every constant derived from the preceding recursion.  EXP-007D validates
the parallel-sum geometry as a proof target, not its constants as a theorem.

### 7.1 A proved low-complexity route: predictable decorrelation gaps

The unrestricted unthinned recursion above remains open, but a closely related
algorithm admits a direct finite-time proof.  Before update \(k\), advance the
joint data chain by a predictable decorrelation gap \(b_k\), without changing
the parameter.  Let \(\mathcal F_k\) contain the parameter history, delay
profile, participation decision, and all data before the fresh block.  Assume
the conditional law \(\nu_k\) of the participating agents' joint sample obeys

\[
\|\nu_k-\pi_{q_k}\|_{\rm TV}\le\delta_k
\quad\text{almost surely},
\]

where \(\pi_q\) is the stationary joint law.  This property follows, for
example, from a uniform joint-chain mixing bound evaluated at \(b_k\).  It does
not assert that the new sample is independent of the iterate.

For a fixed \(q\), define

\[
\bar H_k=\frac1q\sum_{i=1}^q H_{i,k},\qquad
A=\mathbb E_{\pi_q}\bar H_k,\qquad
K_q=\lambda_{\max}
\left(\mathbb E_{\pi_q}[\bar H_k^\top\bar H_k]\right).
\]

Assume

\[
\|H_{i,k}\|_2\le L,\qquad
\lambda_{\min}\!\left(\frac{A+A^\top}{2}\right)\ge\mu>0.
\]

Total-variation duality gives the conditional bounds

\[
\begin{aligned}
\mathbb E[\langle e,\bar H_ke\rangle\mid\mathcal F_k]
&\ge \mu_\delta\|e\|^2,\\
\mathbb E[\|\bar H_ke\|^2\mid\mathcal F_k]
&\le K_\delta\|e\|^2,
\end{aligned}
\]

where

\[
\mu_\delta=\mu-2L\delta,\qquad
K_\delta=K_q+2L^2\delta.
\]

The first quantity is positive whenever \(\delta<\mu/(2L)\).

**Theorem 3 (decorrelated delayed mean-square contraction).**  Fix \(q\),
\(\eta\), and delays \(0\le\tau_i\le D\).  Let

\[
\tau_{\rm rms}
=\left(\frac1q\sum_{i=1}^q\tau_i^2\right)^{1/2}.
\]

Suppose the conditional total-variation bound above holds with
\(\delta_k\le\delta<\mu/(2L)\).  For the homogeneous recursion

\[
e_{k+1}
=e_k-\frac{\eta}{q}\sum_{i=1}^qH_{i,k}e_{k-\tau_i},
\]

define

\[
c(\eta)
=1-2\eta\mu_\delta
+\eta^2\left(K_\delta+4L^2\tau_{\rm rms}\right)
+\eta^4L^4\tau_{\rm rms}^2.
\]

If

\[
\eta L\le1,\qquad
\eta\left(K_\delta+4L^2\tau_{\rm rms}\right)
+\eta^3L^4\tau_{\rm rms}^2
<2\mu_\delta,
\tag{T3}
\]

then \(c(\eta)<1\) and

\[
\max_{k-2D\le j\le k}\mathbb E\|e_j\|^2
\le
c(\eta)^{\lfloor k/(2D+1)\rfloor}
\max_{-2D\le j\le0}\mathbb E\|e_j\|^2.
\]

*Proof.*  Write

\[
a_k=\bar H_ke_k,\qquad
r_k=\frac1q\sum_{i=1}^qH_{i,k}
(e_{k-\tau_i}-e_k).
\]

For
\(R_k=\max_{k-2D\le j\le k}\mathbb E\|e_j\|^2\), the identity

\[
e_{k-\tau_i}-e_k
=\eta\sum_{s=k-\tau_i}^{k-1}
\frac1q\sum_{j=1}^qH_{j,s}e_{s-\tau_j}
\]

and two applications of Jensen's inequality imply

\[
\mathbb E\|r_k\|^2
\le
\eta^2L^4\tau_{\rm rms}^2R_k.
\tag{20}
\]

The conditional drift and second-moment bounds give

\[
\mathbb E\|e_k-\eta a_k\|^2
\le
(1-2\eta\mu_\delta+\eta^2K_\delta)
\mathbb E\|e_k\|^2.
\tag{21}
\]

Moreover, \(\eta L\le1\) gives
\(\|e_k-\eta a_k\|\le2\|e_k\|\) pathwise.  Combining (20), (21), and
Cauchy--Schwarz,

\[
\begin{aligned}
\mathbb E\|e_{k+1}\|^2
&=
\mathbb E\|e_k-\eta a_k-\eta r_k\|^2\\
&\le
\left[
1-2\eta\mu_\delta
+\eta^2(K_\delta+4L^2\tau_{\rm rms})
+\eta^4L^4\tau_{\rm rms}^2
\right]R_k\\
&=c(\eta)R_k.
\end{aligned}
\]

Condition (T3) makes \(c(\eta)<1\).  The sliding envelope cannot increase,
and after \(2D+1\) updates every element of the old window has been replaced
by an element bounded by \(c(\eta)R_k\).  Iteration proves the claim.
\(\square\)

The largest safe step in (T3) is obtained by solving a monotone scalar cubic
and clipping at \(1/L\).  It requires neither a preconditioner nor a matrix
inverse.  Agent count enters through \(K_q\), and the complete delay profile
enters through \(\tau_{\rm rms}\), rather than only through \(D\).

The preceding coefficient is convenient but not tight because it bounds the
cross term after expanding the square.  A direct \(L_2\) triangle inequality
gives a stronger result.  Define

\[
c_0(\eta)=1-2\eta\mu_\delta+\eta^2K_\delta,\qquad
d_\tau(\eta)=\eta^2L^2\tau_{\rm rms}.
\]

The same estimates (20)--(21) imply

\[
\begin{aligned}
\{\mathbb E\|e_{k+1}\|^2\}^{1/2}
&\le
\{\mathbb E\|e_k-\eta a_k\|^2\}^{1/2}
+\eta\{\mathbb E\|r_k\|^2\}^{1/2}\\
&\le
\left[
\sqrt{c_0(\eta)}+d_\tau(\eta)
\right]\sqrt{R_k}.
\end{aligned}
\]

Therefore the sharper sufficient condition is

\[
\boxed{
\sqrt{1-2\eta\mu_\delta+\eta^2K_\delta}
+\eta^2L^2\tau_{\rm rms}<1
}
\tag{T3-sharp}
\]

with contraction coefficient

\[
c_{\rm sharp}(\eta)
=
\left[
\sqrt{1-2\eta\mu_\delta+\eta^2K_\delta}
+\eta^2L^2\tau_{\rm rms}
\right]^2.
\]

For \(\tau_{\rm rms}=0\), (T3-sharp) recovers the fresh-sample sufficient
condition \(\eta<2\mu_\delta/K_\delta\) exactly.  Its boundary and its
rate-optimal interior step are both one-dimensional scalar solves.  The
rate-optimal step minimizes \(c_{\rm sharp}(\eta)\); selecting the largest
stable root would give vanishing theorem slack and is not the recommended
controller.

If the recursion additionally contains a conditionally centered innovation
\(\xi_k\) satisfying
\(\mathbb E[\|\xi_k\|^2\mid\mathcal F_k]\le\Omega_q\) and orthogonal to the
homogeneous update conditional on \(\mathcal F_k\), the same proof gives the
block recursion

\[
R_{k+2D+1}\le c(\eta)R_k+\eta^2\Omega_q
\]

and hence a residual of at most
\(\eta^2\Omega_q/[1-c(\eta)]\).

This corollary is not automatic for affine TD driven by finite-gap Markov
data.  At the projected fixed point, the one-step TD innovation can have a
nonzero conditional mean and can be conditionally correlated with the random
Jacobian.  The following result removes both requirements for the
predictably decorrelated algorithm.

**Theorem 4 (affine decorrelated delayed finite-time bound).**  In addition to
the setting of Theorem 3, consider

\[
e_{k+1}
=e_k-\frac{\eta}{q}\sum_{i=1}^q
\left(H_{i,k}e_{k-\tau_i}-\xi_{i,k}\right).
\tag{22}
\]

Assume \(\|\xi_{i,k}\|\le G\), and define the stationary aggregate innovation
and its second moment by

\[
\bar\xi_k=\frac1q\sum_{i=1}^q\xi_{i,k},\qquad
\mathbb E_{\pi_q}\bar\xi_k=0,\qquad
\Omega_q=\mathbb E_{\pi_q}\|\bar\xi_k\|^2.
\]

No conditional centering or orthogonality between \(\bar H_k\) and
\(\bar\xi_k\) is imposed.  Let

\[
\begin{aligned}
a_\delta(\eta)
&=1-\eta\mu_\delta
  +\eta^2(2K_q+4L^2\delta),\\
\beta_\delta(\eta)
&=\frac{4\eta G^2\delta^2}{\mu_\delta}
  +\eta^2(2\Omega_q+4G^2\delta),\\
h_\tau(\eta)
&=2\eta^4L^4\tau_{\rm rms}^2,\\
g_\tau(\eta)
&=2\eta^4L^2G^2\tau_{\rm rms}^2.
\end{aligned}
\tag{23}
\]

For \(\tau_{\rm rms}>0\), suppose \(a_\delta(\eta)>0\), set

\[
\lambda_\star(\eta)
=\sqrt{\frac{h_\tau(\eta)}{a_\delta(\eta)}},
\]

and define

\[
\begin{aligned}
c_{\rm aff}(\eta)
&=(1+\lambda_\star)a_\delta(\eta)
 +(1+\lambda_\star^{-1})h_\tau(\eta)\\
&=\left[
\sqrt{a_\delta(\eta)}
+\sqrt{2}\eta^2L^2\tau_{\rm rms}
\right]^2,\\
d_{\rm aff}(\eta)
&=(1+\lambda_\star)\beta_\delta(\eta)
 +(1+\lambda_\star^{-1})g_\tau(\eta).
\end{aligned}
\tag{24}
\]

When \(\tau_{\rm rms}=0\), use the continuous-limit definitions
\(c_{\rm aff}=a_\delta\) and \(d_{\rm aff}=\beta_\delta\).
If \(c_{\rm aff}(\eta)<1\), then for

\[
R_k=\max_{k-2D\le j\le k}\mathbb E\|e_j\|^2,\qquad
m=2D+1,
\]

the block envelope satisfies

\[
R_{k+m}
\le
\frac{d_{\rm aff}}{1-c_{\rm aff}}
+c_{\rm aff}
\left(
R_k-\frac{d_{\rm aff}}{1-c_{\rm aff}}
\right)_+.
\tag{25}
\]

Consequently, for every integer \(n\ge0\),

\[
\boxed{
R_{nm}
\le
c_{\rm aff}^nR_0
+\frac{d_{\rm aff}}{1-c_{\rm aff}}.
}
\tag{26}
\]

*Proof.*  Write

\[
v_k=e_k-\eta(\bar H_ke_k-\bar\xi_k),\qquad
r_k=\frac1q\sum_{i=1}^qH_{i,k}
(e_{k-\tau_i}-e_k).
\]

Total-variation duality and
\(\mathbb E_{\pi_q}\bar\xi_k=0\) give

\[
\left\|
\mathbb E[\bar\xi_k\mid\mathcal F_k]
\right\|
\le2G\delta.
\tag{27}
\]

Moreover,

\[
\mathbb E_{\pi_q}
\|\bar H_ke-\bar\xi_k\|^2
\le2K_q\|e\|^2+2\Omega_q,
\]

and the integrand is bounded by \((L\|e\|+G)^2\).  Hence

\[
\mathbb E[
\|\bar H_ke-\bar\xi_k\|^2
\mid\mathcal F_k]
\le
(2K_q+4L^2\delta)\|e\|^2
+2\Omega_q+4G^2\delta.
\tag{28}
\]

Using (27), Young's inequality with parameter \(\mu_\delta\), and the drift
bound preceding Theorem 3 yields

\[
\mathbb E\|v_k\|^2
\le
a_\delta(\eta)R_k+\beta_\delta(\eta).
\tag{29}
\]

The affine telescoping identity is

\[
e_{k-\tau_i}-e_k
=\eta\sum_{s=k-\tau_i}^{k-1}
\left[
\frac1q\sum_{j=1}^qH_{j,s}e_{s-\tau_j}
-\bar\xi_s
\right].
\]

Unlike (20), this identity retains the innovation.  Jensen's inequality,
\(\|H_{i,k}\|\le L\), and \(\|\bar\xi_s\|\le G\) imply

\[
\mathbb E\|r_k\|^2
\le
2\eta^2L^4\tau_{\rm rms}^2R_k
+2\eta^2L^2G^2\tau_{\rm rms}^2.
\tag{30}
\]

For any \(\lambda>0\),

\[
\|v_k-\eta r_k\|^2
\le
(1+\lambda)\|v_k\|^2
+(1+\lambda^{-1})\eta^2\|r_k\|^2.
\]

Combining (29)--(30) gives the one-step inequality

\[
\mathbb E\|e_{k+1}\|^2
\le
c_\lambda R_k+d_\lambda,
\]

where

\[
c_\lambda=(1+\lambda)a_\delta
+(1+\lambda^{-1})h_\tau,\qquad
d_\lambda=(1+\lambda)\beta_\delta
+(1+\lambda^{-1})g_\tau.
\]

The choice \(\lambda_\star=\sqrt{h_\tau/a_\delta}\) minimizes
\(c_\lambda\) and gives (24).  Put
\(R_\star=d_{\rm aff}/(1-c_{\rm aff})\).  If \(R_k\ge R_\star\), every new
element entering the sliding envelope is at most
\(c_{\rm aff}R_k+d_{\rm aff}\le R_k\).  After \(m=2D+1\) steps, all elements
of the old envelope have been replaced, proving
\(R_{k+m}\le c_{\rm aff}R_k+d_{\rm aff}\).  If \(R_k<R_\star\), the same
one-step inequality keeps the enlarged envelope
\(\max\{R_k,R_\star\}\) below \(R_\star\).  These two cases give (25), and
iteration gives (26). \(\square\)

For \(\delta=0\) and zero delay, (26) has

\[
c_{\rm aff}=1-\eta\mu+2\eta^2K_q,\qquad
\frac{d_{\rm aff}}{1-c_{\rm aff}}
=
\frac{2\eta\Omega_q}{\mu-2\eta K_q}.
\]

Thus the stochastic residual decreases linearly with \(\eta\), while a
finite decorrelation error contributes an additional
\(O(G^2\delta^2/\mu_\delta^2)\) mixing-bias floor.  Cross-agent dependence
enters both \(K_q\) and \(\Omega_q\); increasing \(q\) cannot remove their
off-diagonal components.

**Corollary 3.1 (geometric joint mixing).**  If the participating joint chain
satisfies

\[
\sup_z\|P_q^b(z,\cdot)-\pi_q\|_{\rm TV}
\le C_qr_q^b,\qquad 0<r_q<1,
\]

then Theorem 3 applies with
\(\delta=C_qr_q^b\).  Choosing

\[
b\ge
\frac{\log(C_q/\delta_{\rm target})}{-\log r_q},
\qquad
0<\delta_{\rm target}<\frac{\mu}{2L},
\]

is sufficient.  A predictable upper confidence bound on \(C_qr_q^b\) may be
substituted on its simultaneous confidence event.  This separates three
low-dimensional controls: participation \(q\), decorrelation gap \(b\), and
step size \(\eta\).

Theorem 3 closes a rigorous delayed-Markov route for the decorrelated
algorithm.  It does not close the more aggressive unthinned recursion in
Section 7; that distinction must remain explicit in the paper.

### 7.2 Correlation-limited minimax lower bound

The phrase *beyond linear speedup* requires an impossibility result, not only
an upper bound whose constants saturate.  The following Gaussian location
experiment is a zero-delay, one-step-mixing, sub-Gaussian Markov-learning
subclass.  Hence its lower bound applies to every claimed broader class that
includes such additive sub-Gaussian observations.  The bounded-innovation
assumption used for the affine upper bound is a separate sufficient condition.

At round \(t\), the learner requests \(q_t\in\{1,\ldots,N\}\) observations

\[
Y_{i,t}=\theta+C_t+\varepsilon_{i,t},\qquad i=1,\ldots,q_t,
\tag{27}
\]

where \(C_t\stackrel{\rm iid}{\sim}{\cal N}(0,\sigma_c^2)\),
\(\varepsilon_{i,t}\stackrel{\rm iid}{\sim}{\cal N}(0,\sigma_e^2)\), and all
innovations are independent across rounds.  The common innovation \(C_t\)
models state or environment correlation and the private innovations model
agent-specific sampling noise.  Put

\[
\rho=\frac{\sigma_c^2}{\sigma_c^2+\sigma_e^2}.
\]

**Theorem 5 (correlation-limited speedup and adaptive-budget lower bound).**
For a fixed participation level \(q\), the minimax squared-error risk after
\(T\) rounds in (27) is

\[
\inf_{\widehat\theta}\sup_{\theta\in\mathbb R}
\mathbb E_\theta(\widehat\theta-\theta)^2
=
\frac1T\left(\sigma_c^2+\frac{\sigma_e^2}{q}\right).
\tag{28}
\]

Relative to a single agent, the exact speedup is

\[
S(q,\rho)
=
\frac{q}{1+(q-1)\rho}
\le \min\{q,\rho^{-1}\},
\tag{29}
\]

where the second term is omitted at \(\rho=0\).  Consequently, for every
\(\rho>0\), speedup is bounded even as the number of available agents tends to
infinity.

More generally, let \(q_t\) be predictable from observations strictly before
round \(t\), let a round with \(q_t\) agents cost \(h+q_t\), and impose the
pathwise budget

\[
\sum_t(h+q_t)\le B.
\]

Then every adaptive participation rule and estimator obeys

\[
\sup_{\theta\in\mathbb R}
\mathbb E_\theta(\widehat\theta-\theta)^2
\ge
\frac{1}{
B\displaystyle\max_{1\le q\le N}
\frac{1}{(h+q)(\sigma_c^2+\sigma_e^2/q)}
}
=
\frac1B\min_{1\le q\le N}
(h+q)\left(\sigma_c^2+\frac{\sigma_e^2}{q}\right).
\tag{30}
\]

For continuous \(q\in[1,N]\), the minimizer is

\[
q_{\rm cont}^\star
=
\left[
\sqrt{\frac{h\sigma_e^2}{\sigma_c^2}}
\right]_{[1,N]}
=
\left[
\sqrt{\frac{h(1-\rho)}{\rho}}
\right]_{[1,N]},
\tag{31}
\]

with \(q_{\rm cont}^\star=N\) when \(\rho=0\).  The integer optimum is one of
the two feasible integers adjacent to (31).

*Proof.*  Conditional on \(q\), a round has covariance

\[
\Sigma_q=\sigma_e^2I_q+\sigma_c^2\mathbf 1\mathbf 1^\top.
\]

Sherman--Morrison gives the Fisher information

\[
I(q)=\mathbf 1^\top\Sigma_q^{-1}\mathbf 1
=\frac{q}{\sigma_e^2+q\sigma_c^2}
=\frac{1}{\sigma_c^2+\sigma_e^2/q}.
\tag{32}
\]

For fixed \(q\), generalized least squares over \(T\) independent rounds is
unbiased with constant risk \(1/[TI(q)]\).  Conversely, under the prior
\(\theta\sim{\cal N}(0,\tau^2)\), posterior variance is
\((\tau^{-2}+TI(q))^{-1}\).  Bayes risk lower-bounds minimax risk; sending
\(\tau\) to infinity proves (28).  Dividing the \(q=1\) risk by (28) proves
(29).

For a predictable adaptive rule, its decision kernel contributes no
\(\theta\)-dependent likelihood term conditional on the observed history.
Conditional on the realized decisions and observations, the Gaussian
likelihood remains quadratic in \(\theta\), with total precision
\(\sum_t I(q_t)\).  Under the same Gaussian prior, its posterior variance is

\[
\left(\tau^{-2}+\sum_t I(q_t)\right)^{-1}.
\]

The pathwise budget yields

\[
\sum_t I(q_t)
\le
B\max_{1\le q\le N}\frac{I(q)}{h+q}.
\]

Therefore the Bayes risk is at least
\(\{\tau^{-2}+B\max_q I(q)/(h+q)\}^{-1}\).  Taking
\(\tau\to\infty\) proves (30).  Finally,

\[
(h+q)\left(\sigma_c^2+\frac{\sigma_e^2}{q}\right)
=h\sigma_c^2+\sigma_e^2+q\sigma_c^2+\frac{h\sigma_e^2}{q}
\]

is convex in \(q>0\); its derivative vanishes at (31). \(\square\)

The lower bound is deliberately independent of the affine upper-bound
certificate.  The two results meet at the same structural conclusion:
off-diagonal correlation imposes a non-vanishing noise component, and the
optimal participation level balances that component against per-round
overhead.  Mixing gaps and communication latency may be included in \(h\);
they can only strengthen the impossibility statement.

### 7.3 Predictable dual-anytime certification

The registered pair-sharing model permits a rigorous online controller without
treating either mixing or cross-agent correlation as known.  Let
\(\{X_j^{(p)}\}\) be observed stay indicators with conditional mean \(p\), and
let \(\{X_j^{(\rho)}\}\) be observed pair-sharing indicators with conditional
mean \(\rho\).  Whether the next observation from either stream is acquired
must be predictable.  This allows the number of observations at a decision
time to depend arbitrarily on past data and actions.

For either Bernoulli stream, write \(S_n=\sum_{j=1}^nX_j\) and define the
beta-binomial mixture likelihood ratio

\[
M_n(u)
=
\frac{
B(S_n+\tfrac12,n-S_n+\tfrac12)
}{
B(\tfrac12,\tfrac12)
u^{S_n}(1-u)^{n-S_n}
}.
\tag{33}
\]

**Theorem 6 (dual-anytime safe adaptation).**  Assign
\(\alpha_p+\alpha_\rho=\alpha\), and let \(u_{p,n}^+\) and
\(u_{\rho,n}^+\) be the upper endpoints of

\[
{\cal C}_{s,n}
=
\{u\in(0,1):M_{s,n}(u)<1/\alpha_s\},
\qquad s\in\{p,\rho\}.
\tag{34}
\]

Round both endpoints upward on any deterministic grids.  Suppose:

1. \(p\in[1/2,1)\), and the joint mixing certificate
   \(\delta(p,b)\) is non-decreasing in \(p\);
2. the aggregate curvature \(K_q(\rho)\) and aggregate additive-noise proxy
   \(\Omega_q(\rho)\) are non-decreasing in \(\rho\);
3. every block action is measurable with respect to data available before the
   block and satisfies Theorem 3 or Theorem 4 after substituting
   \(u_p^+\) and \(u_\rho^+\).

Then, with probability at least \(1-\alpha\), every selected action at every
decision time satisfies its corresponding true-parameter stability
condition.  The statement remains valid when the observation counts are
adaptive stopping times.

*Proof.*  Under parameter \(u\), (33) is the likelihood under a
\({\rm Beta}(1/2,1/2)\) mixture alternative divided by the likelihood under
\({\rm Bernoulli}(u)\).  It is therefore a nonnegative martingale with initial
value one.  Ville's inequality gives

\[
\mathbb P_u\{\exists n:M_n(u)\ge1/\alpha_s\}\le\alpha_s.
\]

This is indexed by intrinsic sample count, so predictably skipping
observations or evaluating at an adaptive stopping time does not change the
guarantee.  Applying the result to both streams and taking a union bound gives
simultaneous coverage \(p\le u_p^+\) and \(\rho\le u_\rho^+\) at all decisions
with probability at least \(1-\alpha\).  Upward rounding preserves coverage.
On this event, monotonicity gives

\[
\delta(p,b)\le\delta(u_p^+,b),\qquad
K_q(\rho)\le K_q(u_\rho^+),\qquad
\Omega_q(\rho)\le\Omega_q(u_\rho^+).
\]

Thus each action certified with the upper endpoints satisfies the same
inequalities at the true parameters.  Predictability prevents current-block
data from leaking into its action. \(\square\)

For the symmetric two-state chain,
\(\delta(p,b)=\tfrac12(2p-1)^b\).  In the pair-sharing model,

\[
K_q(\rho)
=\lambda_{\max}\left[
\frac{Q_{\rm diag}}q
+\frac{q-1}{q}
\{Q_{\rm mean}+\rho(Q_{\rm diag}-Q_{\rm mean})\}
\right],
\]

and \(Q_{\rm diag}-Q_{\rm mean}\succeq0\) is a conditional covariance, proving
the required Loewner monotonicity.  Updating the two success/trial counters is
\(O(1)\); the rank-one TD curvature aggregation and finite candidate search
remain \(O(qd)\) time and \(O(d)\) memory.  The theorem covers observable
sharing or collision indicators.

### 7.4 Latent sharing from observable collisions

Direct sharing metadata is unnecessary when shared agents reuse the same
observable Markov sample.  Let \(C,E_1,E_2\) be independent draws from a
stationary distribution \(\pi\), and let
\(B_1,B_2\stackrel{\rm iid}{\sim}{\rm Bernoulli}(\sqrt\rho)\).  The learner
observes

\[
Y_i=B_iC+(1-B_i)E_i,\qquad i\in\{1,2\},
\]

where the expression denotes sample selection rather than numerical
interpolation.  The masks \(B_i\) are hidden.  Define

\[
c_\pi=\mathbb P(E_1=E_2)=\sum_y\pi(y)^2,\qquad
X=\mathbf 1\{Y_1=Y_2\}.
\]

**Proposition 4 (latent-collision identity).**  The stationary collision
probability is

\[
\vartheta
=\mathbb E_\pi X
=c_\pi+(1-c_\pi)\rho.
\tag{35}
\]

*Proof.*  With probability \(\rho\), both agents select \(C\) and collide
surely.  On the complementary event, every remaining pair of selected sources
is independent with marginal \(\pi\), and collides with probability
\(c_\pi\). \(\square\)

The identity also holds for a bounded similarity kernel after replacing one
and \(c_\pi\) by its shared-source and independent-source expectations.

Suppose collision probe \(j\) is taken after a predictable gap \(b_j\), and
the conditional law of its latent sources is within total variation
\(\delta_j\) of their stationary joint law.  Put
\(\mu_j=\mathbb E[X_j\mid{\cal F}_{j-1}]\).  Since
\(X_j\in[0,1]\),

\[
|\mu_j-\vartheta|\le\delta_j.
\tag{36}
\]

For \(\alpha_\rho\in(0,1)\), define

\[
r_n(\alpha_\rho)
=
\sqrt{
\frac{
\log\{\pi^2n^2/(6\alpha_\rho)\}
}{2n}
}
\]

and

\[
\vartheta_n^+
=
\min\left\{
1,\,
\frac1n\sum_{j=1}^nX_j
+r_n(\alpha_\rho)
+\frac1n\sum_{j=1}^n\delta_j
\right\}.
\tag{37}
\]

**Theorem 7 (anytime latent-correlation certificate).**  If
\(c_\pi^-\le c_\pi<1\), then with probability at least
\(1-\alpha_\rho\), simultaneously for every adaptive collision-probe count
\(n\ge1\),

\[
\rho
\le
\rho_n^+
:=
\left[
\frac{\vartheta_n^+-c_\pi^-}{1-c_\pi^-}
\right]_{[0,1]}.
\tag{38}
\]

If the persistence confidence sequence in Theorem 6 is assigned failure
probability \(\alpha_p\), and every \(\delta_j\) is computed from its
pre-probe upper endpoint, then (38) and the mixing sequence hold jointly with
probability at least \(1-\alpha_p-\alpha_\rho\).

*Proof.*  For fixed \(n\), the martingale Hoeffding inequality gives

\[
\mathbb P\left\{
\frac1n\sum_{j=1}^n(\mu_j-X_j)
>
\sqrt{\frac{\log(1/\alpha_n)}{2n}}
\right\}
\le\alpha_n.
\]

Set \(\alpha_n=6\alpha_\rho/(\pi^2n^2)\) and take a union bound over all
\(n\).  Combining the resulting event with
\(\vartheta\le\mu_j+\delta_j\) proves
\(\vartheta\le\vartheta_n^+\) simultaneously.  From (35),
\((\vartheta-c)/(1-c)\) is non-increasing in \(c\) for
\(\vartheta\le1\); hence substituting \(c_\pi^-\) yields (38).  The final
claim follows from Theorem 6 and a second union bound. \(\square\)

For three independent symmetric two-state latent chains,

\[
\delta_j
\le
\min\left\{
1,\frac32(2p_j^+-1)^{b_j}
\right\}.
\]

The certificate stores only collision, probe-count, and cumulative-bias
scalars.  Computing a collision is \(O(1)\) once two agent samples are
available.  No hidden mask, covariance matrix, or preconditioner is used.
When \(c_\pi\) is unknown, setting \(c_\pi^-=0\) remains valid but more
conservative; estimating a useful lower bound for general state spaces is a
separate statistical problem.

## 8. SDDE representation

The matching stochastic delay differential equation must retain the
state-dependent quadratic variation generated by the random Jacobian:

\[
dE(t)
=
-\frac1q\sum_{i=1}^{q}A_iE(t-\tau_i(t))\,dt
+\sqrt{\eta}\,
\Sigma_q^{1/2}
\big(E(t),E(t-\tau_1),\ldots,E(t-\tau_q)\big)\,dW(t).
\]

Near the equilibrium, the multiplicative component satisfies a quadratic
bound of the form

\[
\operatorname{tr}
\left[\Sigma_{{\rm mult},q}(e)
\Sigma_{{\rm mult},q}(e)^\top\right]
\le
C K_{\rm mix}(q)\|e\|^2.
\]

An additive constant diffusion cannot reproduce the correlation-dependent
stability boundary.  The SDDE provides the Lyapunov--Krasovskii interpretation;
the discrete-time theorem remains the paper's convergence guarantee.

## 9. Low-complexity estimator and controller

The exact finite-MRP audit forms \(Q_q\) only to verify the mechanism.  The
online method need not store a \(d\times d\) matrix.

For a unit probe direction \(v_m\), compute

\[
y_m=\bar H_{q,m}v_m,\qquad
z_m=\bar H_{q,m}^\top y_m.
\]

A damped stochastic power update tracks the dominant direction,

\[
v_{m+1}
=
\frac{(1-\beta_m)v_m+\beta_m z_m}
{\|(1-\beta_m)v_m+\beta_m z_m\|},
\]

while an upper-confidence exponential average of \(\|y_m\|^2\) tracks
\(K_q\).  A separate predictable mixing certificate selects the smallest
decorrelation gap \(b_m\) for which
\(\widehat\delta_m^+(b_m)<\widehat\mu_m^-/(2\widehat L_m^+)\).
Because a TD Jacobian is rank one, the curvature probe requires
\(O(qd)\) arithmetic and \(O(d)\) memory, matching the order of aggregating the
agent updates.  The safe step is the positive scalar root of (T3), clipped at
\(1/\widehat L_m^+\).  The control decision uses only scalar lookup or
bisection.

Participation \(q_k\) is selected over a finite candidate set by substituting
this safe root, its decorrelation cost \(b_k\), and the correlation-limited
additive noise estimate into a finite-budget risk surrogate.  It is a bound
only under the conditional centering and orthogonality conditions above.  All
decisions for block \(m+1\) use probes completed by block \(m\).  No
actor--critic, inverse Hessian, covariance matrix, or preconditioner is
required.

## 10. Proof status

| Component | Status |
|---|---|
| Exchangeable aggregate-Jacobian identity | proved |
| Pair-sharing closed form and agent-count saturation | proved |
| Fresh no-delay mean-square contraction | proved |
| Additive-noise residual bound | proved under conditional centering and orthogonality |
| Affine finite-gap Markov-TD bound | proved and tested |
| Correlation-limited minimax lower bound | proved; exact Gaussian subclass |
| Dual-anytime mixing/correlation certificate | proved for observable pair sharing |
| Latent-collision correlation certificate | proved for hidden pair sharing |
| Heterogeneous-delay independent-time theorem | proved |
| Matrix-free exact lifted operator | verified to machine precision |
| Finite-state Markov jump theorem | proved |
| Decorrelated delayed Markov theorem | proved |
| Unthinned joint Markov mixing theorem | open |
| Predictable two-state mixing certificate | proved/tested |
| General finite-state anytime estimator | open |
| Unthinned affine Markov-TD finite-time bound | open |
| SDDE-to-discrete approximation error | open |

The next local-theory task is to validate the dual-anytime
\((q,b,\eta)\) controller and extend its observable-sharing certificate to a
latent correlation probe.  A Poisson-equation argument for unthinned data
remains a strictly stronger optional extension.
