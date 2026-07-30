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

## 7. General mixing main theorem: proof obligations

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
\(K_{\rm mix}(q)\).  Because a TD Jacobian is rank one, this probe requires
\(O(qd)\) arithmetic and \(O(d)\) memory, matching the order of aggregating the
agent updates.  The control decision is then the scalar lookup

\[
\eta_k
=
\left[
\widehat\eta_{\rm delay}^{-1}
+\widehat K_{\rm mix}^{+}(q_k)/(2\widehat\mu)
\right]^{-1}.
\]

Participation \(q_k\) is selected over a finite candidate set by substituting
this safe step and the correlation-limited additive noise estimate into the
finite-budget risk bound.  No actor--critic, inverse Hessian, covariance
matrix, or preconditioner is required.

## 10. Proof status

| Component | Status |
|---|---|
| Exchangeable aggregate-Jacobian identity | proved |
| Pair-sharing closed form and agent-count saturation | proved |
| Fresh no-delay mean-square contraction | proved |
| Additive-noise residual bound | proved |
| Heterogeneous-delay independent-time theorem | proved |
| Matrix-free exact lifted operator | verified to machine precision |
| Finite-state Markov jump theorem | proved |
| Tractable joint Markov mixing bound | open |
| Predictable online estimator guarantee | open |
| SDDE-to-discrete approximation error | open |

The next local-theory task is to compare the scalar joint rule with the exact
lifted boundary, then extend the lifted construction to a Markov jump operator.
The latter isolates the finite-state dependence exactly before a
Poisson-equation argument addresses general geometrically mixing data.
