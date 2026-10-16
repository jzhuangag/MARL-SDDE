## Material Passport

- Origin Skill: academic-research-suite / research-engineer
- Origin Mode: prove
- Origin Date: 2026-09-07
- Verification Status: VERIFIED
- Version Label: observable_signed_reference_theory_v1

# Observable affine reference and signed-drift error

Date: 2026-09-07

Status: exact-model corollary for PDSG-OBS-001.  It does not replace the
centralized-critic/JVP analysis required for nonlinear MARL.

## 1. Observation identity

For an affine packet born with conditional row `r_p` and mixed policy cache
`chi_p`, write

\[
G_p=r_p^\top(\chi_p-\theta^\star)+\xi_p,
\qquad
\mathbb E[\xi_p\mid\mathcal F_{r(p)-}]=0.
\tag{1}
\]

At receipt the learner knows `r_p`, `chi_p`, and `G_p`, so the same ordinary
training packet reveals

\[
Y_p=r_p^\top\chi_p-G_p=r_p^\top\theta^\star-\xi_p.
\tag{2}
\]

This is not a separate sensor.  It is the observable system-identification
content already present in the applied gradient packet.

The estimator used in PDSG-OBS-001 is

\[
\widehat\theta^star_{p+1}=\widehat\theta^star_p
-\gamma r_p
\frac{r_p^\top\widehat\theta^star_p-Y_p}{\|r_p\|^2},
\qquad 0<\gamma<2.
\tag{3}
\]

It starts from the current parameter vector.  Translating `theta`, `chi`, and
`theta*` by the same vector therefore translates the estimate and leaves every
prediction, score, and graph action unchanged.

## 2. Exact energy identity

Let `e_p=hat(theta)^star_p-theta^star` and
`P_p=r_pr_p^T/||r_p||^2`.  Conditional on the completed predictable row,

\[
e_{p+1}=(I-\gamma P_p)e_p
-\gamma\frac{r_p\xi_p}{\|r_p\|^2}.
\tag{4}
\]

Consequently,

\[
\mathbb E_p\|e_{p+1}\|^2
=\|e_p\|^2
-\gamma(2-\gamma)\frac{(r_p^\top e_p)^2}{\|r_p\|^2}
+\gamma^2\frac{\mathbb E_p\xi_p^2}{\|r_p\|^2}.
\tag{5}
\]

Equation (5) is exact; it does not assume independent coordinates or call
consecutive Markov observations iid.  Summation controls cumulative normalized
prediction error plus an explicit packet-noise floor.

## 3. Persistent excitation in the registered cyclic-owner model

One six-launch owner cycle contains one conditional row per agent.  Every row
has diagonal coefficient `m+c` and off-diagonal absolute sum `c`, where
`m=0.4` and `c` is the coupling.  The row matrix is therefore strictly
diagonally dominant by `m`.  Varah's inverse bound and
`||A||_2<=sqrt(n)||A||_infinity` give

\[
\sigma_{\min}(R)\ge \frac{m}{\sqrt n}.
\tag{6}
\]

Moreover each row norm is at most
`sqrt((m+c)^2+c^2)`.  Hence the normalized six-row Gram matrix obeys

\[
\sum_{s=p}^{p+5}P_s\succeq
\lambda_{\rm PE} I,
\qquad
\lambda_{\rm PE}=
\frac{m^2}{n\{(m+c)^2+c^2\}}>0.
\tag{7}
\]

The bound holds for every donor offset and both registered Markov modes; it is
not an average over favorable paths.

For a noiseless block of length `B` satisfying (7), the projection-energy
identity and

\[
\|P_se_p\|\le \|P_se_s\|+
\gamma\sum_{u=p}^{s-1}\|P_ue_u\|
\]

give the conservative block contraction

\[
\|e_{p+B}\|^2\le
\left[1-
\frac{\gamma(2-\gamma)\lambda_{\rm PE}}
{2(1+\gamma^2B^2)}\right]\|e_p\|^2.
\tag{8}
\]

Martingale packet noise adds its propagated quadratic term.  Thus the
estimator contracts toward an explicit stochastic neighborhood; the result
does not claim exact convergence under persistent noise.

## 4. Reference error becomes signed-score error

For one candidate graph action, let

\[
a=h^\top(\theta-\theta^\star),
\qquad
\mu=r^\top(\chi-\theta^\star),
\]

and let the paired launch score be

\[
D=-w a\mu+\frac{Lw^2}{2}(\mu^2+\sigma^2)+wd|\mu|.
\tag{9}
\]

If `||e||<=E`, `||h||<=H`, `||r||<=R`, `|a|<=A`, and `|mu|<=M`, replacing
`theta*` by the predictable estimate changes the score by at most

\[
\epsilon(E)=
w\{ARE+MHE+HRE^2\}
+\frac{Lw^2}{2}RE(2M+RE)
+wdRE.
\tag{10}
\]

This follows from Cauchy--Schwarz, the product identity, and
`||x|-|y||<=|x-y|`.  Taking uniform constants over the null-plus-one-edge
action set makes (10) a simultaneous score-error bound.  The selected action
then loses at most `2 epsilon(E)` relative to the true signed-score minimizer.

Substitution into the paired Lyapunov theorem yields

\[
\sum_{p<N}\kappa_p\mathbb E\|g_p^0\|^2
\le F(\theta^0)-F_\star+
\frac{Q_0^2}{2V}+
\sum_{p<N}\mathbb E[R_p+2\epsilon(\|e_p\|)]
+\frac{NB_Q}{V}.
\tag{11}
\]

Equations (5), (7), and (10) therefore close the exact-model chain from
completed packet data to online graph decisions and finite-time learning
error.  In a nonlinear critic the reference is no longer a finite vector
`theta*`; the paper must instead bound the local critic/JVP score error using
the separate sparse-alignment estimator interface.

## 5. Verified implementation correspondence

`online_reference_theory.py` implements (5), the lower bound in (7), the
coefficient in (8), and (10).  Unit tests verify the energy identity by exact
two-point noise quadrature, check (7) against numerical Gram eigenvalues for
all registered coupling/mode values, test translation equivariance, and
confirm that (10) dominates an explicit score perturbation.
