# Finite-budget adaptation threshold sandwich

## Separated class

The positive statement is deliberately not uniform over degenerate
instances. Let \({\cal K}\) be a public compact class satisfying:

1. \(0<\theta_-\leq\theta_j\leq\theta_+<\infty\) and
   \(|\theta_1-\theta_0|\geq\Delta_{\min}>0\);
2. \(0\leq\lambda\leq1-\gamma\), \(\gamma>0\);
3. a finite, stability-screened \((q,b)\) catalogue, with positive message
   costs \(h+q\) and environment costs \(b\);
4. budgets lie on a public ray
   \((B_m(s),B_e(s))=(\lfloor s\beta_m\rfloor,
   \lfloor s\beta_e\rfloor)\), \(\beta_m,\beta_e>0\), and
   \(D\leq D_{\max}\);
5. whenever adaptation is claimed worthwhile, the all-agent baseline has
   oracle coefficient gap at least \(g_{\min}>0\); all catalogues have the
   stated uniform conditioning and finite-risk constants.

These constants are public class constants, not hidden instance- or
horizon-dependent quantities.

## Three thresholds

For target error \(0<\delta<1/2\), define
\(d_\delta=\operatorname{kl}(1-\delta,\delta)\).

**Necessary identification scale \(B_N\).** This is the least budget-ray
scale at which a predictable occupation measure can satisfy both directional
AC-7 information constraints

\[
 I_{0\to1}\geq d_\delta,\qquad I_{1\to0}\geq d_\delta,
\]

as well as message and environment accounting. If \(s<B_N\), binary data
processing makes \(\delta\)-correct identification impossible.

**Sufficient safe scale \(B_S\).** This is the least scale at which one can
afford a fixed catalogue probe block with Bhattacharyya exponent at least
\(\log(1/\delta)\), pay delay, and satisfy the theorem-derived strict fallback
inequality

\[
 C_{\rm probe}(s)+C_{\rm delay}(s)+
 \delta C_{\rm wrong}(s)<G_{\rm commit}(s),                 \tag{3}
\]

with low-instance safety deficit at most \(\epsilon\).

**Oracle scale \(B_{\rm oracle}\).** This is the first scale at which the
known-instance oracle action is feasible and its baseline improvement after
delay can amortize the class-minimal nonzero identification and delay cost.
It is an oracle benchmark, not an implementable unknown-instance rule.

## Why a uniform sandwich exists on the separated class

Uniform geometric mixing and finite catalogue compactness give positive
finite constants \(i_-,i_+,a_-,a_+\), depending only on \({\cal K}\), such
that the directional KL/Bhattacharyya information of a valid probe block is
linear in its length between these constants, while its two resource costs
are between \(a_-n\) and \(a_+n+D_{\max}\). Hence

\[
 c_Nd_\delta\leq B_N(\delta),\qquad
 B_{\rm id}^{\rm suff}(\delta)\leq c_S\log(1/\delta)+c_D.   \tag{4}
\]

For an action \(a=(q,b)\), let \(r=\lambda^b\). With \(N\) usable updates,
the exact scalar terminal-mean risk is

\[
 R_j(a,N)=\frac1{qN}+\theta_j\frac{N+2\sum_{k=1}^{N-1}(N-k)r^k}{N^2},
\]

so, uniformly on \({\cal K}\),

\[
 R_j(a,N)=\frac{\kappa_j(a)}N+O_{\cal K}(N^{-2}),\qquad
 \kappa_j(a)=\frac1q+\theta_j\frac{1+r}{1-r}.              \tag{5}
\]

The usable update rate on the budget ray is

\[
 \rho_a=\min\left\{\frac{\beta_m}{h+q},
                    \frac{\beta_e}{b}\right\},
\]

and delay changes \(N_a(s)\) by at most \(D_{\max}+O(1)\), not the covariance
law. Thus \(R_j(a;s)=K_j(a)/s+O_{\cal K}(s^{-2})\), where
\(K_j(a)=\kappa_j(a)/\rho_a\). A probe block of \(n\) updates plus delay costs
at most \(L_{\cal K}(n+D_{\max})/s^2\), while correct commitment gains at
least \(g_{\min}/s+O(s^{-2})\). If
\(\delta W_{\max}\leq g_{\min}/4\), then (3) holds once

\[
 s\geq \max\left\{B_{\rm oracle},
 \frac{4L_{\cal K}(n+D_{\max})}{g_{\min}},
 B_{\rm id}^{\rm suff}(\delta), B_\epsilon\right\}.         \tag{6}
\]

Combining (4)--(6), for \(0<\delta\leq\delta_0<1/2\) and a fixed admissible
safety target,

\[
 B_N\leq B_S\leq C_{\cal K}B_N+C'_{\cal K}
                 (B_{\rm oracle}+B_\epsilon+1),             \tag{7}
\]

For fixed admissible \(\epsilon\) and \(0<\delta\leq\delta_0<1/2\), the
additive feasibility scales are public class constants and
\(d_\delta\) is bounded away from zero, so they can be absorbed to give
\(B_S/B_N\leq\widetilde C_{\cal K}\). Alternatively, displaying them gives
the class-constant/logarithmic form (7). No constant depends on the unknown
instance or horizon.

## Interpretation

- Below \(B_N\), reliable regime identification is impossible; a minimax or
  theorem-derived fallback is therefore the justified behavior.
- Above \(B_S\), explore-then-commit is strictly worthwhile on high-gain
  instances and has certified low-instance safety deficit at most
  \(\epsilon\).
- The statement is a threshold comparison, not a proof that a Track-and-Stop
  controller matches every controlled-belief occupation optimum.

The distinction matters: the threshold is driven jointly by correlation
speedup, changing-\(q\) information and message cost, stride-induced
decorrelation and environment cost, delay-reduced usable horizon, wrong
post-identification optimization, and baseline safety. A pure identification
cost theorem does not imply (3), (5), or (6).
