# Conservative-certificate robustness

## Scope

This note closes the theory interface between the finite-horizon Lyapunov
controller and conservative upper certificates.  It applies to every fixed
catalogue action \(a=(q,b,\eta)\) satisfying the assumptions of the affine
Markov-learning theorem.  It does not assume that the upper certificates are
tight and does not use evaluation return.

Write \(\delta=C_qr_q^b\), \(\mu_\delta=\mu-2L\delta\), and

\[
a=1-\eta\mu_\delta+\eta^2(2K_q+4L^2\delta),
\qquad
\beta=\frac{4\eta G^2\delta^2}{\mu_\delta}
       +\eta^2(2\Omega_q+4G^2\delta).
\]

Let \(h\) and \(g\) denote the delay terms in the main theorem.  With the
Young parameter \(\lambda=\sqrt{h/a}\), the certified replacement-block
recursion is

\[
R_{m+1}\le cR_m+d,
\quad
c=(\sqrt a+\sqrt h)^2,
\quad
d=(1+\sqrt{h/a})\beta+(1+\sqrt{a/h})g,
\]

with the usual continuous interpretation when \(h=0\).

## Corollary: safety under conservative certificates

Suppose the predictable calibration interface returns

\[
K_q^+\ge K_q,
\qquad
\Omega_q^+\ge\Omega_q,
\qquad
r_q^+\ge r_q,
\qquad r_q^+<1.
\]

Let \(\delta^+=C_q(r_q^+)^b\), and construct \(a^+\), \(\beta^+\),
\(\lambda^+=\sqrt{h/a^+}\), \(c^+\), and \(d^+\) from the upper values.
If \(\mu-2L\delta^+>0\) and \(c^+<1\), then the true replacement-block
recursion, evaluated with the same \(\lambda^+\), obeys

\[
R_{m+1}\le c^+R_m+d^+.
\]

Consequently, the conservative finite-horizon score

\[
U_a^+=R_a^{+,\star}+(c_a^+)^n
       (R_0-R_a^{+,\star})_+,
\qquad
R_a^{+,\star}=\frac{d_a^+}{1-c_a^+},
\]

is a valid upper bound on the true risk.  Conservatism can remove an action
from the certified set, but it cannot make an uncertified action appear safe.

### Proof

The map \((K,\delta)\mapsto a\) is coordinatewise nondecreasing on the
certified domain, and \((\Omega,\delta)\mapsto\beta\) is coordinatewise
nondecreasing whenever \(\mu-2L\delta>0\).  Therefore \(a^+\ge a\) and
\(\beta^+\ge\beta\).  The one-step delayed Young inequality is valid for any
positive predictable \(\lambda\).  Substitute \(\lambda^+\) in the true
inequality.  Replacing \(a\) and \(\beta\) by their upper values increases
both its contraction and forcing coefficients, giving \(c^+\) and \(d^+\).
Iteration proves the claim.

The shared \(\lambda^+\) is important.  Comparing two independently optimized
Young parameters term by term is not a valid monotonicity proof.

## Quantitative degradation

Let \(\Delta\delta=\delta^+-\delta\),
\(\Delta K=K_q^+-K_q\), and
\(\Delta\Omega=\Omega_q^+-\Omega_q\).  Then

\[
\Delta a
=2\eta^2\Delta K
 +(2\eta L+4\eta^2L^2)\Delta\delta,
\]

and the exact forcing perturbation is

\[
\Delta\beta
=2\eta^2\Delta\Omega+4\eta^2G^2\Delta\delta
 +4\eta G^2\left[
 \frac{(\delta^+)^2}{\mu-2L\delta^+}
 -\frac{\delta^2}{\mu-2L\delta}
 \right].
\]

The optimized contraction perturbation is

\[
\Delta c=(\sqrt{a^+}+\sqrt h)^2-(\sqrt a+\sqrt h)^2.
\]

For \(h>0\), a valid forcing bound is

\[
\Delta d\le
(1+\sqrt{h/a^+})\Delta\beta
+\sqrt h\left(\frac1{\sqrt a}-\frac1{\sqrt{a^+}}\right)\beta
+\frac{\sqrt{a^+}-\sqrt a}{\sqrt h}g.
\]

When \(h=0\), \(\Delta d=\Delta\beta\).  Finally, if all compared actions
satisfy \(c,c^+\le\bar c<1\) and \(d,d^+\le\bar d\), the finite-horizon score
obeys

\[
|U_a^+-U_a|
\le
\left(nR_0+\frac{\bar d}{(1-\bar c)^2}\right)\Delta c
+\frac{\Delta d}{1-\bar c}
=:\varepsilon_a.
\]

Thus, if \(\widehat a\in\arg\min_a U_a^+\) and
\(a^\star\in\arg\min_a U_a\),

\[
U_{\widehat a}
\le U_{a^\star}+2\max_a\varepsilon_a.
\]

This separates the two consequences of conservative calibration: safety is
preserved exactly, while selection quality degrades continuously with the
certificate slack.

