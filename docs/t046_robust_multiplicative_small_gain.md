# T-046 robust multiplicative small-gain theorem

## Decision

T-046 closes a rigorous bounded-multiplicative finite-horizon extension of the
additive phase theorem. It is a robust envelope, not the desired sharp
Poisson/mixing rate. The result is useful for a scoped theory/TSP manuscript,
but the general multiplicative-TD headline remains conditional because the
certificate can be conservative and need not preserve a narrow phase margin.

## Lifted recursion

Write delayed TD in lifted form

\[
x_{t+1}=Cx_t+u_t+E_tx_t,
\]

where \(C\) contains the mean delayed drift, \(u_t\) is the additive Markov
innovation, and

\[
E_t=-\eta J[A(Z_t)-\bar A]S_D.
\]

Let \(x_t^0\) solve the additive recursion with the same initial state and
inputs, and suppose \(\|E_t\|_{\rm op}\le\epsilon\) almost surely. Define the
finite-horizon deterministic impulse gain

\[
G_T=\sum_{k=0}^{T-1}\|C^k\|_{\rm op},
\qquad g_T=\epsilon G_T.
\]

## Theorem 1: robust pathwise small gain

If \(g_T<1\), then

\[
\max_{t\le T}\|x_t\|
\le\frac{1}{1-g_T}\max_{t\le T}\|x_t^0\|,
\]

and

\[
\max_{t\le T}\|x_t-x_t^0\|
\le\frac{g_T}{1-g_T}\max_{t\le T}\|x_t^0\|.
\]

### Proof

Variation of constants gives

\[
x_t-x_t^0=\sum_{s<t}C^{t-1-s}E_sx_s.
\]

Taking the maximum over the finite path yields
\(\|x-x^0\|_\infty\le g_T\|x\|_\infty\). Combining this with
\(\|x\|_\infty\le\|x^0\|_\infty+\|x-x^0\|_\infty\) proves both claims.

For TD, one may use
\(\epsilon\le\eta\sup_z\|A(z)-\bar A\|_{\rm op}\). Delay is retained in
\(C\) and hence in \(G_T\); the bound never declares the current Markov sample
independent of the iterate.

## Corollary 2: computable finite-risk envelope

Let \(R_T^0=\mathbb E\|Sx_T^0\|^2\) be the exact additive terminal risk from
T-037, and let

\[
U_T^0=\sum_{t=0}^{T}\mathbb E\|x_t^0\|^2.
\]

Because the maximum is bounded by the sum,

\[
\mathbb E\|S(x_T-x_T^0)\|^2
\le \left(\frac{g_T}{1-g_T}\right)^2 U_T^0=:\Delta_T.
\]

Cauchy--Schwarz gives the two-sided envelope

\[
\max\{0,R_T^0-2\sqrt{R_T^0\Delta_T}-\Delta_T\}
\le R_T
\le R_T^0+2\sqrt{R_T^0\Delta_T}+\Delta_T.
\]

Thus an additive phase ordering is certified for multiplicative TD whenever
the preferred action's upper endpoint is below the comparator's lower
endpoint. All inputs are prospective: T-037 risks, the lifted mean matrix,
and a public uniform sample-matrix bound.

## Relation to T-042 and scope

T-046 uses a worst-case bounded perturbation and therefore does not exploit
Markov cancellation or cross-agent averaging in the multiplicative term.
T-042's exact weighted Poisson decomposition remains the route to a sharper
mean-square constant. The robust theorem is nonetheless a valid general-TD
extension on the explicit class \(g_T<1\).

The headline may say “multiplicative finite-state Markov TD” only with this
small-gain condition and a reported nonvacuity certificate. Without such a
certificate, the paper must keep T-037 additive and T-036 exact
mode-enumerating results as its main finite-time claims. It may not call the
unrestricted multiplicative remainder solved.

## Verification

CPU tests verify the geometric scalar gain, pathwise inequality for delays 0,
1, and 3 under random bounded perturbations, refusal to produce a finite
factor when \(g_T\ge1\), exact recovery at zero multiplicative deviation, and
phase-envelope logic.
