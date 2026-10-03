# Learning-excess-to-testing reduction for finite deployment

Date: 2026-09-05

Status: **proved for predictable sensing followed by a finite fresh
deployment; not a theorem for general interleaved asynchronous MARL.**

## Setting

Let the unknown regime be (j\in\{0,1\}).  The sensing history is

\[
\mathcal F_t=\sigma(U,A_1,Y_1,\ldots,A_t,Y_t),
\]

where the auxiliary randomization (U) is independent of the regime and
(A_t=(q_t,b_t,\eta_t)) is \(\mathcal F_{t-1}\)-measurable.  The sensing phase
ends at a dual-budget-bounded stopping time \(\tau\).  The final deployment
\(\widehat U\in\mathcal U_B\) is \(\mathcal F_\tau\)-measurable.  The finite
catalogue \(\mathcal U_B\) contains the deployed action and its remaining
budget or horizon, so two deployments with different continuation resources
are distinct elements.

For each regime and deployment define the public excess-risk table

\[
\Delta_j(u)=L_j(u)-L_j^*\ge 0.
\]

The deployment must use a fresh or reset Markov stream, or otherwise satisfy
the following noncancellation inequality for the complete policy:

\[
\operatorname{Reg}_j(\pi)
\ge \mathbb E_j\Delta_j(\widehat U).
\tag{1}
\]

Equation (1), rather than independence by itself, is the assumption used by
the reduction.

## Separation and decoder

Define

\[
g_{\rm sep}
=\min_{u\in\mathcal U_B}\max\{\Delta_0(u),\Delta_1(u)\}.
\]

The necessary action-separation condition is (g_{\rm sep}>0): no deployment
is simultaneously optimal in both regimes.  Define the deterministic proof
decoder

\[
d(u)=
\begin{cases}
1,&\Delta_0(u)\ge \Delta_1(u),\\
0,&\Delta_0(u)<\Delta_1(u),
\end{cases}
\]

and

\[
g_j=\min_{u:d(u)\ne j}\Delta_j(u).
\]

For a finite catalogue, (g_{\rm sep}>0) implies (g_0,g_1>0).  The decoder
uses both fixed hypothesis models and their public risk tables.  It does not
reveal the true regime to the algorithm and is therefore not circular.

## Theorem 1: excess learning risk implies a regime test

Let \(\phi=d(\widehat U)\),
\(\alpha=\mathbb P_0(\phi=1)\), and
\(\beta=\mathbb P_1(\phi=0)\).  Under (1),

\[
\alpha\le \frac{\operatorname{Reg}_0(\pi)}{g_0},
\qquad
\beta\le \frac{\operatorname{Reg}_1(\pi)}{g_1}.
\tag{2}
\]

### Proof

On the event \(\phi\ne j\), the definition of \(g_j\) gives
\(\Delta_j(\widehat U)\ge g_j\).  Consequently,

\[
\operatorname{Reg}_j(\pi)
\overset{(1)}\ge
\mathbb E_j\Delta_j(\widehat U)
\ge g_j\mathbb P_j(\phi\ne j),
\]

which proves (2).  \(\square\)

## Corollary 1: connection to AC-7

Assume the stopped adaptive change-of-measure theorem AC-7 applies.  When
\(\alpha+\beta<1\), binary data processing yields

\[
\mathbb E_0\!\sum_{t\le\tau}\mathcal I_{01}(Z_t,A_t)
\ge \operatorname{kl}(1-\alpha,\beta),
\]

\[
\mathbb E_1\!\sum_{t\le\tau}\mathcal I_{10}(Z_t,A_t)
\ge \operatorname{kl}(1-\beta,\alpha).
\]

Substituting (2), with the usual monotonicity on the separated error region,
gives

\[
\mathbb E_1\!\sum_{t\le\tau}\mathcal I_{10}(Z_t,A_t)
\ge
\operatorname{kl}\!\left(
1-\frac{\operatorname{Reg}_1}{g_1},
\frac{\operatorname{Reg}_0}{g_0}
\right).
\tag{3}
\]

In particular, if \(\max_j\operatorname{Reg}_j\le r<\min(g_0,g_1)/2\), the
information requirement is at least
\(\operatorname{kl}(1-r/g,r/g)\), where \(g=\min(g_0,g_1)\).

## Corollary 2: safe identification opportunity cost

Suppose the all-agent baseline is optimal under regime zero,
\(\operatorname{Reg}_0\le\epsilon\), and
\(\operatorname{Reg}_1\le r\).  Then (3) becomes

\[
\mathbb E_1\!\sum_{t\le\tau}\mathcal I_{10}(Z_t,A_t)
\ge
\operatorname{kl}\!\left(1-\frac r{g_1},\frac\epsilon{g_0}\right).
\tag{4}
\]

If every identification action lies in a predictable pathwise-safe catalogue
\(\mathcal A_{\rm safe}\), define the regime-one occupation cost

\[
\mathcal C^{\rm safe}_{1,B}(x)
=\inf_{\nu\in\mathcal O^{\rm safe}_{1,B}:
\int \mathcal I_{10}\,d\nu\ge x}
\int c_1\,d\nu,
\]

where \(c_1\) is opportunity loss relative to the regime-one oracle.  Every
policy in this restricted class obeys the implicit bound

\[
r\ge
\mathcal C^{\rm safe}_{1,B}\!\left[
\operatorname{kl}\!\left(1-\frac r{g_1},\frac\epsilon{g_0}\right)
\right].
\tag{5}
\]

If \(\mathcal I_{10}(z,a)\le\Gamma_Bc_1(z,a)\) throughout the safe catalogue,
(5) implies

\[
r\ge\Gamma_B^{-1}
\operatorname{kl}\!\left(1-\frac r{g_1},\frac\epsilon{g_0}\right).
\]

For mutually absolutely continuous finite Gaussian histories,
\(\epsilon=0\) and (r<g_1) require infinite discriminating information.
Thus finite-budget exact no-harm forbids a nontrivial switch unless a common
safe-optimal action or a zero-harm informative action exists.

## Why the general online claim does not follow

### Expected safety does not imply pathwise-safe occupation

A single-regime occupation relaxation under regime one can choose the
regime-one oracle immediately, assigning zero opportunity cost while ignoring
the damage caused by the same predictable kernel under regime zero.  A valid
expected-safety lower bound therefore needs coupled occupation measures
\((\nu_0,\nu_1)\), common policy-kernel constraints, and likelihood-flow
constraints.

A one-round counterexample is enough.  Let actions (a_0,a_1) be the two
regime oracles, with wrong-action gap (g), and choose (a_1) with probability
(p) before observing data.  Regime-zero safety requires (gp\le\epsilon),
while regime-one regret is (g(1-p)\ge g-\epsilon).  A regime-one-only
occupation program incorrectly chooses (a_1) and returns zero.

### Interleaved data can violate noncancellation

Without a fresh deployment or the explicit inequality (1), a wrong deployment
may occur only on conditionally easy histories.  Conditional risk can then be
below the unconditional oracle table and cancel the wrong-action gap in total
expected excess.  A continuously interleaved learner may also have no single
final deployment action from which to decode a test.

## Scope decision

- **GO:** arbitrary predictable sensing, bounded stopping, a finite separated
  deployment catalogue, and fresh/reset deployment or explicit
  noncancellation.
- **Conditional GO:** pathwise-safe identification, for which (5) supplies a
  valid information--opportunity-cost lower bound.
- **NO-GO with current machinery:** general asynchronous MARL with continuing
  interleaved learning and only expected no-harm.

The next missing theorem for the broader goal is a two-law safety-constrained
occupation program coupled by one predictable policy kernel.  Until that is
proved and matched by an executable policy, this result must not be described
as a general asynchronous-MARL convergence or safety theorem.
