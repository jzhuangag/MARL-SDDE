# T-052 exact two-agent fingerprint decision theorem

## Why T-052 is a new theorem, not a T-051A rescue

T-051A permanently fails S7 at 1.0501063679 versus the frozen 1.05 gate. Its
certificate treated each block match rate as an arbitrary bounded variable.
T-052 uses an additional registered structural fact: (q_p=2). With exactly
two probe actors, the block statistic is a single match indicator and hence
has a fully known Bernoulli law under the trajectory-switch model. T-051A is
not reanalysed or relabelled; any use of T-052 requires a new experiment ID.

## Theorem 1: exact probe count law

For task collision probability (c_L), T-051 gives the pair-match
probability

\[
 m(\rho)=c_L+(1-c_L)\rho.
\]

With two actors, block (j) produces (W_j\in\{0,1\}), so independent probe
blocks satisfy

\[
 K_n=\sum_{j=1}^nW_j\sim\operatorname{Binomial}(n,m(\rho)). \tag{1}
\]

No independence among pairs is needed because there is only one pair.
Accidental state-trajectory collisions remain in (m(\rho)).

## Theorem 2: exact plug-in action risk

For a realised count (k), the controller forms

\[
 \widehat\rho_k=\Pi_{[0,1]}
 \frac{k/n-c_L}{1-c_L}
\]

and selects

\[
 \widehat q_k\in\arg\min_{q\in\mathcal Q}
 (h+q)\left(\widehat\rho_k+
 \frac{1-\widehat\rho_k}{q}\right),                   \tag{2}
\]

with the public smaller-(q) tie break. For true correlation \(\rho\), its
expected leading coefficient is exactly

\[
 \mathbb E K_{\widehat q}(\rho)
 =\sum_{k=0}^n {n\choose k}m(\rho)^k[1-m(\rho)]^{n-k}
 K_{\widehat q_k}(\rho).                              \tag{3}
\]

Equation (3) is a finite sum, not a fitted estimate or an asymptotic normal
approximation. It automatically prices near-boundary classification errors
by their actual action gap. CPU tests verify that (3) never exceeds the valid
T-051 Hoeffding upper bound on the registered grid.

## Corollary: fully charged leading risk ratio

Each two-agent block costs (h+2) message units. If (B_L) remains for
learning and (q_0) is the strong no-probe fixed baseline, then

\[
 \frac{R_{\rm plug}(\rho)}{R_{q_0}(\rho)}
 =\frac{B_L+n(h+2)}{B_L}
 \frac{\mathbb E K_{\widehat q}(\rho)}{K_{q_0}(\rho)}. \tag{4}
\]

Fingerprint environment transitions and delay must still be charged by any
experiment. Equation (4) is exact for the leading stationary risk law; a
sampled finite-horizon claim still requires the public contraction horizon
and a prospective experiment.

Computing (3) costs (O(n+|\mathcal Q|n)) offline or at audit time. Online,
the controller counts matches and scans the three-action catalogue. There is
no matrix estimate, preconditioner, or additional online optimization.
