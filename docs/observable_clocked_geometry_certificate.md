# Observable geometry certificate for Lyapunov-clocked optimism

## Causal contract

At asynchronous learner event \(k\), an agent block \(i_k\) is selected by the
arrival clock.  The optimism decision must be made before buying the second
current-parameter oracle.  It may use only the history through event \(k-1\),
the identity of the arriving block, public clock estimates, and any probe whose
trajectory/oracle/communication cost has already been charged.  Computing a
lookahead gradient and then deciding not to count it is forbidden.

The local pseudo-gradient model used by the first certificate layer is

\[
g_k=A_k x_k+\zeta_k,
\]

where \(A_k\) is a local game Jacobian and \(\zeta_k\) contains Markov sampling
and critic error.  Suppose the past-data estimator supplies the predictable
event

\[
\|A_k-\widehat A_k\|_2\le\varepsilon_{A,k},
\qquad
\max\{\|A_k\|_2,\|\widehat A_k\|_2\}\le L_k.
\]

This document derives what can be certified from that event.  It does not yet
claim that a usable \(\varepsilon_{A,k}\) has been obtained from general Markov
policy-gradient data.

## From operator error to a safe log-drift value

For coordinate \(i\), define

\[
M_{i,0}(A)=I-\eta E_iA,
\qquad
M_{i,1}(A)=I-\eta E_iA(I-\eta A),
\]

where action one purchases the fresh extra-gradient oracle.  With arrival
probabilities \(r_i\) and a common positive-definite clock metric \(P\), let

\[
q_u(A)=\lambda_{\max}\!\left(
P^{-1/2}\sum_i r_iM_{i,u}(A)^\top P M_{i,u}(A)P^{-1/2}
\right).
\]

Weyl's inequality and

\[
\|M^\top PM-\widehat M^\top P\widehat M\|_2
\le \|P\|_2\|M-\widehat M\|_2
   (\|M\|_2+\|\widehat M\|_2)
\]

give the following conservative radii, where
\(\kappa_P=\|P\|_2\|P^{-1}\|_2\):

\[
R_0=2\kappa_P(1+\eta L)\eta\varepsilon_A,
\]

\[
R_1=2\kappa_P(1+\eta L+\eta^2L^2)
\eta\varepsilon_A(1+2\eta L).
\]

Therefore a safe lower bound on the log-drift value of optimism is

\[
\underline\Delta_k=
\log\{q_0(\widehat A_k)-R_{0,k}\}
-\log\{q_1(\widehat A_k)+R_{1,k}\},
\]

provided the first logarithm is positive; otherwise the certificate returns
\(-\infty\) and the controller fails closed.  The action becomes

\[
u_k=1\{V\underline\Delta_k>Z_k\}.
\]

This is the same variable used by the finite-budget Lyapunov controller, not a
post-hoc phase label.

The implementation in `observable_geometry.py` verifies the matrix interval
against sampled perturbations and confirms that exact rotation is positive,
exact potential geometry is negative, and large uncertainty fails closed.

## What remains statistically open

A generic full \(d\times d\) Jacobian estimate costs \(O(d^2)\) storage and is
not the intended deployed algorithm.  The low-complexity route must exploit
block structure: exponentially weighted cross-block secants or fixed-rank JVP
sketches can be updated in \(O(rd)\) arithmetic and memory for constant sketch
rank \(r\).  A full-space certificate from such a sketch additionally needs a
declared residual spectral bound; a sketch alone cannot certify an unseen
orthogonal direction.

For Markov actor--critic data, the required confidence event must also handle:

1. adaptively chosen regressors and block arrivals;
2. temporal mixing bias and critic approximation error;
3. metric changes as the policy moves;
4. loss of persistent excitation near equilibrium;
5. simultaneous coverage over every decision time used by the queue.

Without an excitation condition or a separately charged probe, potential and
rotational operators can be observationally indistinguishable on the visited
subspace.  Hence an unrestricted no-probe certificate is impossible.  The
next CPU feasibility audit must explicitly vary excitation, mixing, arrival
imbalance, switch persistence, and signal-to-noise ratio.  It must stop if the
confidence radius makes the controller collapse to `never`, or if a required
probe consumes the dynamic headroom found in LCO-H1.

## Authorization boundary

This deterministic perturbation layer is theorem support only.  It authorizes
neither a sensor pilot nor GPU work by itself.  A separate outcome-free design
must first specify the low-rank estimator, its charged observations, the
confidence theorem, new development seeds, and headroom-retention gates.
