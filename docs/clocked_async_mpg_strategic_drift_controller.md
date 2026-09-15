# Strategic-drift controller for practical asynchronous MARL

Status: theorem/interface design with executable scalar optimizer.  It extends
the pathwise constant-step mechanism to a deep-policy implementation without
assuming that a neural-network global Lipschitz matrix is known.  It has not
yet received standard-benchmark evidence.

## 1. One story, not a second research problem

Centralized training launches one single-flight rollout packet for each
non-shared actor.  Execution remains decentralized and uses no extra
communication.  Fast actors can remove a rollout barrier, but a packet arriving
late was evaluated against old teammate policies.  Therefore two forces act on
the same update:

1. **clock gain:** heterogeneous workers produce useful packets sooner;
2. **strategic drift:** coupled teammate policies move while the packet is in
   flight, reducing the packet's current value.

The controlled exact and stochastic MPG experiments established this
rate--coupling phase.  The practical algorithm should measure the second force
in policy space rather than use delay alone.  A delay of ten harmless teammate
updates and a delay of one large interacting update need not have the same
risk.

## 2. Arrival-time lower bound

Let the shared training objective be `J(theta_i, theta_-i)`.  Agent `i` owns
`theta_i`, and single flight means its arriving proposal `d_i` was computed at
the current own block but at a birth-time teammate vector `bar theta_-i`.
Write the current teammate vector as `theta_-i` and scale the proposal by
`x in [0,1]`.

Assume on the declared trust region that

\[
|d_i^\top\nabla^2_{ii}J\,d_i|\le 2C_i,
\qquad
\|d_i^\top\nabla^2_{i,-i}J\|_*\le M_i\|d_i\|. \tag{1}
\]

Suppose a packet-derived simultaneous confidence statement gives the
birth-time directional derivative lower bound

\[
d_i^\top\nabla_iJ(\theta_i,\bar\theta_{-i})\ge A_i. \tag{2}
\]

Apply the fundamental theorem of calculus first along the own-proposal path
and then along the teammate-policy path.  For
`Delta_-i=||theta_-i-bar theta_-i||`, equations (1)--(2) give

\[
\boxed{
J(\theta_i+xd_i,\theta_{-i})-J(\theta_i,\theta_{-i})
\ge A_i x-C_i x^2-S_i x,}
\qquad
S_i=M_i\|d_i\|\Delta_{-i}. \tag{3}
\]

Any Markov-packet confidence radius is subtracted inside `A_i`; any certified
upper error in `M_i` or `Delta_-i` is added to `S_i`.  Crucially, the stale term
vanishes at `x=0`.  A coarse bound that subtracts a constant even when the
proposal is rejected cannot support a meaningful controller.

For tabular finite-horizon softmax policies, bounded rewards and scores give
finite explicit `C_i` and `M_i`.  For deep actors, the implementation will use
a trust-region curvature upper bound and a held-out upper confidence estimate
of teammate KL drift.  That empirical substitute must be reported as a deep
extension; the exact theorem must not silently claim global neural smoothness.

## 3. Lyapunov debt and a closed-form online action

Define the certified interaction penalty

\[
r_k(x)=C_kx^2+S_kx. \tag{4}
\]

The nonnegative virtual queue

\[
Q_{k+1}=[Q_k+r_k(x_k)-\epsilon_k]_+ \tag{5}
\]

tracks a declared cumulative certificate-risk budget.  Its quadratic
Lyapunov drift obeys

\[
\frac{Q_{k+1}^2-Q_k^2}{2}
\le \frac{(r_k(x_k)-\epsilon_k)^2}{2}
+Q_k(r_k(x_k)-\epsilon_k). \tag{6}
\]

At each packet arrival, minimize this drift upper bound while rewarding the
observable directional gain.  Equivalently, maximize

\[
V A_kx-Q_k(C_kx^2+S_kx),\qquad 0\le x\le x_{\max}. \tag{7}
\]

The solution is not a candidate scan and needs no Hessian inverse or generic
QP:

\[
x_k^*=\Pi_{[0,x_{\max}]}
\left(\frac{VA_k-Q_kS_k}{2Q_kC_k}\right), \tag{8}
\]

with the obvious boundary solution when `Q_k C_k=0`.  It is `O(1)` after the
packet statistics and teammate drift have been computed.  A strict no-harm
mode replaces `x_max` by

\[
x_{\rm safe}=\min\{x_{\max},[(A_k-S_k)/C_k]_+\}, \tag{9}
\]

with the linear-case convention.  The budgeted mode is more useful for
learning because it can take temporarily risky updates and subsequently raises
their price.

Projection in (5) gives the pathwise accounting identity

\[
\sum_{k<K}r_k(x_k)\le\sum_{k<K}\epsilon_k+Q_K. \tag{10}
\]

On the simultaneous confidence event, summing (3) relates this debt directly
to cumulative objective change.  This is why the virtual queue is not an
unrelated systems constraint: it prices the same curvature-plus-strategic-drift
term that appears in the policy performance bound.

### Predictable noisy directional values

A high-probability confidence sequence is sufficient but is not necessary for
an expectation-level finite-time result.  Let a sample-split validation half
give

\[
\widehat A_k=A_k+\xi_k,\qquad
\mathbb E[\xi_k\mid\mathcal G_k]=0,qquad
\mathbb E[|\xi_k|\mid\mathcal G_k]\le s_k, \tag{11}
\]

where `G_k` includes the proposal half, current policy, debt and certificate
coefficients but not the independent validation half.  The selected `x_k`
depends on `xi_k`, so replacing `A_k` by `Ahat_k` without a selection term would
be wrong.  Since `0<=x_k<=1`, however,

\[
\mathbb E[A_kx_k\mid\mathcal G_k]
\ge
\mathbb E[\widehat A_kx_k\mid\mathcal G_k]-s_k. \tag{12}
\]

Thus the sample-split controller pays an explicit `sum_k s_k` term rather than
claiming conditional unbiasedness after selection.  If the scalar validation
inner product has variance proxy `sigma_k^2/m_k`, then
`s_k<=sigma_k/sqrt(m_k)` by Cauchy--Schwarz.

The drift-plus-penalty comparison also survives.  For any predictable
comparator `y_k` satisfying `r_k(y_k)<=epsilon_k`, the exact minimization in
(7), the queue drift (6), and (12) give

\[
\frac1K\sum_{k<K}\mathbb E[A_kx_k]
\ge
\frac1K\sum_{k<K}\mathbb E[A_ky_k]
-\frac{B_Q}{V}
-\frac1K\sum_{k<K}s_k
-\frac{Q_0^2}{2VK}, \tag{13}
\]

where `B_Q` is any uniform upper bound on
`(r_k(x)-epsilon_k)^2/2`.  Equation (13) is a finite-horizon certified-gain
comparison, not a last-iterate convergence theorem.  Combining (3), (10) and
(12) further yields

\[
\mathbb E[J(\theta^K)-J(\theta^0)]
\ge
\sum_{k<K}\mathbb E[\widehat A_kx_k-r_k(x_k)]
-\sum_{k<K}s_k. \tag{14}
\]

For vanishing average validation error, (13)--(14) recover the oracle
controller asymptotically.  With fixed batch size they give a noise floor,
which matches the observed terminal tradeoff and must be reported.

## 4. Relation to the pathwise Krasovskii theorem

There are two levels of the same mechanism:

- the pathwise Lyapunov--Krasovskii theorem uses known cross-block smoothness
  to select a conservative constant step for arbitrary bounded completion
  order and proves finite-time stationarity;
- the strategic-drift controller observes each proposal's directional value
  and teammate policy drift, then chooses its scale online via (8).

The publication theorem should use a composite potential consisting of the
objective gap, the interaction-history energy, and `Q_k^2/2`.  The first two
control delayed stochastic approximation; the third controls cumulative
certificate risk.  A complete theorem still must show that the confidence
process and policy-drift proxy dominate the coefficients in (3) for the stated
tabular MPG class.  The deep version may inherit the controller architecture
but must be labeled approximate unless that domination is verified.

## 5. Pre-GPU gates

No standard MARL run is authorized until all of the following are fixed:

1. the exact finite-horizon softmax constants or a formally stated local trust
   region for (1);
2. either a simultaneous Markov-packet lower confidence bound for `A_k` or the
   predictable-noise assumptions and `s_k` term in (11)--(14);
3. a teammate-drift upper certificate that is zero at packet birth;
4. the risk-budget and hard-shield semantics, with no retrospective switching;
5. distinct-policy MAPPO/HAPPO, raw async, delay-only async, fully utilized
   barrier, and the proposed drift controller under identical transition work;
6. actual and logical wall-clock, utilization, transition work, policy drift,
   time-to-return, final return, and certificate-debt reporting.

This interface keeps the research claim narrow: it is not generic asynchronous
RL, and it is not asynchronous action execution.  It is wall-clock stochastic
optimization of distinct interacting policies under observable strategic
staleness.
