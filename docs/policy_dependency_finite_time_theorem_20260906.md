# Finite-time theorem for sparse policy-dependency synchronization

Date: 2026-09-06

Status: proved abstract drift-plus-penalty theorem plus a stated MARL
sufficient condition.  This document does not claim that the Markov estimator
or the controlled-SDDE approximation is complete, and it does not authorize a
learning experiment.

## 1. Event-time object

Let `F_k` be the information just before asynchronous decision epoch `k` and
let `A_k` be a finite predictable action set containing a zero-cost null
action.  The action is either a dispatch-time cache refresh or a receipt-time
packet step; the physical actions are never collapsed into a simultaneous
decision.

Let

\[
U_k=F(\theta^k)-F_\star+C_k+H_k\ge 0
\tag{1}
\]

be the learning, outgoing-cache, and pending-rollout energy defined in the
four-event ledger.  For every `a` in `A_k`, let `D_k(a)` be an
`F_k`-measurable upper bound satisfying

\[
\mathbb E[U_{k+1}-U_k\mid\mathcal F_k,A_k=a]\le D_k(a).
\tag{2}
\]

The communication cost is `c_k(a)` in `[0,c_max]`, the target average cost is
`bar c`, and

\[
Q_{k+1}=[Q_k+c_k(A_k)-\bar c]^+.
\tag{3}
\]

The executable score `hat D_k(a)` may be noisy.  Define its conditional
expected uniform error

\[
e_k=\mathbb E[\sup_{a\in A_k}|\widehat D_k(a)-D_k(a)|
\mid\mathcal F_k].
\tag{4}
\]

For a tradeoff parameter `V>0`, the policy selects

\[
A_k\in\arg\min_{a\in A_k}
\{V\widehat D_k(a)+Q_kc_k(a)\}.
\tag{5}
\]

Thus Lyapunov drift chooses both graph refreshes and receipt-time update mass;
it is not only a proof device.

## 2. Noisy drift-comparison lemma

For any `F_k`-measurable comparator distribution `pi_k` over `A_k`,

\[
\begin{aligned}
&\mathbb E[D_k(A_k)+Q_k(c_k(A_k)-\bar c)\mid\mathcal F_k]\\
&\quad\le
\mathbb E_{a\sim\pi_k}[D_k(a)+Q_k(c_k(a)-\bar c)]
+2e_k.
\end{aligned}
\tag{6}
\]

The proof uses the minimizing property of (5) and applies the uniform score
error once to the selected action and once to the comparator.  The factor two
is therefore necessary without a one-sided estimator.

The queue identity gives

\[
\frac{Q_{k+1}^2-Q_k^2}{2}
\le Q_k(c_k(A_k)-\bar c)+B_Q,
\qquad B_Q=\frac{c_{\max}^2}{2},
\tag{7}
\]

when `0<=bar c<=c_max`.

## 3. Main finite-time theorem

Assume (1)--(5), `0<=bar c<=c_max`, and that a predictable comparator
`pi_k` satisfies

\[
\mathbb E_{a\sim\pi_k}[c_k(a)\mid\mathcal F_k]\le\bar c
\tag{8}
\]

and

\[
\mathbb E_{a\sim\pi_k}[D_k(a)\mid\mathcal F_k]
\le-\gamma S_k+r_k,
\qquad \gamma>0,
\tag{9}
\]

where `S_k` is the owner-block stationarity measure at receipt epochs and is
zero at other event types.  Then, for every `K>=1`,

\[
\frac1K\sum_{k=0}^{K-1}\mathbb E[S_k]
\le
\frac{U_0+Q_0^2/(2V)}{\gamma K}
+\frac{B_Q}{\gamma V}
+\frac{\sum_{k<K}\mathbb E[r_k+2e_k]}{\gamma K}.
\tag{10}
\]

### Proof

Multiply (2) and (6) by `V`, add (7), and use (8)--(9).  This yields

\[
\begin{aligned}
&\mathbb E\left[V(U_{k+1}-U_k)
+\frac{Q_{k+1}^2-Q_k^2}{2}\right]\\
&\quad\le -V\gamma\mathbb E[S_k]
+V\mathbb E[r_k+2e_k]+B_Q.
\end{aligned}
\tag{11}
\]

Sum (11), telescope, use `U_K>=0` and `Q_K^2>=0`, then divide by
`V gamma K`.  This proves (10).

The theorem is an event-time stationarity guarantee, not a last-iterate Nash
equilibrium result.  With bounded cumulative average remainders and
`V` proportional to `sqrt(K)`, the queue price contributes
`O(K^{-1/2})`.  A vanishing total bound additionally requires the average
estimation, Markov, and sparse-tail remainders to vanish.

## 4. Pathwise communication budget

Equation (3) implies

\[
\frac1K\sum_{k<K}c_k(A_k)
\le\bar c+\frac{Q_K-Q_0}{K}.
\tag{12}
\]

Assume every non-null action costs at least `c_min>0` and
`|hat D_k(a)|<=D_max`.  Comparing any positive-cost action with the null
action shows that the null action is selected whenever

\[
Q_k>\frac{2VD_{\max}}{c_{\min}}.
\tag{13}
\]

Consequently

\[
Q_k\le \frac{2VD_{\max}}{c_{\min}}+c_{\max}
\tag{14}
\]

for `Q_0=0`, and the budget violation in (12) is `O(V/K)`.  Setting
`V` proportional to `sqrt(K)` balances learning and budget errors.

## 5. MARL sufficient condition for the comparator

At a receipt epoch owned by `i`, write

\[
a_i=\nabla_iF(\theta^k),\qquad
d_i=\sum_{\ell\ne i}p_{\ell i}
(\theta_i^k-\chi_{i\to\ell}^k).
\tag{15}
\]

Let a fully charged reference packet satisfy

\[
\mathbb E_k[g_i]=a_i+b_i,\quad
\|b_i\|\le\beta_{i,k},\quad
\mathbb E_k\|g_i-\mathbb E_kg_i\|^2\le\sigma_i^2.
\tag{16}
\]

Let `delta_i,k>=||d_i||`, let the pending-history linear coefficient be
`h_i,k`, and let the total quadratic curvature be

\[
\Lambda_{i,k}=L_i+P_i^{out}+\kappa_{i,k}.
\tag{17}
\]

For a reference step `eta<=1/(4 Lambda_i,k)`, the return-time inequality gives

\[
D_k(a^{ref})
\le-\frac\eta2\|a_i\|^2+r_{i,k}^{ref},
\tag{18}
\]

where one valid explicit remainder is

\[
\begin{aligned}
r_{i,k}^{ref}={}&\eta(\beta_{i,k}+\delta_{i,k}+h_{i,k})^2
+\eta\delta_{i,k}\beta_{i,k}\\
&+\eta h_{i,k}(\beta_{i,k}+\sigma_i)
+\frac{\Lambda_{i,k}\eta^2}{2}
(2\beta_{i,k}^2+\sigma_i^2)
+r_{i,k}^{birth}.
\end{aligned}
\tag{19}
\]

Here `r_birth` is the declared replacement-rollout energy minus the completed
packet's removed history energy.  Equation (18) follows from
`E||g_i||<=||a_i||+beta_i,k+sigma_i`, the four-event drift identity, Young's
inequality, and the step restriction.  It is deliberately conservative but
contains only named quantities.

If receipt owner `i` has conditional activation probability at least
`pi_min>0`, then (9) holds with

\[
S_k=\|\nabla F(\theta^k)\|^2\mathbf 1\{k\text{ is a receipt}\},
\qquad \gamma=\frac{\eta\pi_{\min}}2,
\tag{20}
\]

after averaging the owner-specific remainder.  The sparse estimator adds
`r_i,k^sp` from the sparse-dependency audit to `e_k` or `r_k`, but never to
both.

## 6. What is and is not closed

Closed here are the noisy action-comparison factor, queue drift, telescoping
finite-time bound, explicit queue cap, and a sufficient receipt-step
remainder.  The exact deterministic cache and pending-history identities were
proved separately and are inputs to `D_k`.

Still open are:

1. a Markov or regenerative construction proving (16) for the executable
   replay estimator;
2. a nonvacuous bound for `e_k`, including the sparse tail and Taylor error;
3. activation and replacement conditions making the average of (19) small;
4. a finite-horizon weak approximation from the discrete process to the
   controlled hybrid SDDE.

The discrete theorem does not depend on the SDDE.  The SDDE remains a phase
and policy-design interpretation only until item 4 is proved.
