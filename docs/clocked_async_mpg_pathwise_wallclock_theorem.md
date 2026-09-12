# Pathwise finite-time wall-clock theorem for clocked independent PG

Status: discrete theorem chain closed under the assumptions below; Nash
conversion remains conditional on the standard interiority/mismatch constant.
This document is theorem-facing and contains no empirical claim.

## 1. Executable architecture and assumptions

There are `n` distinct policy blocks.  Agent `i` alone writes `theta_i` and has
at most one packet in flight.  At event `k`, packet `I_k=i` was born at event
`b_k`; immediately after its application, agent `i` reads the new joint policy
and launches its next fixed-cost packet.  Hence

\[
\theta_i^k=\theta_i^{b_k}. \tag{1}
\]

Let `f=Phi_star-Phi` and let `g_i=nabla_i f`.  For all `x,y`, assume

\[
\|g_i(x)-g_i(y)\|
\le \sum_j L_{ij}\|x_j-y_j\|, \qquad L_{ij}\ge0, \tag{2}
\]

and block smoothness with diagonal constant `L_ii`.  The packet has the
pre-application decomposition

\[
\widehat g_{i,k}=g_i(\theta^{b_k})+b_{i,k}+\zeta_{i,k}, \tag{3}
\]

where `b_(i,k)` is predictable, `||b_(i,k)||<=beta_i`,
`E[zeta_(i,k)|F_k^-]=0`, and
`E[||zeta_(i,k)||^2|F_k^-]<=nu_i^2/m`.  The fixed-horizon Markov REINFORCE
lemma supplies these constants with `beta_i=beta_i(H)` and batch size `m`.
For the window conversion assume the packet estimator is pathwise bounded by
`Ghat`; the existing score/reward bound supplies such a finite value in the
tabular softmax model.

The completion order can be arbitrary and history-dependent.  It only must
satisfy:

1. event delay `k-b_k<=D`;
2. every block occurs at least once in every `B` consecutive events.

No iid event mark, stationary activation probability or independence between
successive completion identities is assumed.

## 2. Pathwise Lyapunov certificate

Set `L_off` equal to `L` with zero diagonal and define

\[
\ell_i^{\rm off}=\sum_{j\ne i}L_{ij},\qquad
u_j=\max_i \ell_i^{\rm off}L_{ij}^{\rm off},\qquad
w_j=\alpha u_j. \tag{4}
\]

For past applied steps `Delta^r=theta^(r+1)-theta^r`, define

\[
\mathcal H_k=
\sum_{d=1}^{D}(D-d+1)
\sum_j w_j\|\Delta_j^{k-d}\|^2, \tag{5}
\]

with missing pre-initialization steps zero, and

\[
\mathcal V_k=f(\theta^k)+\frac{D(1+\delta)}2\mathcal H_k,
\qquad \delta>0. \tag{6}
\]

Choose the common step no larger than the minimum positive root of

\[
L_{jj}\alpha+(1+\delta)D^2u_j\alpha^2\le1,
\quad j=1,\ldots,n. \tag{7}
\]

Computing (4)--(7) costs `O(n^2)` plus `n` scalar roots.  Applying a packet
costs `O(dim(theta_i))`.

### Lemma 1: one-event drift for any arriving block

For every realization of the past and every possible `I_k=i`, expectation
only over the centered packet innovation gives

\[
\begin{aligned}
\mathbb E[\mathcal V_{k+1}\mid\mathcal F_k^-,I_k=i]
\le{}&\mathcal V_k-\frac{\alpha}{2}\|g_i(\theta^k)\|^2\\
&+\frac{1+\delta^{-1}}2\alpha\beta_i^2\\
&+\frac12\left[
L_{ii}\alpha^2+(1+\delta)D^2w_i\alpha^2
\right]\frac{\nu_i^2}{m}. \tag{8}
\end{aligned}
\]

Proof.  By (1), the stale-gradient mismatch contains only teammate blocks.
Cauchy--Schwarz and the event-delay bound give

\[
\|g_i(\theta^k)-g_i(\theta^{b_k})\|^2
\le \ell_i^{\rm off}D
\sum_{r=b_k}^{k-1}\sum_jL_{ij}^{\rm off}
\|\Delta_j^r\|^2. \tag{9}
\]

The shift of (5) removes one copy of every stored squared step and adds `D`
copies of the new one.  Since `w_j>=alpha ell_i^off L_ij^off`, its negative
shift absorbs (9) for every `i`, rather than only on average over a random
mark.  The block-smoothness remainder and the new history step are nonpositive
after the stale-gradient square is combined precisely when (7) holds.
Young's inequality splits predictable bias from mismatch; conditional
centering cancels the linear innovation term and its second moment yields the
last line of (8).  This proves (8).  The implementation independently
enumerates compatible quadratic histories, all possible activated blocks,
two-point centered noise and nonzero bias.

Telescoping (8) for an arbitrary completion sequence gives, with

\[
R_{i}(\alpha)=\frac{1+\delta^{-1}}2\alpha\beta_i^2
+\frac12\left[L_{ii}\alpha^2
+(1+\delta)D^2w_i\alpha^2\right]\frac{\nu_i^2}{m}, \tag{10}
\]

the activated-block bound

\[
\sum_{k=0}^{K-1}\mathbb E\|g_{I_k}(\theta^k)\|^2
\le \frac{2\mathcal V_0}{\alpha}
+\frac{2}{\alpha}\sum_{k=0}^{K-1}R_{I_k}(\alpha). \tag{11}
\]

## 3. From activated blocks to joint-policy stationarity

Define

\[
C_j=\sum_i\ell_i L_{ij},\qquad
\ell_i=\sum_jL_{ij},\qquad C_{\max}=\max_j C_j. \tag{12}
\]

For every event `k` and block `i`, let `t_i(k)` be its first activation in
`[k,k+B-1]`.  Smoothness, Cauchy--Schwarz and the pathwise packet bound imply

\[
\|g_i(\theta^k)\|^2
\le2\|g_i(\theta^{t_i(k)})\|^2
+2\ell_iB\sum_{r=k}^{t_i(k)-1}
L_{i,I_r}\|\Delta^{r}_{I_r}\|^2. \tag{13}
\]

An activation can serve as `t_i(k)` for at most `B` starting events, and any
applied step belongs to at most `B` windows.  Summing (13) therefore yields the
deterministic window inequality

\[
\sum_{k=0}^{K-B}\|\nabla f(\theta^k)\|^2
\le 2B\sum_{k=0}^{K-1}\|g_{I_k}(\theta^k)\|^2
+2B^2C_{\max}K\alpha^2\widehat G^2. \tag{14}
\]

Combining (11)--(14), and writing `R_max=max_i R_i`, gives the finite-time
joint-policy result

\[
\boxed{
\frac1{K-B+1}\sum_{k=0}^{K-B}
\mathbb E\|\nabla f(\theta^k)\|^2
\le
\frac{4B\mathcal V_0}{\alpha(K-B+1)}
+\frac{4BK R_{\max}(\alpha)}{\alpha(K-B+1)}
+\frac{2B^2C_{\max}K\alpha^2\widehat G^2}{K-B+1}.} \tag{15}
\]

For `K>=2B`, `alpha` of order `K^(-1/2)`, `H` growing logarithmically so that
`beta_i(H)^2=O(K^(-1/2))`, and fixed batch variance, (15) is
`O(B/sqrt(K)+B^2/K)` up to explicit MPG constants.  This is a best-iterate or
uniform-random-iterate stationarity statement; it is not last-iterate global
optimality.

## 4. Deterministic wall-clock conversion

Suppose every service duration of agent `i` lies in `[a_i,b_i]`, with `a_i>0`.
By elapsed time `T`, that worker has completed at least `floor(T/b_i)` packets.
Consequently the first

\[
K_T^- = \sum_i\left\lfloor\frac{T}{b_i}\right\rfloor \tag{16}
\]

events occur no later than `T`.  A conservative valid event-coverage window is

\[
B=1+\max_i\sum_{j\ne i}
\left(1+\left\lceil\frac{b_i}{a_j}\right\rceil\right), \tag{17}
\]

and the same expression upper-bounds packet event delay after harmless extra
endpoint slack.  Substituting `K=K_T^-` in (15) is a wall-clock guarantee with
no renewal-limit approximation and no optional-stopping issue: it analyzes a
deterministic number of events guaranteed to have occurred by `T`.

Equation (16) exposes aggregate useful compute, while (17) exposes the slow-
essential-block and interaction-history cost.  This is the intended rate--
coupling tension.  It does not claim that aggregate rate erases a slow policy
block.

## 5. Conditional Nash conversion

Under the registered softmax interiority and distribution-mismatch condition,
suppose the MPG Nash gap satisfies

\[
\operatorname{Gap}(\theta)\le C_{\rm MPG}\|\nabla\Phi(\theta)\|. \tag{18}
\]

For a uniform random iterate among the first `K_T^--B+1` states, Jensen's
inequality converts (15) into

\[
\mathbb E\operatorname{Gap}(\theta^{\rm out})
\le C_{\rm MPG}\sqrt{\text{right-hand side of (15)}}. \tag{19}
\]

The constant `C_MPG` must be instantiated from policy-interiority and
occupancy-mismatch assumptions for the chosen MPG class.  Without such a
uniform constant the paper may claim potential stationarity, not a uniform
Nash guarantee.

## 6. Boundaries and remaining paper obligations

- Multiple simultaneous packets for one policy block break self-freshness and
  are outside the theorem.
- Unbounded service times require a high-probability truncation or a different
  infinite-history Lyapunov functional; (16)--(17) do not cover them.
- The theorem handles fixed-horizon episodic Markov packets reset from the
  declared start distribution.  A continuing-chain critic requires a Poisson-
  equation replacement for (3).
- `C_MPG` is still an explicit assumption, not a universally proved constant.
- A matching lower/separation result and standard nonlinear MARL evidence are
  still required for an ICML-level package.

Within these boundaries, the discrete delay, packet bias/noise, arbitrary
bounded asynchronous order and wall-clock conversion now belong to one
executable mechanism.  SDDE is unnecessary for correctness; it may appear only
as an optional diffusion interpretation if it yields an additional result.
