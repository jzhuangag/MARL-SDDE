# Publication theorem chain for Two Clocks

Date: 2026-09-02.

Status: consolidated theorem-facing statement and proof map.  It contains no
new empirical result.  The chain is exact under the assumptions below; neural
PPO/Adam/learned-critic implementations remain empirical extensions.

## 1. One model and one executable rule

Let `theta=(theta_1,...,theta_n)` be distinct policy blocks in a finite
discounted Markov potential game, and minimize
`f(theta)=Phi_star-Phi(theta)>=0`.  Agent `i` alone writes `theta_i` and has at
most one fixed-work packet in flight.  At learner event `k`, owner `I_k=i`
returns a packet born at joint-policy version `b_k`.  It is applied immediately,

\[
 \theta_i^{k+1}=\theta_i^k-\alpha\widehat g_{i,k},
\]

and the owner launches its next packet from the new joint policy.  All other
blocks remain unchanged.  Single-flight ownership gives the pathwise identity

\[
 \theta_i^k=\theta_i^{b_k}.                                      \tag{1}
\]

Thus a packet is fresh in its owner block even when teammate blocks have
changed.  This is event-driven centralized training with unchanged
decentralized execution.

Assume the following on the theorem class.

1. `f` is lower bounded and block smooth, and
   \[
   \|\nabla_i f(x)-\nabla_i f(y)\|
   \le\sum_j L_{ij}\|x_j-y_j\|,\quad L_{ij}\ge0.                 \tag{2}
   \]
2. Event delay is at most `D`, and every policy block completes at least once
   in every `B` consecutive learner events.
3. A packet uses a fresh reset replica, a fixed horizon `H`, and a declared
   batch size `m`.  Completion metadata is conditionally independent of its
   trajectory innovation.
4. At the pre-application filtration,
   \[
   \widehat g_{i,k}=\nabla_i f(\theta^{b_k})+b_{i,k}+\zeta_{i,k},\quad
   \|b_{i,k}\|\le\beta_i,
   \quad\mathbb E_k\zeta_{i,k}=0,
   \quad\mathbb E_k\|\zeta_{i,k}\|^2\le\nu_i^2/m.              \tag{3}
   \]
5. The applied packet norm is pathwise bounded by `Ghat`.

For tabular factorized softmax policies, the finite-horizon likelihood-gradient
identity supplies (3), with explicit truncation bias and packet bound.  No
within-trajectory iid assumption is used.

## 2. The upper theorem

Let `L_off` be `L` with zero diagonal and define

\[
 \ell_i^{\rm off}=\sum_{j\ne i}L_{ij},\qquad
 u_j=\max_i\ell_i^{\rm off}L_{ij}^{\rm off},\qquad
 w_j=\alpha u_j.                                                \tag{4}
\]

For applied steps `Delta^r=theta^(r+1)-theta^r`, set

\[
 \mathcal H_k=\sum_{d=1}^{D}(D-d+1)
 \sum_jw_j\|\Delta_j^{k-d}\|^2,
 \qquad
 \mathcal V_k=f(\theta^k)+\frac{D(1+\delta)}2\mathcal H_k.       \tag{5}
\]

Choose the common step no larger than the smallest positive block root of

\[
 L_{jj}\alpha+(1+\delta)D^2u_j\alpha^2\le1.                    \tag{6}
\]

### Theorem 1: pathwise event-time drift

For every bounded completion order and every possible arriving block `i`,

\[
 \mathbb E_k\mathcal V_{k+1}
 \le \mathcal V_k-\frac\alpha2\|\nabla_i f(\theta^k)\|^2
 +R_i(\alpha),                                                  \tag{7}
\]

where

\[
 R_i(\alpha)=\frac{1+\delta^{-1}}2\alpha\beta_i^2
 +\frac12\left[L_{ii}\alpha^2
 +(1+\delta)D^2w_i\alpha^2\right]\frac{\nu_i^2}{m}.             \tag{8}
\]

The proof uses (1) before applying (2), so diagonal owner drift never enters
the history energy.  Weighted Cauchy--Schwarz bounds only teammate motion.  The
triangular shift of (5) removes one copy of every live past step and adds `D`
copies of the new step; (6) absorbs the remaining stale-gradient and curvature
terms.  Conditional centering cancels the innovation cross term, while Young's
inequality pays the predictable truncation bias.

Let

\[
 C_j=\sum_i\left(\sum_\ell L_{i\ell}\right)L_{ij},
 \qquad C_{\max}=\max_j C_j,
 \qquad R_{\max}=\max_iR_i.                                    \tag{9}
\]

The `B`-event coverage assumption converts activated-block descent into
full-policy stationarity:

\[
\boxed{
 \frac1{K-B+1}\sum_{k=0}^{K-B}
 \mathbb E\|\nabla f(\theta^k)\|^2
 \le
 \frac{4B\mathcal V_0}{\alpha(K-B+1)}
 +\frac{4BKR_{\max}}{\alpha(K-B+1)}
 +\frac{2B^2C_{\max}K\alpha^2\widehat G^2}{K-B+1}.}            \tag{10}
\]

For `K>=2B`, `alpha=Theta(K^(-1/2))`, fixed packet variance and a logarithmic
trajectory horizon making `beta_i(H)^2=O(K^(-1/2))`, (10) is
`O(B/sqrt(K)+B^2/K)` up to the displayed MPG constants.  It is an average or
random-iterate potential-stationarity result, not last-iterate global
optimality.

## 3. Wall-clock conversion

Suppose each service duration of agent `i` belongs to `[a_i,b_i]`, with
`a_i>0`.  By time `T`, at least

\[
 K_T^- =\sum_i\left\lfloor T/b_i\right\rfloor                 \tag{11}
\]

packets have completed.  A conservative event-delay and coverage bound is

\[
 \overline B=1+\max_i\sum_{j\ne i}
 \left(1+\left\lceil b_i/a_j\right\rceil\right).               \tag{12}
\]

Substituting `K=K_T^-` and `B=D=overline B` where one common bound is desired
gives a deterministic wall-clock guarantee.  Equation (11) records aggregate
useful compute; (12) records the slow-block and interaction-history price.
Neither can replace the other.

Under an explicit softmax-interiority and discounted-occupancy-mismatch
constant `C_MPG`, a uniformly sampled event iterate also satisfies

\[
 \mathbb E\operatorname{Gap}(\theta^{\rm out})
 \le C_{\rm MPG}\sqrt{\text{right-hand side of (10)}}.          \tag{13}
\]

Without that declared constant the result is potential stationarity, not a
uniform Nash guarantee.

## 4. Lower and impossibility side

### Theorem 2: strategically essential service clock

Consider the separable one-state identical-interest subclass

\[
 \Phi_v(x)=\sum_i\left(v_ix_i-\frac{\mu_i}{2}x_i^2\right),
 \qquad v_i\in\{-\Delta_i,+\Delta_i\},                          \tag{14}
\]

with Gaussian packet noise `N(0,sigma_i^2)`.  Any method returning an
`epsilon`-stationary point for `epsilon<Delta_i` with both sign errors at most
`delta<1/2` must satisfy

\[
 \mathbb E N_i\ge
 \frac{\sigma_i^2}{2\Delta_i^2}
 \operatorname{kl}(1-\delta,\delta).                            \tag{15}
\]

For periodic service time `s_i`, the `N_i`-th packet cannot arrive before
`s_iN_i`; hence

\[
 \mathbb ET\ge\max_i
 \frac{s_i\sigma_i^2}{2\Delta_i^2}
 \operatorname{kl}(1-\delta,\delta).                            \tag{16}
\]

For an independent Poisson clock of rate `lambda_i`, compensated-count
optional stopping gives the parallel expression

\[
 \mathbb ET\ge\max_i
 \frac{\sigma_i^2}{2\lambda_i\Delta_i^2}
 \operatorname{kl}(1-\delta,\delta).                            \tag{17}
\]

Thus aggregate completions by fast agents cannot erase a slow strategically
essential actor.  Equation (16) lies in a periodic subclass of the bounded
service model used by the upper theorem; (17) is a stochastic-clock extension.

### Proposition 3: strategic-staleness obstruction

If a returned direction has norm `s` and its current block gradient is known
only to lie in a radius-`B_s` ball around it, the best uniformly valid smooth
progress certificate along that direction is

\[
 \sup_{x\ge0}\inf_{\|g-\widehat g\|\le B_s}
 \left\{x\langle g,\widehat g\rangle
 -\frac L2x^2\|\widehat g\|^2\right\}
 =\frac{[s-B_s]_+^2}{2L}.                                      \tag{18}
\]

When cross-agent drift reaches the packet signal, no positive scalar step can
have a uniform positive certificate from that packet alone.  A two-agent
quadratic potential gives two indistinguishable birth packets with opposing
arrival-time cross effects, so this is an information obstruction rather than
an artifact of the upper proof.

## 5. The rate--coupling phase and what is matched

Theorems 1--2 and Proposition 3 form one phase statement:

- heterogeneous service can create more sequential fresh policy queries per
  unit time than a global frozen-policy barrier;
- only off-diagonal teammate motion consumes the single-flight Lyapunov
  margin;
- excessive interaction-weighted motion destroys uniformly certified stale
  progress;
- every full-policy guarantee still pays the clock of each essential block.

The matching is deliberately partial.  Stochastic accuracy and the essential
periodic/Poisson service clock match at order on a separable subclass.  The
off-diagonal upper term and (18) match the favorable/unfavorable coupling
boundary qualitatively.  A minimax lower bound reproducing every `D^2`, `B`
and Markov-mixing constant is not proved.  The paper must say
"instance-sensitive upper/lower phase", not "globally optimal rate".

The single-flight separation is strict on a declared family.  If
`L_off=0`, (6) permits `alpha<=min_j 1/L_jj` for arbitrary packet delay.
A black-box full-vector stale-gradient bound that places diagonal curvature in
the delay history pays a spurious delay penalty on the same separable game.
This is the mathematical role of distinct policy ownership.

## 6. Complexity and the role of Lyapunov/SDDE

Computing (4)--(6) once costs `O(n^2)` for a dense interaction envelope and
less for a sparse declared envelope.  Each arrival performs an ordinary
`O(dim(theta_i))` block update.  There is no online QP, Hessian inverse,
preconditioner, agent-count scan, paid sensor or dynamic collaboration graph.

Lyapunov--Krasovskii theory is both a design and analysis tool: its history
weights select the certified common step in (6), and the same functional gives
the finite-time bound.  An SDDE is unnecessary for correctness.  It may be
included only as a separately validated diffusion interpretation; it must not
replace the discrete event-time proof or be used to claim a stronger neural
result.

## 7. Proof and evidence map

- fixed-horizon likelihood gradient, bias and packet bound:
  `clocked_async_mpg_markov_packet_lemma.md`;
- pathwise drift, coverage window and wall-clock conversion:
  `clocked_async_mpg_pathwise_wallclock_theorem.md`;
- safe-progress and essential-agent lower side:
  `clocked_async_mpg_lower_phase_certificate.md`;
- conditional Nash conversion and exact finite-game checks:
  `clocked_async_mpg_markov_packet_lemma.md` and `trajectory_interface.py`;
- theorem/novelty matching boundary:
  `two_clocks_novelty_and_matching_audit.md`;
- neural noncoverage decision:
  `two_clocks_neural_certificate_audit.md`.

The two positive CPU confirmations validate the theorem-facing phase but are
not proof.  Standard neural MARL remains the next empirical gate and requires a
separate outcome-free GPU preregistration.
