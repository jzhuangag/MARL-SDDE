# Finite-time theorem for Lyapunov-coupled asynchronous actor--critic

Date: 2026-09-05.

Status: **conditional theorem closed; the per-packet high-probability
implementation failed its preregistered nonvacuity gate.**

## 1. Setting and executable action

Let `f=-Phi` be a lower-bounded smooth Markov-potential objective with actor
blocks `theta_i`.  There is at most one in-flight packet per owner.  Every
owner appears at least once in every `B` application events.  At event `k`,
owner `I_k` returns `(g_hat_k,F_hat_k)` and the server constructs the actor
radius `r_k`, critic radius `c_k`, and stale-operator radius `epsilon_c,k`
from the separately proved reset-trajectory packet interface.

The server minimizes the observable convex quadratic `U_k(alpha,beta)` in
Equation (8) of `coupled_actor_critic_performance_bound.md` over

\[
 0\le\alpha\le\bar\alpha_k,
 \qquad 0\le\beta\le\bar\beta_k\le 1/L_c.              \tag{1}
\]

It then applies the returned stale actor and critic statistics with those two
scales and propagates the declared critic certificate.  This is the algorithm,
not an oracle step: the QP inputs are packet values, stored birth certificates,
public constants, and server-owned version paths.

The first theorem uses

\[
 \mathscr V_k=f(\theta_k)-f_{\inf}+\lambda c_k^2       \tag{2}
\]

and sets optional history curvature to zero.  Strategic and critic staleness
are already paid in `r_k` and `epsilon_c,k`.  This avoids counting the same
delay once as robust bias and again as Krasovskii energy.

## 2. Activated-block progress

On the simultaneous packet-confidence event, define

\[
 C_i=L_i+2\lambda\kappa_i^2,\quad
 R_k=r_k+2\lambda\kappa_i c_k,\quad
 s_k=\min\{\bar\alpha_k,C_i^{-1}\}.                    \tag{3}
\]

The exact QP is no worse than its feasible `beta=0` actor comparator, so

\[
 \mathscr V_k-\mathscr V_{k+1}
 \ge \tfrac12s_k[\|\widehat g_k\|-R_k]_+^2.           \tag{4}
\]

For `s_min=min_{k<K}s_k>0` and
`Delta_V=mathscr V_0-inf_k mathscr V_k`, telescoping yields

\[
 \sum_{k<K}a_k^2\le\frac{2\Delta_V}{s_{\min}},
 \qquad a_k=[\|\widehat g_k\|-R_k]_+.                 \tag{5}
\]

The packet event also gives

\[
 \|\nabla_{I_k}f(\theta_k)\|
 \le a_k+2r_k+2\lambda\kappa_{I_k}c_k.                \tag{6}
\]

Equation (6) explicitly records the price of actor uncertainty and critic
tracking; it does not rename a TD residual as a policy gradient.

## 3. Critic tracking

Assume `epsilon_c,k<=epsilon_c` and each critic action box contains a fixed
`beta_0` with `0<beta_0<=1/L_c`.  Compare the joint QP to the feasible action
`(alpha,beta)=(0,beta_0)`.  When `c_k>epsilon_c/mu_c`, put
`d_k=mu_c c_k-epsilon_c`.  Since `beta_0 d_k<=c_k`,

\[
 \mathscr V_k-\mathscr V_{k+1}
 \ge\lambda\beta_0 c_kd_k
 \ge\lambda\beta_0\mu_c c_k
      [c_k-\epsilon_c/\mu_c]_+.                        \tag{7}
\]

Thus, with `c_floor=epsilon_c/mu_c`,

\[
 \frac1K\sum_{k<K}c_k^2
 \le4c_{\rm floor}^2
 +\frac{2\Delta_V}{\lambda\beta_0\mu_cK}.             \tag{8}
\]

The target motion induced by actor updates is already inside the full QP.
It does not invalidate the critic-only comparator because that comparator sets
`alpha=0` for the one-step comparison.

## 4. Full-policy stationarity

Let `C_motion` be the block cross-smoothness window constant from the existing
activation-coverage lemma, and let

\[
 P_K=\sum_{k<K}\|\theta_{k+1}-\theta_k\|^2.            \tag{9}
\]

If `r_k<=r_bar` and `kappa_i<=kappa_bar`, (8) implies

\[
 \frac1K\sum_{k<K}
 (2r_k+2\lambda\kappa_{I_k}c_k)^2
 \le 8\bar r^2+8\lambda^2\bar\kappa^2
 \left(4c_{\rm floor}^2+
 \frac{2\Delta_V}{\lambda\beta_0\mu_cK}\right)
 =:\overline z_K^2.                                    \tag{10}
\]

Combining (5), (6), and the deterministic `B`-event window inequality gives

\[
 \boxed{
 \frac1{K-B+1}\sum_{k=0}^{K-B}\|\nabla f(\theta_k)\|^2
 \le
 \frac{2B\{4\Delta_V/s_{\min}+2K\overline z_K^2\}
       +2B^2C_{\rm motion}P_K}{K-B+1}.}                \tag{11}
\]

This is a finite-resource event-time theorem.  If
`bar_alpha=Theta(K^-1/2)`, the packet radii and critic floor vanish at the
matching scale, and applied gradients are bounded, (11) gives the usual
`O(B/sqrt(K)+B^2/K)` stationarity rate plus explicit confidence floors.
Every reset trajectory used to make those floors vanish remains charged.

Under the separately stated softmax-interiority and occupancy-mismatch
condition, Equation (9) of `clocked_async_mpg_markov_packet_lemma.md` converts
the gradient criterion to a unilateral Nash-gap bound.  Without those
constants the theorem claims potential stationarity only.

## 5. Comparator and scope

At each realized certificate state the chosen pair has certified drift no
worse than every feasible fixed `(alpha,beta)` pair and no worse than the
online diagonalized rule.  This is pointwise certificate dominance, not a
claim that counterfactual algorithms visit the same future states.  End-to-end
dominance over tuned asynchronous actor--critic baselines remains an
experimental question.

The theorem covers bounded finite-horizon reset trajectories, tabular or
linear SPD critic regression, non-informative packet completion, bounded event
coverage, and Markov potential games.  It does not cover continuing chains,
nonlinear critics, arbitrary general-sum games, or outcome-dependent worker
selection.  Wall-clock conversion is secondary and may reuse the existing
bounded-service event-count lemma; it is not required for (11).

## 6. Honest gate state

The algebraic theorem is closed conditionally, but its proposed practical
high-probability implementation is stopped.  In the preregistered optimistic
audit, only 20.83% of owner cases were jointly nonvacuous within 8,192 charged
transitions and the median minimum was 65,536.  This was with zero variance,
zero staleness and an exact critic, so ordinary sampled packets cannot rescue
it.  See `validation_coupled_actor_critic_certificate_nonvacuity.md`.

Any successor must change the proof/algorithm interface, not the failed gate.
The admissible next route is expectation-level predictable sample splitting;
it cannot inherit the per-event no-harm statement of this theorem.
