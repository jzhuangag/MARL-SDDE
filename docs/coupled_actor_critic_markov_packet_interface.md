# Markov packet interface for the coupled actor--critic controller

Date: 2026-09-05.

Status: **finite-schedule, reset-trajectory interface proved under declared
bounded linear/tabular assumptions.**  It closes the packet-to-certificate
step; it does not by itself close the full-game performance theorem.

## 1. Executable packet

At birth event `b`, a worker snapshots the joint policy `theta_b`, the
centralized critic `w_b`, and their version numbers.  It runs `m` independent
reset trajectories of exactly `H` transitions.  Every transition is charged.
The worker returns two vector averages:

- an owner actor statistic `g_hat_b`, evaluated at `(theta_b,w_b)`;
- a centralized critic operator statistic `F_hat_b`, evaluated at the same
  snapshot.

One trajectory may contribute to both averages.  Actor and critic statistics
within a trajectory need not be independent.  The only independence used is
between reset replicas conditional on the birth filtration.  Completion time
may depend on public workload and actor identity, but not on realized rewards,
states, or gradient innovations.  The server stores `theta_b`, `w_b`, and the
birth critic radius `c_b`; it already knows the complete intervening version
path when the packet arrives.

## 2. Simultaneous statistical event

Suppose each trajectory-level actor vector has Euclidean norm at most `C_a`
and dimension `d_a`; analogously use `C_c,d_c` for the critic vector.  For a
declared schedule of at most `K` packets and total coordinate count
`d=d_a+d_c`, coordinate Hoeffding plus a finite union bound gives

\[
 \varepsilon_x(m,\delta)
 =\sqrt{d_x}\,C_x
   \sqrt{\frac{2\log(2Kd/\delta)}{m}},
 \qquad x\in\{a,c\}.                                  \tag{1}
\]

With probability at least `1-delta`, (1) holds simultaneously for actor and
critic averages at every scheduled application event.  Adaptively chosen
birth policies do not invalidate the statement: apply conditional Hoeffding
at each birth filtration and then take the finite union bound.  No transition
inside a Markov trajectory is declared iid.  This deliberately conservative
finite-schedule radius is the theorem interface; an empirical standard error
cannot silently replace it.

For a nonvacuous observable alternative, let `S_j^2` be the unbiased sample
variance of coordinate `j` across the reset replicas and let that coordinate
lie in `[-C_j,C_j]`.  Applying the empirical Bernstein inequality to both
signs and union-bounding over the same schedule gives coordinate radii

\[
 r_j=\sqrt{\frac{2S_j^2\log(4Kd/\delta)}{m}}
 +\frac{14C_j\log(4Kd/\delta)}{3(m-1)},                \tag{1b}
\]

and vector radius `sqrt(sum_j r_j^2)`.  This is data-dependent but still a
certificate; unlike a raw standard error, it retains the bounded-range term.
The scalar inequality is from Maurer and Pontil, *Empirical Bernstein Bounds
and Sample Variance Penalization*, COLT 2009.  Its use here is only across
independent reset replicas, not across Markov transitions.

## 3. Arrival-time actor radius

Assume the population actor statistic satisfies

\[
 \|g_i(\theta,w)-\nabla_i f(\theta)\|
 \le \beta_{i,H}+B_i\|w-w^\star(\theta)\|,             \tag{2}
\]

where `beta_i,H` is finite-horizon/truncation bias.  For a tabular
finite-horizon TD-advantage actor, (2) follows by inserting the value error in
the TD residual; bounded scores give an explicit `B_i`.  More general critics
must prove their own analogue of (2).

If `L_theta,i` is a joint-policy Lipschitz constant for the owner gradient,
the server constructs

\[
 r_k=\varepsilon_a+\beta_{i,H}+B_i c_b
       +L_{\theta,i}\|\theta_k-\theta_b\|.             \tag{3}
\]

Thus `||g_hat_b-nabla_i f(theta_k)|| <= r_k` on the simultaneous event.  The
last displacement is computed from versioned parameters, not inferred from
rewards.  A bounded event delay `D` and applied-update envelope `a_max G`
also give the coarser public bound

\[
 \|\theta_k-\theta_b\|\le D\,a_{\max}G.               \tag{4}
\]

## 4. Arrival-time critic radius

Let the current population critic operator be

\[
 F_\theta(w)=A_\theta(w-w^\star(\theta)),
 \qquad \mu_c I\preceq A_\theta\preceq L_c I.         \tag{5}
\]

For example, a projected tabular finite-horizon Monte-Carlo value regression
has diagonal `A_theta`; a positive occupancy lower bound supplies `mu_c`.
Assume the population operator is Lipschitz in policy and critic argument with
constants `L_F,theta` and `L_F,w`.  The stale returned statistic then obeys

\[
 \|\widehat F_b-F_{\theta_k}(w_k)\|
 \le \varepsilon_c
 +L_{F,\theta}\|\theta_k-\theta_b\|
 +L_{F,w}\|w_k-w_b\|=:\epsilon_{c,k}.                 \tag{6}
\]

For `0<=beta<=1/L_c`, spectral calculus gives

\[
 \|e_k-\beta\widehat F_b\|
 \le(1-\mu_c\beta)c_k+\beta\epsilon_{c,k}.            \tag{7}
\]

The `1/L_c` restriction is essential.  A restriction based only on
`1/mu_c` is not a valid SPD contraction interval when the critic is
ill-conditioned.  If the actor step moves the critic target by at most
`kappa_i alpha ||g_hat_b||`, the certified next radius is

\[
 c_{k+1}=(1-\mu_c\beta_k)c_k+\beta_k\epsilon_{c,k}
          +\kappa_i\alpha_k\|\widehat g_b\|.           \tag{8}
\]

Equations (3), (6), and (8) are exactly the observable inputs used by the
two-dimensional Lyapunov QP.

## 5. No double counting of staleness

The first executable theorem sets the optional history curvature `h_i=0`.
All policy and critic staleness is paid once through (3) and (6).  It therefore
does **not** add a Lyapunov--Krasovskii history penalty on top of the same
version displacement.  A later expectation-level refinement may trade the
robust version radius for a history energy, but only after proving a new drift
and replacing—not duplicating—the corresponding term.

This choice keeps SDDE/Lyapunov--Krasovskii analysis optional.  The main design
use of Lyapunov remains substantive: the composite actor-risk and critic-target
certificate is minimized online to choose both actor and critic scales.

## 6. Remaining theorem obligations

The interface above closes the finite-schedule concentration and stale linear
critic contraction algebra.  Before a sampled learning experiment is
authorized, the same finite Markov potential game must instantiate and check:

1. explicit `C_a,C_c,B_i,L_theta,L_F,theta,L_F,w,kappa_i,mu_c,L_c`;
2. nonvacuity of (1)--(3) at a fully charged trajectory count;
3. activated-owner coverage and conversion to the declared game metric;
4. a comparison theorem against a strong fixed-timescale asynchronous
   actor--critic baseline.

Continuing-chain sampling, nonlinear critics, general-sum equilibrium claims,
and outcome-dependent completion remain outside this theorem.
