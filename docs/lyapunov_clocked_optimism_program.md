# When asynchrony rotates the game: Lyapunov-clocked optimism

## The single story

Asynchronous MARL does more than delay gradients.  Agent-by-agent completions
turn the joint policy update into a randomly clocked coordinate game.  In a
locally potential/cooperative region, the cheapest plain coordinate gradient
is contractive and an extra-gradient correction slows it down.  In a locally
rotational/competitive region, the same plain update injects energy even with
an exact oracle, while an optimistic/extra-gradient update dissipates it.

The proposed problem is therefore:

> Learn at the speed of heterogeneous arrivals while spending a limited
> optimism budget only when the current game geometry needs rotational
> damping, and guarantee last-iterate stability rather than merely average
> regret.

The controller has one purpose: minimize a Lyapunov drift certificate of the
clocked game.  It jointly controls the step applied to an arriving policy block
and whether that update buys an arrival-fresh optimistic anchor.  A fresh
anchor temporarily freezes parameter commits, evaluates the lookahead oracle
under the current joint policy, applies the owner block, and fully charges the
additional actor interaction and serialized learner time.  Virtual
clock/resource debts prevent fast agents or optimism calls from silently
consuming the entire wall-clock/compute budget.  Decentralized execution is
unchanged; this is a centralized-training mechanism.

## Exact asynchronous phase boundary

Consider the two-player bilinear local game with pseudo-gradient

\[
F(x)=\omega Jx,\qquad
J=\begin{bmatrix}0&1\\-1&0\end{bmatrix},
\]

and let one uniformly selected agent block update at each event.  Write
\(s=\eta\omega\in(0,1)\).  A plain coordinate gradient update has exact
conditional mean-square multiplier

\[
q_{\rm G}(s)=1+\frac{s^2}{2}>1.
\]

The corresponding coordinate extra-gradient update has multiplier

\[
q_{\rm EG}(s)=1-\frac{s^2}{2}+\frac{s^4}{2}<1.
\]

If an optimistic update is used independently with probability \(p\), the
exact multiplier is

\[
q(s,p)=1+\frac{s^2}{2}-p\left(s^2-\frac{s^4}{2}\right).
\]

Hence strict mean-square contraction is possible exactly when

\[
p>p_{\min}(s)=\frac{1}{2-s^2}.
\]

This is a phase boundary and a resource lower bound: below this optimism
frequency, no scheduling arrangement independent of the current state can
stabilize the rotational subclass in mean square.

This boundary survives heterogeneous asynchronous clocks.  If agent one is
selected with any probability \(r\in(0,1)\), use the diagonal Lyapunov metric

\[
P_r=\operatorname{diag}\!\left(\frac{1-r}{r},1\right).
\]

For the randomized plain/optimistic update, direct matrix multiplication gives

\[
\mathbb E[M_k^\top P_rM_k-P_r]
=(1-r)s^2\{1-p(2-s^2)\}I.
\]

Thus clock heterogeneity changes the conditioning and noise sensitivity of the
energy, but not the exact optimism-frequency phase boundary.  In particular,
the rare agent's coordinate receives weight proportional to its inverse
arrival probability.  This is the first concrete role of *clock debt*: an
online rate estimate or debt process must track this metric when arrival rates
are unknown or time-varying; pretending the Euclidean metric is valid produces
a false instability conclusion.

Now consider a local quadratic potential with normalized curvature
\(s=\eta\mu\in(0,1)\).  The uniform-coordinate factors are

\[
q_{\rm G}^{\rm pot}(s)=\frac{1+(1-s)^2}{2},\qquad
q_{\rm EG}^{\rm pot}(s)=\frac{1+(1-s+s^2)^2}{2},
\]

and
\(q_{\rm G}^{\rm pot}<q_{\rm EG}^{\rm pot}<1\).  Thus always paying for
optimism is strictly slower in the potential subclass.  These are the two
opposing factors; neither was introduced by a selected benchmark.

## Lyapunov controller

Let \(E_k\) be the certified lifted-state energy of the delayed asynchronous
game and let

\[
Z_{k+1}=[Z_k+u_k-\bar u]_+
\]

be optimism-resource debt.  For a candidate plain/optimistic pair with
certified energy multipliers \(q_{0,k}\) and \(q_{1,k}\), one drift-plus-cost
decision is

\[
u_k=1
\quad\Longleftrightarrow\quad
V E_k(q_{0,k}-q_{1,k})>Z_k,
\]

subject to the hard remaining oracle/compute budget.  In a potential phase the
left side is negative, so optimism is never purchased.  In a rotational phase
it is positive and the queue admits calls until their marginal stability value
is priced by debt.

### Delay changes the executable contract

A nominal extra-gradient computed entirely at the same stale state is not a
delay correction.  A lifted second-moment audit gives spectral radius greater
than one for the pure bilinear subclass at delays 1, 2, and 4 even when every
update uses that stale extra-gradient.  The proposed algorithm therefore must
use a genuinely arrival-fresh anchor on \(u_k=1\); otherwise the delay claim is
false.

For delay \(D\), define
\(z_k=(x_k,x_{k-1},\ldots,x_{k-D})\) and let \(M_{i,0,D}\) be the lifted
stale-coordinate transition, while \(M_{i,1,D}\) uses the current-state
coordinate extra-gradient.  Under iid arrival probabilities \(r_i\), the exact
second-moment operator at fixed anchor probability \(p\) is

\[
\mathcal K_D(p)=
\sum_i r_i\{(1-p)M_{i,0,D}^{\otimes2}
+pM_{i,1,D}^{\otimes2}\}.
\]

Mean-square stability holds exactly when
\(\rho(\mathcal K_D(p))<1\).  At \(D=0\) this reduces to
\(p>1/(2-s^2)\).  Numerical theorem checks show the minimum fresh-anchor
frequency rises substantially with delay; the analytic upper/lower bound for
general \(D\) remains a proof obligation, not a completed theorem.

For delayed dynamics the intended Lyapunov--Krasovskii energy is

\[
\mathcal L_k=
\|x_k-x^\star\|_P^2
+\sum_{\ell=1}^{D}c_\ell\|x_{k+1-\ell}-x_{k-\ell}\|^2
+\frac{1}{2}Z_k^2
+\frac{1}{2}\sum_i Q_{i,k}^2,
\]

where \(Q_i\) is effective-clock debt for agent \(i\).  The history term pays
for delayed rotational feedback; the clock debts price heterogeneous update
rates; the resource debt prices optimism.  This is the natural place for
Lyapunov drift.  An SDDE is useful only as a continuous-time interpretation of
the delayed randomly clocked dynamics; the discrete finite-time theorem must
remain primary.

## Required theorem, not yet a claim

The candidate becomes viable only if one proof covers the same executable
algorithm:

1. predictable estimates or confidence bounds for the local symmetric and
   skew energy terms from Markov policy-gradient data;
2. a one-step lifted Lyapunov drift inequality under block arrivals, bounded
   delay, Markov noise, and stochastic optimism calls;
3. an impossibility/lower bound below the phase boundary;
4. queue stability and finite-time last-iterate VI/Nash-gap convergence above
   the boundary, with explicit wall-clock and oracle costs;
5. a no-waste result in potential regions and a clock-imbalance term that
   vanishes under the debt controller;
6. an implementation whose fresh-anchor extra-gradient cost is fully charged and compared
   with equal-cost extra batching, always optimism, fixed optimism frequency,
   optimistic mirror descent/extragradient, and strong asynchronous MARL
   baselines.

## Novelty boundary and experimental path

Extra-gradient, optimism, asynchronous learning, delay-adaptive online
optimization, and Lyapunov queues all have prior art.  A defensible paper claim
would have to be their specific Markov-game result: a certified phase-dependent
optimism budget under heterogeneous asynchronous policy clocks, with a
matching lower boundary and last-iterate/wall-clock guarantee.

The next gate is CPU-only.  It must test the exact phase theorem under
stationary and switching symmetric/skew regimes, heterogeneous arrivals,
delay, and correlated oracle noise against strong fixed policies.  Only after
the theorem and CPU headroom both pass may the project design a standard
competitive/cooperative MARL benchmark.  No GPU is currently authorized.
