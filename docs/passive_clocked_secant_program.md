# Asynchrony reveals what it rotates: passive clocked secants

## Unified research claim

Asynchronous multi-agent learning creates a randomly clocked coordinate game.
Those coordinate perturbations can destabilize rotational interactions, but
they also provide the excitation needed to observe cross-agent responses.  The
same asynchrony is therefore both the source of instability and the source of
identification.

The proposed mechanism has two clean parts:

1. consecutive mandatory current-gradient evaluations form passive secants;
2. a Lyapunov resource queue spends costly extra-gradient corrections only
   when the secants certify rotational damping value.

No optimism call is purchased for sensing.  This removes the information tax
that caused LCO-S0, LCO-V0, and the perfect paid-sensing upper bound LCO-U0 to
fail.

## Low-complexity observable

Within a stationary local linear phase \(F(x)=Ax\), two ordinary gradient
evaluations give

\[
\Delta g=F(x_k)-F(x_{k-1})=A\Delta x,
\qquad \Delta x=x_k-x_{k-1}.
\]

The scalar alignment

\[
a_k=\frac{\langle\Delta x,\Delta g\rangle}{\|\Delta x\|^2}
\]

is a directional symmetric response.  The orthogonal residual

\[
r_k^2=\frac{\|\Delta g\|^2}{\|\Delta x\|^2}-a_k^2
\]

is a directional rotational response.  Computing \((a_k,r_k)\) costs \(O(d)\)
and constant memory; it forms no Hessian or Jacobian, solves no QP, and buys no
additional rollout or backward pass.

For the exact potential/rotation pair used in the phase theorem,
\((a,r)=(1,0)\) and \((0,1)\), respectively.  More generally one secant is only
directional.  Asynchronous coordinate arrivals provide repeated block
directions; persistent excitation is the condition that turns their weighted
collection into a geometry certificate.

## Lyapunov role

Let the passive observations define a predictable lower confidence bound
\(\underline\Delta_k\) on the log-energy benefit of optimism.  The online
decision remains

\[
u_k=\mathbf 1\{V\underline\Delta_k>Z_k\},
\qquad
Z_{k+1}=[Z_k+u_k-\bar u]_+.
\]

The secant controls *what the action is worth*; the Lyapunov debt controls
*whether that value justifies the remaining resource*.  The previously proved
supermartingale bridge converts gain-weighted decision errors into an anytime
last-iterate energy bound.  A Markov-noise theorem must now control the secant
error and phase-change contamination.

## Execution contract and proof boundary

The primary contract remains centralized training with decentralized
execution.  A central learner has the current joint parameters and the
mandatory current pseudo-gradient used by an asynchronous block update.
Standard MARL implementations that expose only one stale local gradient block
need a separate partial-secants construction; the full-gradient assumption
must not be hidden.

The first CPU audit is restricted to the zero-delay local linear game.  It
must test whether passive current secants recover the exact-phase headroom
under heterogeneous clocks, switching phases, scarce optimism, and perturbed
mandatory gradients.  Passing would authorize a confidence theorem and an
independent CPU confirmation, not a GPU benchmark.  Failure would stop this
mainline rather than trigger another sensing heuristic.
