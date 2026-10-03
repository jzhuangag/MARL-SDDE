# Registered-catalogue Lyapunov participation

## Design object

Let `A_task` be a finite participation catalogue fixed using development data
or system constraints before confirmation.  For a candidate q, a charged
probe leaves

\[
N(q)=\min\left\{
\left\lfloor\frac{B_m-C_m^{\rm probe}}{h+qH}\right\rfloor,
\left\lfloor\frac{B_e-C_e^{\rm probe}}{H}\right\rfloor
\right\}
\]

learning updates.  If the one-step Lyapunov inequality has dependence-aware
noise factor

\[
v(q,\rho)=\rho+\frac{1-\rho}{q},
\]

the finite-budget variance term is proportional to `v(q,rho)/N(q)`.  A
return-free probe produces a simultaneous upper certificate `rho_U`; the
online action is

\[
q^+\in\arg\min_{q\in A_{\rm task}}
\frac{v(q,\rho_U)}{N(q)}.
\]

Thus Lyapunov analysis is a design tool: its contraction--forcing bound gives
the executable score, and the exact physical budgets determine its horizon.
No benchmark evaluation return is observed by the selector.

## Why catalogue registration is not a loophole

The theorem does not assert that the same q grid is useful for every optimizer
and task.  Architecture, optimizer minibatching, and task geometry can rule
out actions whose finite-budget coefficient is poor even when their pure
correlation variance is small.  Those restrictions must be learned only from
development data or supplied by system constraints, then frozen before fresh
confirmation.  Confirmation still compares against the complete fixed-q
catalogue, including actions unavailable to the online selector.

This separation is the standard development/confirmation interface:

1. development fixes `A_task` and every controller constant;
2. the online controller sees only the charged dependence probe;
3. fresh confirmation evaluates return against all fixed baselines.

## Guarantee relative to the registered catalogue

Suppose the correlation certificate satisfies `rho <= rho_U` with probability
at least `1-delta`, and the Lyapunov risk certificate is monotone in rho.  On
that event, selecting q+ minimizes the registered upper risk bound.  If the
probe test chooses between separated low- and high-dependence instances with
directional error probabilities at most delta, the expected certificate obeys

\[
\mathbb E\,\overline R(q_{\widehat j})
\le (1-\delta)\overline R(q_j^*)
   +\delta\max_{q\in A_{\rm task}}\overline R(q),
\]

where `q_j^*` is the certificate minimizer for instance j after charging the
probe.  Adding the explicit probe, delay, and decision-error terms gives the
finite-budget learning-value admission rule already used by the TSP theorem.

The guarantee is intentionally relative to the preregistered task catalogue,
not to an action selected after confirmation.  Complete fixed-q experiments
quantify any remaining catalogue approximation gap.

## Why the V2 progress controller is excluded

The V2 controller optimizes an immediate four- or eight-update validation
difference.  That quantity is not the same as the finite-budget Lyapunov
certificate above, and its repeated sensing cost changes the learning horizon.
TSP-V3 restores the theorem--code interface: one probe estimates dependence,
the Lyapunov score prices the remaining horizon, and the learner then commits.

