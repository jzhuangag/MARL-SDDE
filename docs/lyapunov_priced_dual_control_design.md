# Lyapunov-priced dual control: replacing periodic probes

## The single algorithmic idea

LCO-S0 shows that an optimistic extra-gradient call has two inseparable
effects.  It damps rotational game dynamics now, and its already-paid gradient
pair reveals local geometry for later decisions.  A fixed probe period prices
neither effect correctly: frequent probes consume the stability budget in
potential phases, while sparse probes miss short rotational phases.

The replacement is a causal one-step dual-control rule.  Let \(p_k\) be the
predicted probability of rotational geometry, \(Z_k\) the optimism-resource
debt, \(V\) the Lyapunov tradeoff, and

\[
g(p)=(1-p)g_{\rm pot}+pg_{\rm rot}
\]

the posterior expected reduction in certified log energy from optimism.  The
relative myopic value at belief \(p\) and debt \(z\) is

\[
\phi(p,z)=\min\{0,-g(p)+z/V\}.
\]

If no query is bought, the next belief is only the Markov prediction.  If the
query is bought, its fingerprint produces posterior \(p^+(Y)\), which is then
predicted to the next event.  The two one-step objectives are

\[
J_0=\phi(Tp,z_0),
\]

and

\[
J_1=-g(p)+z/V+\mathbb E_Y[\phi(Tp^+(Y),z_1)],
\]

where \(z_0=[z-\bar u]_+\) and \(z_1=[z+1-\bar u]_+\).  The controller buys the
query iff \(J_1<J_0\).  The next value is a hinge of the scalar posterior.
Posterior calibration gives

\[
\mathbb E[p^+(Y)\mathbf 1_A]=\Pr(H=1,Y\in A),
\]

and the posterior threshold corresponds to a scalar score threshold.  The
expectation therefore has an exact closed form using two Gaussian tail
probabilities.  No QP, belief grid, numerical quadrature, Hessian, or matrix
inverse is needed.  The fingerprint remains \(O(d)\), while belief, debt, and
value-of-information calculations are \(O(1)\).

The information term is not an exploration bonus chosen after seeing data.
For a common next debt,

\[
\phi(Tp,z)-\mathbb E_Y\phi(Tp^+(Y),z)\geq0
\]

by posterior calibration and concavity of the minimum of two affine action
costs.  It is exactly the one-step value of revealing geometry.  Resource debt
then discounts information that cannot justify its optimism cost.

## Research scope and stop rule

This redesign preserves the unified story: heterogeneous asynchronous clocks
create changing potential/rotational geometry; optimism is a scarce
stabilization-and-information resource; Lyapunov debt prices it.  It is not a
new benchmark-specific heuristic and it does not revive the failed periodic
sensor by changing its thresholds.

The current implementation is still a development model because it assumes a
known two-state transition law and Gaussian score emissions.  Before any new
CPU outcome is generated, a development-only runner must be frozen against
the same strong cellwise fixed schedules and exact-phase ceiling.  The old
83,001--83,008 seeds may be used only for architecture development.  Any later
confirmation needs new seeds and, before that, a gain-weighted Markov sensing
error theorem compatible with
`dual_use_sensor_performance_bridge.md`.

No GPU or standard MARL run is authorized by this design note.
