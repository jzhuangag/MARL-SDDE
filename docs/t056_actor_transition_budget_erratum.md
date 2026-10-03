# T-056 actor-transition budget erratum

## Status

This erratum corrects the environment-budget unit in T-056. It is a theory
and standalone-certificate correction, not a change to any registered
scientific trajectory.

## Error

The former T-056 horizon treated one synchronized server update as one
environment unit:

\[
\min\{\lfloor(B_m-C_m)/(h+q)\rfloor, B_e-C_e-D\}.
\]

The experiment contract instead defines `B_e` as the number of individual
actor transitions. A round with participation `q` consumes `q` such
transitions. The old expression therefore omitted division by `q` and also
failed to charge `qD` actor transitions for the delay reserve.

## Corrected horizon

After probe costs, the number of affordable synchronized rounds and completed
learning updates are

\[
H_q=\min\left\{
\left\lfloor\frac{B_m-C_m}{h+q}\right\rfloor,
\left\lfloor\frac{B_e-C_e}{q}\right\rfloor
\right\},
\qquad
N_q=[H_q-D]_+.
\]

For the no-probe fixed baseline,

\[
H_{q_0}^{0}=\min\left\{
\left\lfloor\frac{B_m}{h+q_0}\right\rfloor,
\left\lfloor\frac{B_e}{q_0}\right\rfloor
\right\},
\qquad
N_{q_0}^{0}=[H_{q_0}^{0}-D]_+.
\]

An action is feasible only if the probe and the `D` in-flight rounds fit both
budgets. Its charged use is

\[
C_m+(N_q+D)(h+q),\qquad C_e+(N_q+D)q.
\]

## Code and test alignment

`feasible_fixed_horizon` now computes message-limited and
actor-transition-limited round counts before subtracting the delay. The unit
test includes an environment-binding example in which increasing `q` strictly
reduces the available learning horizon and a case in which the `qD` delay
reserve is infeasible.

## Impact boundary

- The nonlinear MinAtar runner already used `remaining_environment // q` and
  charged probe and learning actor transitions explicitly. It is unchanged.
- Existing scientific trajectories, endpoint files, seeds, gates, and reported
  experimental outcomes are unchanged.
- Any future T-056 finite-risk table or corollary must recompute its horizon
  with the corrected expression.
- Historical documents remain provenance; this erratum is the authoritative
  interpretation of the T-056 environment budget.
