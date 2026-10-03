# T-072 dual-use graph architecture calibration

## Purpose and taint status

T-072 tests whether T-071A failed because graph adaptation lacks value or
because its independent probe removes learning transitions. This calibration
reuses every frozen T-071A cell and seed. Its controller architecture and
constants were selected after inspecting a four-seed subset. It is therefore
outcome-informed design evidence, never a pilot, significance test, or paper
result.

## Predictable dual-use block

All ten observations in every block remain learning updates. At a decision
block, the first five updates also produce a fingerprint. Only the first two
fingerprint observations score the seven recipient actions. The proposal is
fixed before the remaining five observations arrive. Those five observations
both update the proposal/local shadow and provide the conditional no-harm
comparison. A failed comparison rolls the model back to the same-data shadow.

The safety state is

\[
Q_{i,k+1}=[Q_{i,k}+\widehat\Delta_{i,k}-\epsilon]_+,
\]

and the recipient selects

\[
a_{i,k}\in\arg\min_a
\widehat L^{\rm pre}_{i,k}(a)
+VQ_{i,k}\lVert w_{i,k}(a)-w^{\rm local}_{i,k}\rVert^2.
\]

The squared displacement is an observable action-dependent upper-bound term
in the quadratic parameter-error Lyapunov drift. Thus accumulated validation
harm suppresses aggressive transfer. Frozen constants are `V=4`, the number
of agents, and rollback margin `-eta/2=-0.02` for gain `eta=0.04`.

## Accounting and comparator

Fingerprint observations are dual-use, not free: they consume the same actor
transitions already charged to learning, and their compressed messages are
charged separately. The controller uses all 240 environment transitions for
learning, zero additional probe transitions, and at most two message units per
decision. The comparator remains the frozen T-070A cellwise best of 2,401
static recipient graphs evaluated in the T-071A common-random-number run.

## Decision rule

The JSON manifest records eight descriptive architecture criteria. Passing
them only authorizes an independent T-072A preregistration with new seeds. It
does not authorize formal, nonlinear, GPU, or HPC4 execution. Failure stops
this architecture without changing its constants.
