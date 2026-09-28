# Online Lyapunov-bandit participation: replacement design

## Design decision

The coefficient-certificate route is not extended by assuming an unavailable
global sensitivity constant for a nonlinear MAPPO policy.  The replacement
controller aligns observation and action in the same training block.

At block `t`, the controller selects one participation action `q_t`, performs
the corresponding training update, and evaluates a bounded Lyapunov risk on a
disjoint validation stream.  It observes

\[
g_t=V_t^{before}-V_t^{after}\in[-1,1].
\]

Only this selected action is executed and charged.  Thus there is no catalogue
branching cost and no local coefficient is extrapolated over a future policy
trajectory.

## Lyapunov design role

Let `c_m(q)` and `c_e(q)` include both the selected training update and its
validation block.  For registered per-block allowances `b_m,b_e`, define

\[
Q_{j,t+1}=[Q_{j,t}+c_j(q_t)-b_j]_+,\quad j\in\{m,e\}.
\]

The quadratic resource Lyapunov function

\[
L_t=\tfrac12(Q_{m,t}^2+Q_{e,t}^2)
\]

produces the action price `Q_m c_m(q)+Q_e c_e(q)`.  The executable package
contains an adversarial EXP3 interface and a conservative sliding-window UCB
interface.  The latter samples each action once, uses only a registered number
of recent observations, and performs a sparse deterministic re-probe.  A
public scale maps clipped signed progress to `[-1,1]`.  Hence Lyapunov drift
directly changes the online action,
rather than appearing only in an after-the-fact convergence proof.

An additional reserve-aware mask admits an action only if its cost leaves
enough of both physical budgets to execute the cheapest registered action for
every remaining decision.  This yields pathwise budget safety independently of
statistical assumptions.

The catalogue scan is `O(|Q|)` per block.  It requires no Hessian, covariance
matrix, global policy Lipschitz constant, or counterfactual branch rollout.

## Performance bridge

The validation risks telescope exactly along the executed trajectory:

\[
V_T=V_0-\sum_{t=0}^{T-1}g_t.
\]

Therefore maximizing realized cumulative validation drift minimizes final
validation risk on that trajectory.  A separate final evaluation stream is
still required for the manuscript return comparison; it is never fed back to
the controller.

For any predictable bounded gain vector, the importance-weighted selected
gain is unbiased conditional on the past.  Standard exponential-weights
analysis then supplies the bandit learning term, while quadratic queue drift
supplies the resource term.  The remaining proof obligation is a single
primal--dual regret theorem with the hard feasibility mask; it no longer
contains an action-to-post-calibration transfer premise.

## Experiment gate

The next experiment is CPU-only and synthetic.  It must verify adaptation,
dual-budget safety, and comparison with the best fixed catalogue action under
stationary and switching bounded gain sequences.  Only after that theorem and
CPU gate pass will a fresh-seed MPE/MaMuJoCo development run be preregistered.
G0-B remains interface evidence and is not reused as performance data.
