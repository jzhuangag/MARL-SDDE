# One substantive correction: amortized packet-response value control

Status: mathematical/code prototype after a PASSED native operational loop.
No qualification return was ranked; no new positive learning result is claimed.
The previous frozen current-packet qualification is preserved unchanged.

## Decision and causal motivation

The native 6+6 qualification proves genuine five-epoch server-owned PPO,
current-bank selection, independent delayed-program service, exact accounting
and semantic replay. Its cost-only report also exposes avoidable duplication:
current_value executes exactly three PPO blocks per ready actor versus one in
eligible_all. This is a structural count, not a return-based hyperparameter
selection. The one correction is to amortize the candidate family both in
training and statistical representation, while preserving actual-return scores
and explicitly retaining cross-agent interactions.

1. Train one unit PPO proposal per ready actor using the qualified server-owned
   five-epoch primitive. Form theta_i(alpha)=theta_i+alpha*(theta_i(unit)-theta_i)
   for all predeclared coefficients. This is a post-PPO parameter line search,
   NOT an assertion that it equals PPO trained at learning rate alpha. Positive
   steps retain the unit server proposal's optimizer moments; idle retains the
   old server moments. All copies/interpolations remain paid and reported.
2. Score the joint candidates with the same exact unclipped likelihood-ratio
   actual-return observation. Derive an alternative uncertainty charge from
   unit single-actor responses plus an observed interaction/nonlinearity
   remainder. Do not assume separability, independent actors or linear returns.
3. Send early private probe records only for ready actors whose policies could
   change. Unchanged actors have likelihood ratio exactly one. Public team
   states/rewards are already paid; all future full-program bytes stay paid,
   including the repeated probe portion. No unjustified compression credit.

The helper implementation is `experiments/amortized_packet_value.py`. It is
not yet the full native controller and must not be described as having passed
new native learning/performance development.

## Exact representation and expected selection bound

Before opening the independent bank, freeze K candidate policies, coefficient
vectors a_k in [0,A]^d, and d eligible unit single-actor basis columns. Let the
unclipped actual-return contrast on episode e be X_e(k), with conditional mean
mu_k. Let Z_e(i) be the same episode's unit single-actor contrast. Freeze an
optional offset h_k using only past information (default zero). DEFINE

    R_e(k) = X_e(k) - sum_i a_(k,i) Z_e(i) - h_k.

This is an identity for arbitrary interactions and arbitrary candidate policies;
it is not a linearization assumption. Define S_Z=trace Cov(Z_e|H) and
r=E[max_k |R_e(k)| | H]. For the empirical priced argmax chosen after m episodes,
known predictable costs cancel from score error and

    E[U(k*)-U(k_hat)|H,j]
      <= 2 V E max_k |Xbar(k)-mu_k|
      <= 2 V A sum_i sqrt(Var(Z_i)/m) + 4 V r
      <= V lambda_j + V d A^2 S_Z/(lambda_j m) + 4 V r.       (G)

Proof: decompose centered score errors using the exact identity. Bound the
linear part by A times the sum of coordinate errors and use conditional
independence/Jensen. For the residual, E max|Rbar-ER| is at most
E max|Rbar|+max|ER| <= 2 E max|R_e|. Cauchy-Schwarz and Young give (G).
Offsets cancel from centered errors; using current samples to fit h or choose
the basis invalidates this fixed-representation argument.

The full-vector bound V lambda + V S_X/(lambda m) remains available, where
S_X=trace Cov(X_e|H). Neither bound universally dominates: the response-basis
bound can help for nearly additive/weakly nonlinear responses, and pays an
explicit remainder in strongly interacting games. Never erase that remainder
or claim a d-dimensional game solely from a low-rank-looking sample matrix.

For a parameter line-search family theta(a)=theta+D a, columns of D occupy
distinct actor blocks, so ||D||_op=max_i ||d_i||. If the per-episode importance
contrast is twice differentiable on this compact family and its parameter
Hessian norm is bounded there by an integrable L_e, Taylor's theorem gives

    max_a |R_e(a)| <= L_e max_i||d_i||^2 * d*(A^2+A)/2      (h=0, A>=1).

Proof: expand F_e(a) around zero and each unit response F_e(e_i) around zero;
their first-order terms cancel. Bound the remaining quadratic term by
||a||_2^2 and the weighted unit remainders by ||a||_1, then use the block-
orthogonal D norm. This interprets the remainder as second order under an
EXPLICIT local regularity/moment assumption. The algorithm does not evaluate
Hessians or require a known numerical L_e: it computes the exact residual from
already scored episode contrasts. It must still pay that residual, and cannot
assume small interactions in a dense game. Smoothness here is of policy
likelihood on a FIXED trajectory, not differentiability of the simulator.

## Observable risk feedback and Lyapunov design

Let the pre-bank meta option j=(m,lambda,g) include geometry g in
{full_vector,response_basis}. Current action selection still maximizes the
FULL actual-return sample score minus resource prices. Compute both families'
statistics from the same bank, without additional candidate neural forwards:

    nu_hat_full = trace(sample covariance of X),       rho_hat_full=0,
    nu_hat_basis = d A^2 trace(sample covariance of Z),
    rho_hat_basis = 4 mean_e max_k |R_e(k)|.

For every fixed option, these are conditionally unbiased for its corresponding
nu and rho, regardless of the predictably chosen bank size m>=2. Thus

    ell_hat_j = Q_C c_probe_j + Q_F f_probe_j + V lambda_j
                 + V nu_hat_g/(lambda_j m_j) + V rho_hat_g

is unbiased all-option probe/certificate feedback. Its predictable hint omits
the last two stochastic terms. Optimistic mirror descent yields the same
finite-time probe regret with residual term

    D^2/(2 eta) + eta/2 sum_t E ||ell_hat_t-hint_t||_2^2.

The earlier selection/drift derivation then holds with the selected geometry's
explicit variance and remainder terms. All full-source/program/proposal and
probe costs remain in the two total-charge queues. A finite fourth moment is
sufficient for finite squared feedback, not automatically a known uniform
numerical bound in a deep MARL task. Selecting the minimum empirical penalty
AFTER observing this bank is not a substitute for predictable meta selection.

This supplies an observable EXPECTED-risk interface, not a realized confidence
certificate. Practical precision/usefulness remains an independent development
question; nonzero selected estimates do not establish positive true gain.
Finite m retains an error floor, and the interaction remainder need not shrink
with m. No HAPPO trajectory dominance, stationarity or universal no-harm follows.

## Actual own-trajectory return and the composite Lyapunov function

For a fixed game and initial-state distribution, beta_t is the current joint
policy theta_t and the committed candidate is theta_(t+1). Consequently the
true signed gain used above satisfies the EXACT identity

    sum_t mu_(t,a_t) = [J_H(theta_T)-J_H(theta_0)]/R.

Thus this is not merely virtual-debt analysis: with a finite uniform return
upper bound J_max the nonnegative composite potential is

    W_t = V [J_max-J_H(theta_t)]/R + (Q_C,t^2+Q_F,t^2)/2.

The unknown constant J_max cancels from drift and need not be supplied to the
controller. The priced empirical actual-return argmax minimizes an estimated
drift of this SAME potential; selection error is paid by the chosen geometry's
expected risk, and predictable meta control prices the cost of estimating it.

For zero initial resource queues and the earlier queue/menu inequality with a conditionally pointwise-feasible
eligible comparator u_t and fixed feasible meta distribution v, the queue-
weighted comparator terms are nonpositive. Writing G_T=sum_t E mu_t(u_t),
telescoping gives the finite-time ACTUAL final-return lower bound

    E J_H(theta_T) >= E J_H(theta_0) + R G_T
       - R*(TB+R_meta)/V
       - R sum_t E_v[lambda_j + nu_(t,g_j)/(lambda_j m_j) + rho_(t,g_j)].

Dropping the terminal nonnegative queue energy only weakens this bound. With
average feasibility alone the queue-weighted terms cannot be silently dropped;
retain them explicitly. Idle is a valid gain reference, but the nonnegative
error terms do not guarantee return improvement. The eligible comparator is
evaluated at OUR visited policies, not along another HAPPO learning trajectory.
This theorem therefore does not assert that our final policy beats HAPPO.

If the game's objective changes with t, add the exact objective-motion sum
sum_t [J_(t+1)(theta_(t+1))-J_t(theta_(t+1))] when telescoping. A moving
environment cannot disappear from this proof. The qualified native tasks have
a fixed environment kernel/objective; their policy process still changes.
Equivalently, final objective change equals R times the summed signed update
gains PLUS that signed motion sum; a worst-case statement subtracts its absolute
value bound. No zero-drift assumption is silently introduced for moving games.

## Computational guarantee and prospective next stage

For any fixed ready-packet workload, executed actor backprop/Adam calls fall
from |alphas|*E to E per ready actor, exactly a factor of three for the current
catalogue. Parameter copies, line interpolation, candidate scoring, interaction
statistics, private data and all other costs remain charged. This guarantees a
primitive-level saving, NOT a threefold full-system speedup or a return gain.
Candidate policies also change, so old qualification trajectories cannot be
relabelled as a replay or predicted performance of this design.

The eligible-only probe uplink likewise saves unnecessary private records
without dropping any changing actor from the joint likelihood ratio. It does
not provide free future data or alter source-policy/critic version lineage.

Bounded design judgment: GO to integrate this ONE amortized complete-loop
correction; NO_GO for an immediate broad performance sweep or confirmation.
Keep the fresh sequential HAPPO anchor and matched eligible asynchronous controls.
Before new native data, freeze the complete new controller, geometry options,
exact new counters, independent fresh seeds and stop rules. If the native
interface and expected-risk feedback pass, the next small registered development
must jointly test useful selection precision, learning response and full costs;
it cannot admit a method by variance reduction alone. Qualification returns
remain outcome-ineligible. No additional redesign series or parameter tuning
on those returns is authorized by this judgment.
