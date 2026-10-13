# Observable forecast, simultaneous risk intervals, and selection

## Model and exact discrete risk

For a fixed policy, actor i has reward R[i,t] = theta + C[t] + E[i,t].
The common and private processes are independent stationary Gaussian AR(1)
chains with public persistence lambda, common variance sigma_c^2, and private
variance sigma_e^2. Each marginal has the same law across coupling regimes.
The controlled TD(0) update with constant feature and discount zero is

e[k+1] = e[k] - eta e[k-D] - eta z[k],

where z is the q-actor mean noise retained every b physical transitions.
Its variance is v_q = sigma_c^2 + sigma_e^2/q and persistence lambda^b.
Initial history errors are all e[0]. Calibration and learning streams are
independent. Actions remain fixed after calibration in this first qualification.

Lift the delayed error history and current noise into a finite-dimensional
linear state. Its transition T(a) has first row `(1,0,...,-eta,-eta)` (the
first two entries combine when D=0), history-shift rows, and last diagonal
entry lambda^b. With stationary unit noise initially, the exact conditional
terminal risk is

R(a) = B(a) (w0-theta)^2 + F(a) v_q,

where B(a) and F(a) are nonnegative deterministic coefficients obtained by
propagating the mean and covariance. For stable T, the code evaluates
Sigma_N = Sigma_inf + T^N (Sigma_0-Sigma_inf) (T^N)^T.
This is an exact discrete calculation; no SDDE approximation is used.

## Observable calibration and coverage theorem

Observe n+1 reward vectors from Q actors. Whiten adjacent observations using
public lambda. The resulting n innovation vectors are independent Gaussian
with mean theta sqrt((1-lambda)/(1+lambda)). Their spatial averages have
variance v_Q. Their spatial contrasts have variance sigma_e^2.

Construct a Student-t interval for theta, a chi-square interval for v_Q with
n-1 degrees of freedom, and a chi-square interval for sigma_e^2 with
n(Q-1) degrees of freedom. Give each two-sided interval error probability
delta/3. Bonferroni gives joint coverage at least 1-delta. No independence
between the three interval events is needed. The sigma_e^2=0 boundary is
handled as a degenerate Gaussian contrast; its interval is [0,0].

For every q <= Q,

v_q = v_Q + (1/q - 1/Q) sigma_e^2.

All coefficients are nonnegative, so the same joint event supplies lower and
upper variance bounds simultaneously for every q. Squaring the target
interval's distance to w0 supplies a lower and upper initial-risk bound.
Substitution into B(a) initial_risk + F(a) v_q therefore gives

L(a) <= R(a) <= U(a)

simultaneously for every stable candidate, every finite usable horizon, and
every data-dependent selection from the catalogue. There is no per-action
normal approximation or multiple-use sample-count inflation.

## Selection and real-risk comparison

Let a_hat minimize the central fitted risk over the complete stable catalogue.
Let a_star minimize the true conditional risk over the same paid catalogue.
On the coverage event,

R(a_hat)-R(a_star) <= U(a_hat)-min_a L(a).

If central errors are uniformly at most epsilon, the familiar consequence is
R(a_hat)-R(a_star) <= 2 epsilon. This is a true-risk statement because both
lower and upper envelopes are available.

A baseline-relative safety declaration requires

U(proposed, full costs) <= L(baseline, full costs) + epsilon_safe.

Comparing two upper envelopes alone does not establish a true-risk difference.
The present CPU study measures the selection bound and its width; it does not
assert universal baseline no-harm when this admission inequality is absent.

## Lyapunov's role

For a stable T, the discrete Lyapunov equation defines the noise covariance
and the finite-horizon quadratic risk. The algorithm uses this risk to select
participation, spacing, and step. Thus the same recursion provides the
prediction, action objective, and exact validation target.

## Computational and transfer scope

For scalar TD the lifted matrix dimension is D+2. Candidate coefficients can
be cached by public costs, persistence, and delay; subsequent scoring is linear
in the number of candidates. The step catalogue is explicit and finite; the
implementation claims its exact minimizer, not continuous global optimality.
Known mixing is an assumption. Whitening with an estimated lambda needs an
additional simultaneous confidence/transfer argument.

The theorem does not cover a nonlinear moving-policy PPO optimizer. Before a
MARL return claim, an actual critic/optimizer response model and its calibration
to later training states must be independently qualified. Neither this theorem
nor a passed numerical gate supplies that missing transfer automatically.
