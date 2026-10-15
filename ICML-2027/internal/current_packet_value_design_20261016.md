# Current-packet value control: substantive successor design

Status: executable controller prototype and conditional mathematical interface;
not a registered native experiment, performance result, or ICML-ready algorithm.
The completed temporal-recycling study remains NO_GO with every original gate.

## 1. Decision, not a renamed old controller

The failed controller selects an action pattern before seeing the current bank.
Fresh actual-return estimates can update only the next action distribution, even
though the next packets, policies and interaction context have changed. Its
one-hot initial full-update pattern and old-bank hint are not current-packet
value identification. The successor makes a different causal decision:

1. Receive packets and freeze the current behavior policy, server-owned Adam
   candidate states, unique full-policy candidate hashes, predictable costs and
   an old-data Monte Carlo potential. Candidate zero is the unchanged policy.
2. Select a probe option j=(m_j, lambda_j) before opening fresh episodes.
3. Collect m_j independent behavior-policy episodes. This bank is training
   data, not benchmark evaluation. The behavior policy does not move mid-bank.
4. Select the CURRENT receipt action by the CURRENT bank's priced score.
5. Commit exactly the frozen selected policy and server-owned optimizer state.
6. Reconcile all charges. Generate future packets only from their captured
   source policy; the current bank may then become future learning data.

Unlike the failed algorithm, the selected score is deliberately NOT declared
unbiased. Selection regret is explicitly charged in the design below. The
probe catalogue, scales and costs are constructor inputs, not retrospectively
fitted constants. The implementation is
`experiments/current_packet_value_control.py`; no frozen source was edited.

## 2. Actual-return observation and filtration

Let H_t denote all information before the probe option is sampled. The finite
candidate policies pi_(t,a), a in A_t, their costs, the behavior policy beta_t
and potential phi_(t,h) are H_t-measurable. The candidate set is COMMON to all
probe options and contains idle action 0, whose policy is beta_t. Deduplicate
identical full-policy candidates before computing vector variance.

Define finite-episode team return J_H(pi)=E_pi[sum_(h=0)^(H-1) r_h] for a common
initial-state distribution and team reward definition. For independent fresh
episodes e drawn from beta_t, use unclipped prefix joint-policy ratios

    rho_(e,h,a) = product_(l=0)^h product_i
                 pi_(t,a,i)(u_(e,l,i)|o_(e,l,i)) /
                 beta_(t,i)(u_(e,l,i)|o_(e,l,i)),
    delta_(e,h) = r_(e,h) + phi_(t,h+1)(s_(e,h+1)) - phi_(t,h)(s_(e,h)),
    X_(e,a) = sum_h (rho_(e,h,a)-1) delta_(e,h) / R.

Here R>0 is a fixed registered scale, phi_(t,H)=0, and the reference contrast
X_(e,0) is exactly zero. Under absolute continuity and integrability, conditional
change of measure and telescoping give

    mu_(t,a) = E[X_(e,a)|H_t]
             = (J_H(pi_(t,a))-J_H(beta_t))/R.

The potential need not equal a value function. It must be frozen independently
of the current scored bank. Reusing the current bank afterward for future
packets does not invalidate this conditional identity. Fitting the potential
or choosing m after viewing this bank would invalidate the stated interface.
Clipped/self-normalized weights or PPO surrogate losses are NOT interchangeable
with X. Positive categorical support alone does not supply uniform moment
bounds. For continuing Markov samples, conditional independence is unavailable:
a separately established bias/covariance correction is required. This design
currently uses independent episodes, not a general Markov-mixing theorem.

## 3. A useful current-context comparator, with selection bias paid

Let c_(t,a) and d_(t,a) be predictable incremental communication and computation
costs. Let C0_t and F0_t be already committed common costs, including source
work, potential fitting, all materialized proposals, reserved eventual incoming
messages and any otherwise hidden data path. Let f_(t,j) be the additional
predictable compute cost of probe option j. All costs use fixed registered
units; reported ledgers must also retain raw native transitions, backward and
forward rows, optimizer calls, bytes and messages.

With nonnegative virtual queues Q_C,t and Q_F,t and V>0, define

    U_t(a) = V mu_(t,a) - Q_C,t c_(t,a) - Q_F,t d_(t,a).

After the bank, choose a_t maximizing the same expression with the sample mean
mu_hat. This is a finite argmax, not an unsupported global return optimum. If
S_t=trace Cov(X_e|H_t), then for any predictable option j,

    E[max_a U_t(a)-U_t(a_t) | H_t,j]
       <= 2 V E[||mu_hat-mu||_infinity | H_t,j]
       <= 2 V sqrt(S_t/m_j)
       <= V lambda_j + V S_t/(lambda_j m_j).                 (1)

Proof: add/subtract empirical utilities at a maximizing true action and the
empirical maximizer. Known costs cancel from estimation errors. Bound both
coordinate errors by the infinity norm, then the Euclidean norm. Conditional
independence gives E||mu_hat-mu||_2^2=S_t/m_j; Jensen and Young finish the bound.
The result also bounds the gap to any H_t-measurable action mixture, not just a
fixed historical action pattern. It is an expectation bound, not a positive
high-probability gain certificate or per-step no-harm statement.

## 4. Probe and queue design derived from (1)

Use the finite probe loss

    ell_(t,j) = Q_F,t f_(t,j) + V lambda_j
                + V S_t/(lambda_j m_j).

The full-vector sample variance trace S_hat from the chosen m_j>=2 episodes is
unbiased for S_t conditional on H_t and j. Therefore all-option feedback

    ell_hat_(t,k) = Q_F,t f_(t,k) + V lambda_k
                    + V S_hat/(lambda_k m_k)

is conditionally unbiased for ell_(t,k), even though the receipt action uses
the same data. This fact does NOT make the selected action score unbiased.

For z_t in the probability simplex, eta>0, h_(t,k)=Q_F,t f_(t,k)+V lambda_k,
set p_t=Proj(z_t-eta h_t), sample j_t~p_t before data, then update
z_(t+1)=Proj(z_t-eta ell_hat_t). For every fixed probe distribution v, Euclidean
optimistic mirror descent gives expected probe regret at most

    R_meta = D^2/(2 eta)
       + eta V^2/2 sum_t E[S_hat_t^2] sum_k 1/(lambda_k^2 m_k^2),          (2)

where simplex diameter D^2<=2. To verify the identity, apply the projection
three-point inequalities at z_t, p_t and z_(t+1), telescope the distance to v,
and bound the residual inner product by eta/2 ||ell_hat_t-h_t||_2^2. Predictable
h_t may be biased as a forecast; the displayed feedback must retain its stated
conditional mean. A finite fourth moment is sufficient for finite R_meta.
The executable prototype implements these exact two projections.

Queue updates use TOTAL charges, never only accepted updates:

    Q_C,t+1 = [Q_C,t + C0_t + c_(t,a_t) - b_C]_+,
    Q_F,t+1 = [Q_F,t + F0_t + d_(t,a_t) + f_(t,j_t) - b_F]_+.

For L_t=(Q_C,t^2+Q_F,t^2)/2 assume the conditional squared cost increments have
half-sum at most B. Combining queue drift, (1) and (2), for any predictable
candidate comparator u_t and fixed probe distribution v,

    E[L_T-L_0] + V sum_t E[mu_t(u_t)-mu_(t,a_t)]
      <= TB + R_meta
       + sum_t E[Q_C,t (C0_t+c_t(u_t)-b_C)]
       + sum_t E[Q_F,t (F0_t+d_t(u_t)+f_t(v)-b_F)]
       + V sum_t E[lambda(v) + S_t A(v)],                  (3)
    lambda(v)=sum_k v_k lambda_k,
    A(v)=sum_k v_k/(lambda_k m_k).

All candidates and comparators are defined at the learner's own visited state.
This is not a comparison of full trajectories generated by different learners.
It does not establish HAPPO superiority, gradient stationarity, physical
in-flight backlog stability, or universal noninferiority.

For a feasible idle/min-probe comparator, bounded conditional gain magnitude G,
S_t<=S_bar and queue-independent nonpositive comparator cost increments, (3)
implies E L_T <= T[B+V(G+lambda(v)+S_bar A(v))]+R_meta (L_0=0).
Hence positive expected cumulative resource violation per round is bounded by
sqrt(2 E L_T)/T for each queue. Strict common-cost slack additionally gives a
time-average expected queue bound by retaining the negative queue terms in
(3). Neither result holds merely because an empirical variance was observed.

Important: with a fixed finite probe catalogue the statistical term in (3)
does not vanish with T. It cannot be relabeled O(sqrt(T)) return regret. A
growing precision budget or an explicitly accepted error floor is necessary.
Virtual queues alone are NOT a hard finite-budget feasibility shield. A common
pre-data feasible candidate menu and admission rule must be defined before
adding a hard-budget claim; probe-dependent menus require a different proof.

## 5. Native integration and one bounded feasibility decision

Proceed with integration, not a new scientific run yet. Two issues must be
resolved together, not in a series of estimator-only wins:

- Use a standard, explicitly specified multi-epoch PPO/HAPPO server-owned update
  primitive. The failed study had just one actor backward/update pass per bank.
  This observation suggests underpowered optimization, but does not prove it.
  Preserve all source data, all server moments, and every candidate computation
  charge. Do not copy a stale worker's optimizer moments onto the server.
  Repeating Adam on the same stored raw gradient ten times is NOT multi-epoch
  PPO. Genuine multi-epoch proposals need immutable eligible trajectory/log-
  probability/advantage records and recomputed losses at each proposal state.
  A delayed data/update-program packet is therefore an explicit candidate
  native adapter, not a silent claim that the existing raw-gradient payload
  already supports standard PPO epochs. Freeze and report that payload change,
  source-policy/critic versions, independent service clock, and data cost before
  using it. Compare both zero-delay and matched-delay versions of the SAME
  update primitive. This adapter has not yet been implemented or qualified.
- Verify learning response using a frozen-initial-policy control evaluated on
  the SAME checkpoint worlds and common action randomness. Old checkpoint
  evaluations used different worlds, so their initial/final difference is not
  itself a paired same-world learning measurement. Evaluation remains excluded
  from controller input. A learning-signal pilot is development, never outcome-
  free operational qualification if it admits a method based on returns.
- The eligible reference for (1) is the unchanged policy or a pre-bank candidate
  built from the same ready packets. A fresh sequential HAPPO update using data
  not yet collected is not in that menu. Retain fresh HAPPO as a strong empirical
  anchor, while adding timing- and resource-matched asynchronous comparators.
  Do not remove HAPPO just to make a result positive.
- Quantify the precision required for useful action discrimination, not just a
  variance reduction ratio. For epsilon normalized expected selection regret,
  (1) needs m >= 4 S_t/epsilon^2; S_hat is a diagnostic, not a certificate for
  S_t. Obtain a justified variance/moment envelope or report the conditional
  scope and choose a separately frozen prospective precision qualification.
- The finite menu is exponential in team size. Unique-policy deduplication
  reduces exact duplicates, not asymptotic complexity. No scalable-team claim.
- Freeze integration, costs, roles, seeds and stop rules BEFORE new native data.
  At most one CPU worker; one bounded initial qualification, with no scientific
  retries. Only a useful complete-loop interface permits new performance
  development and later independent cross-family confirmation.

Four deterministic tests check transaction ordering, frozen-candidate identity,
full charges, genuine selection bias and exact finite-distribution moment
identities. They establish software/algebra consistency only. No successor
native performance test has been run and no new performance gain is claimed.
