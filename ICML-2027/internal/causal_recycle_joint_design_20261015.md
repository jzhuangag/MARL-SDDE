# Causal temporal recycling: a complete-algorithm successor

## Material Passport

Type: implementation and mathematical design; outcome-free, not publication
evidence. Preserves every A1 and variance-qualification failure. Uses ARS
experiment planning/execution. No GPU/HPC4 or manuscript edits authorized here.

## One problem, one design

Maximize native cooperative team learning return at a finite training budget
while controlling *prospective* policy-broadcast and extra score-forward rates.
Distinct actors and a centralized training critic implement CTDE. Raw PPO
gradients are launched from recorded joint policy versions and return through
independently delayed transport links. The server alone owns Adam moments.
Launch frequency is fixed and fully paid; this algorithm does not optimize
unknown launch value or claim compute-worker asynchrony. Its variables are the
joint receipt update pattern and the number of training episodes scored.

The substantive cost correction is temporal recycling, not more samples.
At round t, returned gradients, all Adam variants, the potential, hints, action
distribution, sampled action, and probe count are fixed BEFORE the new bank.
The bank is sampled from the unchanged server behavior. Its first m episodes
provide untruncated, joint-likelihood, actual-return contrasts. Apply the
preselected action, then use that same bank to generate FUTURE gradients and
train the critic. The source version remains the pre-action version. Feedback
never selects the update that generated its own bank. Reuse charges each native
transition once and every additional forward/fitting/proposal operation.

No winner's-curse certificate and no population-variance plug-in are required:
predictable optimistic mirror descent (OMD) consumes unbiased full-information
feedback. A second OMD chooses m using an unbiased sample-variance loss vector.
Lyapunov queue prices enter BOTH decision maps. Independent episodes are the
sampling unit; arbitrary time dependence inside each episode is permitted.

## Frozen mathematical interface

Let F_t contain all past banks, parameters, server optimizer states, returned
packet identities, and independent service outcomes revealed before round t.
Let beta_t be the frozen behavior, pi_t^a a frozen joint candidate, H the episode
length, and g_t(a)=(J(pi_t^a)-J(beta_t))/H. The catalogue is all {0,.5,1}^N
patterns, including idle; missing actor packets map their coordinate to idle.
Costs c_t(a) are known outbound parameter-byte fractions. Define the predictable
extra forward cost f_t(m), including old-bank hint work, and rates b_C,b_F.
Current bank episodes are iid conditionally on F_t. The action and m are sampled
independently from predictable p_t,q_t, before any current episode is opened.
The behavior bank does not depend on these two draws.

For a frozen state/time potential v, define L_h^a as the UNCLIPPED joint
trajectory likelihood ratio through action h and

X_t^a = sum_h (L_h^a-1)[r_h+v_{h+1}(s_{h+1})-v_h(s_h)], v_H=0.

Conditional expectation is J(pi_t^a)-J(beta_t). The proof is importance change
of measure followed by telescoping state potentials; initial states have the
same law. This requires behavior support over every candidate action and finite
moments, NOT an accurate value function. PPO/GAE creates candidates only; no
claim that its gradient is an unbiased return gradient is used.

Let Y=X/H, g=E[Y|F_t], Sigma=Cov(Y|F_t), S=trace(Sigma). Then the mean feedback
has covariance Sigma/m and the usual ddof=1 sample variance trace S_hat obeys
E[S_hat|F_t,m]=S. Consequently S_hat/m' is an unbiased loss for EVERY catalogue
probe count m', even though just m current episodes were scored. It is not a
high-probability upper bound. Finite fourth episode moments control its second
moment. No return evaluation bank enters this construction.

## Joint update and finite-horizon result

Use Euclidean simplex projection P, positive eta,zeta,V, and states w_t,z_t:

p_t=P(w_t+eta[h_t-Q_C,t c_t/V]), a_t~p_t;
q_t=P(z_t-zeta Q_F,t f_t), m_t~q_t;
w_{t+1}=P(w_t+eta[g_hat_t-Q_C,t c_t/V]);
z_{t+1}=P(z_t-zeta[Q_F,t f_t+(V eta/2) S_hat_t (1/m')_{m'}]);
Q_C,t+1=[Q_C,t+c_t(a_t)-b_C]_+;
Q_F,t+1=[Q_F,t+f_t(m_t)-b_F]_+.

The hint h_t can be biased and is clipped only as a PREDICTION. Neither current
feedback nor importance weights are clipped. The Lyapunov is
L_t=(Q_C,t^2+Q_F,t^2)/2 - V J(beta_t)/H. Since beta_{t+1}=pi_t^{a_t}, actual
gain telescopes across the complete sequence, including the last action.
Delayed unfinished gradients are not falsely counted as future return gains.
Their work and reserved incoming bytes are charged at launch and fully drained
at terminal time; only accepted policies change J.

For any fixed probability vectors u over actions and v over probe counts,
let A_t=||g_t-h_t||_2^2 and M_t=E[S_hat_t^2|F_t] (or its conditional-m averaged
version). With simplex diameter squared at most 2, define

R_T = V/eta + V eta/2 sum_t E[A_t]
    + 1/zeta + zeta/2 (V eta/2)^2 sum_t E[M_t] sum_m 1/m^2.

If costs are bounded and B_0 bounds half the sum of squared queue increments,
the implemented two-OMD rule satisfies

E[Q_C,T^2+Q_F,T^2]/2 + V sum_t E[u dot g_t - g_t(a_t)]
 <= T B_0 + R_T + (V eta/2) sum_t E[S_t] sum_m v_m/m
    + sum_t E[Q_C,t(u dot c_t-b_C)+Q_F,t(v dot f_t-b_F)].

Proof: (1) projection optimality gives the OMD regret inequality against u;
its squared residual is ||g_hat-h||^2, whose conditional expectation is
A_t+S_t/m_t. (2) A second projection inequality against v, using Q_F f as its
known hint, replaces the sum of Q_F f(q_t)+(V eta/2)S_t E_q[1/m_t] by the v
counterpart plus the stated meta regret. (3) combine these with quadratic queue
drift and take conditional expectations. No current-bank argmax or selected-
action feedback is substituted in this proof. Adaptive visited states are
allowed because the comparison is along THIS learner's predictable contexts.

For per-round feasible u,v the final queue-price term is nonpositive. This is
an expected fixed-mixture, finite-menu return-gain comparison at visited states,
not best-changing-action regret, gradient stationarity, per-step monotonicity,
or dominance over a separate HAPPO trajectory. Bounded A_t,S_t,M_t and
eta,zeta~T^-1/2, V~T^1/2 give O(sqrt(T)) gain regret after division by V.
With an idle action and a strictly feasible smallest probe, the same inequality
and bounded J imply O(1) time-average EXPECTED backlog plus constants set by
moment bounds. In particular choose u=idle, so its gain is identically zero;
the learner's cumulative gain telescopes to (J_T-J_0)/H, uniformly bounded.
The remaining right-hand side is O(T) under these parameter scalings, hence
E[Q_C,T^2+Q_F,T^2]=O(T). Jensen and the queue recursion then give O(sqrt(T))
expected positive cumulative constraint violation. This argument uses the idle
comparator and bounded telescoping return, not an arbitrary comparator's gain.
Uniform moment bounds are explicit mathematical assumptions, not certificates
estimated from a few real trajectories. They need auditing for any extension.

In the current implementation, hint forward cost <=.375 and smallest-probe
cost <=.25, so b_F=.75 has strict slack at least .125; idle c=0 has .5 slack.
The outbound target is half a full-team policy broadcast per round; the scoring
target is three quarters of two full-team candidate forwards on a training bank.
These are prospective resource operating points, not outcome-selected thresholds.
Catalogue enumeration is exponential in N. Neural forwards cache per-actor
ratios, but the algorithm is not claimed sparse-degree scalable.

## Actual experiment decision, not theory as a performance substitute

Operationally qualify native Spread and Reference, all ten methods, six rounds,
one disjoint seed, without using qualification returns for tuning or ranking.
Verify source hashes/versions, server-owned moments, single-use packets,
independent service, no evaluation leakage, actual charge recurrences and replay.
Only then separately lock a 60-run complete-learning development: two native
tasks, three fresh seeds, ten methods, forty rounds. Strong controls include
fresh sequential HAPPO at three learning rates, immediate latest-packet updates,
delay adaptation, largest-gradient and cyclic scheduling. Baselines are not
forced to wait for score banks or compute unused shadow candidates.

The primary decision is complete native-transition return AUC against the
strongest registered non-recycle baseline in EACH task, with seed consistency
and final-return preservation. Total logical communication AUC over the common
observed range, all compute/probe bills, and CPU are also reported. They are
not hidden if primary gains pass. Three seeds are DEVELOPMENT, not an ICML
confirmation. No GPU launch or manuscript admission follows from software tests.

## Prior-work boundary

Predictable OMD and virtual queues are established tools, not new by themselves.
Rakhlin and Sridharan, COLT 2013, PMLR 30:993-1019, primary record:
https://proceedings.mlr.press/v30/Rakhlin13.html ; arXiv:1208.3728.
Yu, Neely and Wei, NeurIPS 2017, primary record:
https://papers.nips.cc/paper_files/paper/2017/hash/da0d1111d2dc5d489242e60ebcbaf988-Abstract.html ;
arXiv:1708.03741. Their iid stochastic-constraint result is not directly invoked
for policy-dependent costs; the above proof is written for predictable costs.
HAPPO/HAML, CoPPO, MAT, FP3O and B2MAPO already cover important concurrent/block
policy-update ideas. No first-parallel/first-monotone claim is made. The proposed
research hypothesis is causal temporal recycling with fully charged joint
receipt/probe control; defensible novelty still depends on substantive evidence.
