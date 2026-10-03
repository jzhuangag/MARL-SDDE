# Actual-training risk with delayed feedback: a qualified interface, not the final idea

2026-08-31; parent 07d53e4fcbdb2307b43ab85b4e35c96510fef7bd.
Internal AI-assisted mathematical qualification; no performance preregistration.

## Decision

The actual-execution, delayed-feedback and safe-restoration interface CAN be
closed for cumulative **pre-update visited-state prediction risk**. The reference
below is executable, handles overlapping returns, reserves every pending action,
and does not charge hypothetical rejected proposals. Retain this as a checked
reference; do NOT promote it to the intended ICML algorithm.

The original full-state learning-risk objective is not thereby certified.
Nor does this ledger establish a Lyapunov drift/convergence theorem, a regret
guarantee, or a learning-specific novelty. Reject the generic ledger plus shadow
restoration as a sufficient final contribution. No new CPU pilot is authorized.
This closes the bounded integration question from the previous plan without
silently changing the scientific objective or renaming a historical controller.

## 1. Prior-design and literature confrontation

The immutable T-071A implementation already compared a candidate to its own
local shadow and restored the shadow on rejection. Its independent fingerprint
transitions were not learning updates. T-072 already reused learning data,
retained a separate local shadow, and used a reflected validation-debt penalty.
T-071A remained a 4/12 failure and T-072 a 7/8 outcome-informed calibration
failure. Their source, cells, seeds, gates and outcomes are not edited here.

There is a material timing distinction in t072_dual_use_graph_controller.py:
`validation_excess` scores the frozen pre-update proposal against `shadow_pre`,
whereas the reported block risk uses accepted `candidate_post` or `shadow_post`.
The debt also includes rejected proposals. That debt is neither the sum of the
executed post-update risks nor an automatically valid Markov confidence bound.
Disjoint AR trajectory sections alone do not establish conditional independence.
These are reasons not to reuse its safety claim unchanged, not new retuned results.

Conservative Bandits [1], Section 3.3, already gates a proposed arm using a lower
confidence bound on remaining baseline budget and falls back to the baseline.
Appendix A proves safety on simultaneous coverage, separately from its regret
argument. The known positive baseline and multiplicative slack fund exploration.
Thus budget invariants and safe fallback are inherited components.

Prudent-Banker [2] explicitly addresses delayed baseline safety using missing-
feedback corrections and baseline mixtures. Its stated comparison is expected
regret against a fixed full-support baseline, not the all-prefix high-probability
training-risk statement below. This recent preprint is relevant counterevidence
to a broad novelty claim, not a verified substitute for our theorem. Its complete
proofs have not been audited here. Merely adding delay to a safety ledger is not
a defensible distinction.

## 2. Declared model, risk and execution order

For one recipient, let (S_t,R_t,S_{t+1}) be a fixed-policy finite Markov reward
process, with |R_t|<=R, discount 0<=gamma<1 and B=R/(1-gamma). Its value is V.
F_t includes S_t, all available past data and the actual decision randomness,
but not the forthcoming rewards. Require the conditional future environment
law given F_t to equal the registered Markov law given S_t. No reset access or
mixing rate is needed for the identity below, but hidden common noise or changing
acting policies can violate this conditional-law assumption.

Let l_t be the independently TRAINED local shadow, not a one-step local update
of the collaborative parameter history. Let p_t be any available bounded
proposal using timestamped donor information and past learning data. Choose
x_t=l_t+alpha_t(p_t-l_t) before observing R_t, with alpha_t in [0,1] and every
coordinate of l_t,p_t in [-B,B]. Then both x_t and l_t receive the same local TD
transition. For ordinary tabular TD and 0<eta<=1, this box is preserved.
The next proposal may depend on the previously executed, updated x_t: this is
recursive collaborative training, not a readout-only local bank.

The risk charged at decision t is precisely

\[
g_t=(x_t(S_t)-V(S_t))^2-(l_t(S_t)-V(S_t))^2. \tag{1}
\]

The goal of this REFERENCE is sum_(t=0)^(n-1) g_t <= A_0+epsilon n for every
n<=T, with probability at least 1-delta. A_0 and epsilon are explicit allowed
harm, not communication/environment budgets. A reset alpha=0 replaces actual
parameters with the own-shadow parameters before this risk is incurred.
Stopping future donor messages without replacing damaged parameters is not
the same action. The caller must execute the returned vector; scoring an
unexecuted proposal cannot discharge its reservation.

## 3. Observable return identity, with actual parameters frozen

Freeze c_t=x_t(S_t), a_t=l_t(S_t) at execution. Fix return length L and extra
delivery delay D before running. Write K=L+D and

\[
Y_t=\sum_{j=0}^{L-1}\gamma^j R_{t+j},\qquad
X_t=(c_t-a_t)(c_t+a_t-2Y_t).
\]

This label arrives at F_(t+K). During its construction, those transitions can
continue updating both learners; the recorded c_t,a_t do not change. By the
conditional Markov law, E[Y_t|F_t]=V_L(S_t), with
|V_L(s)-V(s)|<=B gamma^L. Expanding the two squares therefore gives

\[
\mu_t:=E[X_t\mid F_t]=(c_t-V_L(S_t))^2-(a_t-V_L(S_t))^2,
\quad |g_t-\mu_t|\le b_t:=2B\gamma^L|c_t-a_t|. \tag{2}
\]

This is a finite-return value-risk identity, not an ordinary one-step TD target
substitution. It applies to the executed values, not a candidate subsequently
rejected or updated before being recorded.

Overlapping returns are not independent. Partition times by t mod K. Within
each class, the preceding label is available before the next choice. The
centered variables mu_t-X_t have conditional mean zero. Since X_t is in
[-4B^2,4B^2], the conditional bounded-variable exponential inequality [3]
gives E[exp(lambda(mu_t-X_t))|F_t]<=exp(lambda^2(4B^2)^2/2).

Zero-displacement actions have g=X=b=0 and can be omitted predictably. Applying
this exponential bound to each class, stopping at its nth active completion,
and union bounding over K classes and at most T counts yields simultaneous
prefix radii

\[
r_j(n)=4B^2\sqrt{2n\log(KT/\delta)},\qquad r_j(0)=0. \tag{3}
\]

The stopping argument can be capped at T; it does not condition on a favorable
realized count or assume independent active actions. Future actions are allowed
to use earlier raw rewards. The fixed release order makes each settled class a
chronological prefix. Outcome-dependent delivery/censoring or arbitrary subset
settlement is NOT supported by this implementation.

For m completed actions, let M_m and P_m be the matured and pending nonzero
actions, with n_j matured counts in class j. An exact deterministic upper
reservation on an individual current-state risk is

\[
u_t=c_t^2-a_t^2+2B|c_t-a_t|\ge0.
\]

On the single event of probability at least 1-delta,

\[
\sum_{t<m}g_t\le
R_m:=\sum_{t\in M_m}(X_t+b_t)+\sum_j r_j(n_j)
                       +\sum_{t\in P_m}u_t. \tag{4}
\]

Labels are counted once. Several labels may reuse the same reward, which changes
the dependence and arithmetic, not the count of physical environment transitions.

## 4. Admission, certificate retention, and finite-time guarantee

Set C_0=0. After any arrivals, tighten the previous prefix certificate by
C_m <- min(C_m,R_m). Never overwrite it with a larger noisy refreshed bound:
that could consume allowance that was not reserved when the actions executed.
At the next action, available allowance is
w=A_0+epsilon(m+1)-C_m>=0. Select the largest feasible alpha on the segment
from the true shadow l to proposal p. At the visited coordinate set d=p(S)-l(S).
The reservation is

\[
u(\alpha)=d^2\alpha^2+2(l(S)d+B|d|)\alpha.
\]

Both coefficients are nonnegative. If u(1)<=w use alpha=1; if w=0 and d!=0,
use zero; otherwise with a=d^2 and b=2(l(S)d+B|d|),

\[
\alpha=\min\left\{1,\frac{2w}{b+\sqrt{b^2+4aw}}\right\}. \tag{5}
\]

When d=0, all alpha have zero current-state reservation. The implementation
uses one conservative halving and then exact shadow restoration if rounding
would overrun w or the total allowance. The formula is maximal in real
arithmetic; its floating-point fallback may be conservative. It never
undercharges the executed vector. Set C_(m+1)=C_m+u(alpha), then consume the
raw transition. An unbounded ulp-by-ulp loop is specifically excluded: when
candidate and shadow nearly coincide, one rounded parameter value can span
an enormous number of alpha ulps.

**Reference theorem (cumulative visited-state safety).** Under the stated fixed
law, boundedness and timing, for all n<=T simultaneously, with probability at
least 1-delta, sum_(t<n) g_t<=A_0+epsilon n.

Proof: on the event for (4), tightening preserves a valid certificate for the
already executed prefix. Adding u bounds the actual next g deterministically.
Induction yields sum g<=C_n, while admission yields C_n<=A_0+epsilon n on every
sample path. Maturation can only reduce C. This includes the final K-1 labels,
which remain reserved at termination; no extra uncharged tail rollout is used.

Since g_t<=4B^2, expected average visited-state risk is at most the own-shadow
average risk plus A_0/n+epsilon+4B^2 delta. This is a comparison inequality:
convergence needs a separately valid baseline convergence theorem for this
sampling law and metric. No general affine-TD rate is proved by the ledger.

The prefix confidence radius is at most
4B^2 sqrt(2K m log(KT/delta)); finite-return bias and pending reservations are
additional costs. Zero initial allowance and zero slack force alpha=0 whenever
the current-state displacement is nonzero. Positive allowance permits nonzero
alpha but does not prove useful learning. A safety theorem is not a gain theorem.

## 5. Two decisive scope counterexamples

**Visited state is not full-state MSE.** With gamma=0, two states, l=(0,0),
x=(0,1), and an observed history entirely in state 0 with zero reward, choose
V^+=(0,1) or V^-=(0,-1). Both laws generate the same observations on that event.
Every visited-state contrast is zero. The uniform full-state contrasts are
-0.5 and +1.5 respectively. The implemented zero-allowance ledger even permits
this parameter change because its current-state reservation is exactly zero.
It would be false to relabel its guarantee as full-state MSE safety.

This is not confined to reducible chains. With transition matrix
[[1-z,z],[z,1-z]], z=10^-4, both environments are irreducible and geometrically
mixing. Starting in state 0, a 64-step no-visit event has probability
(1-z)^64>0.99. Finite-time all-state identification needs quantitative state
coverage or additional state-query access; “the chain is mixing” is insufficient.
This witness rules out the metric implication, not every full-state controller.

**Cumulative safety is not reflected-debt stability.** For excess sequence
(-1,+1), every prefix sum is <=0, but Q_(t+1)=[Q_t+g_t]_+ produces (0,1).
Past gains can finance later harm in a credit constraint. They are discarded
by a reflected queue, whose meaning is different. Replacing one with the other
inside a composite Lyapunov function changes the claim. Calling the admission
invariant a Lyapunov convergence proof would be incorrect.

## 6. Complexity, physical resources, and paper decision

The reference handles one recipient with vector dimension d. It stores O(K)
pending scalar records, propagates up to L return accumulators, recomputes an
O(K) confidence sum, and forms the actual d-vector. Thus this implementation
uses O(d+L+K) arithmetic per action and O(d+K) workspace, not unqualified O(1).
Both learner and shadow must additionally pay their own TD arithmetic. There
is one physical transition per step; reuse of its reward by several labels is
not extra sampling. Donor messages, proposal construction and server costs
remain external and must be charged by any future runner. No complete matched
communication-budget or strong-static-graph comparison is established here.

The local tests include exact return expectations on a correlated two-state
MRP, a 2^6 path unit fixture, executed-versus-proposed action accounting,
chronological arrivals, retained confidence, final pending risk, actual/local
recursive updates, and both scope counterexamples. These are deterministic
verification fixtures, not new seeds, efficacy evidence, or calibration data.

**Paper-level decision: do not launch this wrapper as a new mainline.** The
interface result is useful, but the metric differs from the intended full-state
learning risk and the core budget primitive is inherited. There is no demonstrated
learning-specific advantage or complete Lyapunov finite-time rate. It would be
misleading to claim the original theory-algorithm gap is now solved.

The next bounded work is a single paper-level decision memo, not another generic
wrapper: compare the qualified full-state future-risk references and this cheaper
visited-risk reference, identify exactly what observable information a genuinely
learning-specific mechanism would require, and decide whether a differentiated,
feasible route exists under the original CPU-accessible model. If it does not,
report the necessary model/claim changes explicitly before any efficacy protocol.
Do not silently weaken the risk metric, add a new experiment number, or tune on
T-083A formal outcomes. No GPU is needed for that decision.

## Sources and verification

[1] Wu, Shariff, Lattimore and Szepesvari. Conservative Bandits. ICML 2016,
PMLR 48:1254-1262. [Proceedings](https://proceedings.mlr.press/v48/wu16.html).
Read scope: arXiv:1602.04282v1, Sections 2 and 3.1-3.3, Algorithm 1, Theorem 2
and Appendix A; metadata independently matched to the proceedings.

[2] Hu, Cai and Vlatakis-Gkaragkounis. Prudent-Banker: No Extra Fees for Baseline
Safety in Adversarial Bandits With and Without Delays. arXiv:2605.23351v1, 2026.
[Primary preprint](https://arxiv.org/abs/2605.23351).
Read scope: abstract, Sections 2-4.2 and relevant displayed definitions/algorithm;
no complete proof audit or peer-reviewed-publication claim.

[3] Hoeffding. Probability Inequalities for Sums of Bounded Random Variables.
JASA 58(301):13-30, 1963.
[DOI record](https://doi.org/10.1080/01621459.1963.10500830).
Conditional bounded-variable exponential specialization and color-prefix proof
are written explicitly above; no novelty attribution for the concentration tool.

Fresh metadata/access records: delayed_training_risk_sources.json. Command,
hash, full-regression and replay records: delayed_training_risk_execution.json.
No .bib or paper manuscript is delivered by this internal qualification.
