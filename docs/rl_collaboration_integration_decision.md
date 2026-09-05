# RL integration decision: from readout aggregation to training consequences

2026-08-31. Parent 58b84c1. Internal AI-assisted research/qualification note,
not a paper, efficacy preregistration or claim of new algorithm superiority.

## Decision

The independent-bank readout reference is retained as a **baseline, not the
standalone ICML candidate**. Its stated theorem is not retracted. Two proposed
extensions are rejected: replacing its unbiased label by an ordinary TD
target, and applying its same-history comparator theorem to recursively
trained fixed graphs.

The bounded integration gate is finished, rather than deferred to another
small audit. The selected research problem is now:

> During actual personalized TD training, can a cheap, observable estimate
> of a transfer's effect on subsequent learning risk support useful
> collaboration relative to the learner that would have trained without it?

This is a problem choice, not an established novelty claim. Generic policy
regret, Bellman telescoping and Lyapunov risk metrics are inherited tools.
The nontrivial work is an affordable, valid estimator/controller under
Markov sampling, delayed donors and unequal tasks. No new performance pilot,
formal seeds, GPU job or experiment identifier is authorized by this note.

## 1. RL interfaces: exact qualification

### Ordinary TD cannot replace an unbiased value label

For a fixed-policy Markov reward process with transition P, reward mean r and
discount gamma<1, V=(I-gamma P)^(-1)r. A target bootstrapped from v obeys

\[
\mathbb E[r_t+\gamma v(S_{t+1})\mid S_t=s]
=(T^\pi v)(s)=V(s)+\gamma[P(v-V)](s).
\]

In the reference contrast identity, the extra conditional mean beta_s creates
the missing term 2a(z-c) beta_s. It is not sampling noise and does not
disappear by collecting more copies of the same bootstrapped target.

Exact one-state witness: r=1, gamma=.9, V=10, current local value c=0,
donor prediction z=8 and retention a=.8. The TD label is 1.
The actual executed squared-value-error contrast is -84.48, whereas the
incorrect unbiased-label calculation gives +30.72. The omitted term is
-115.2. Even the sign can be wrong.

### Complete finite-horizon returns are compatible, but change the object

For a fresh trajectory starting at a predeclared state, the full H-step
return G_H is conditionally centered around V_H, not generally V:

\[
\mathbb E G_H=\sum_{k=0}^{H-1}\gamma^kP^kr,\quad
V_H-V=-\gamma^HP^HV,\quad
\|V_H-V\|_\infty\le\gamma^H R_{\max}/(1-\gamma).
\]

If rewards are in [0,Rmax], G_H is bounded with interval width
Rmax(1-gamma^H)/(1-gamma). Its centered moment generating function is bounded
by exp(beta^2 width^2/8); thus the reference's exponential-supermartingale
argument can use the corresponding sub-Gaussian proxy. This requires
committing the prediction before that fresh return and charging every
transition. State resets/generative access cannot be presumed free.

For the public two-state audit P=[[.7,.3],[.2,.8]], r=[0,1], gamma=.9 and H=8,
V_H is approximately [2.328122,4.143247], while V is [4.909091,6.727273].
The roughly -2.58 truncation bias is not removed by more H=8 episodes.
This is a viable finite-return prediction interface, but plain readout
aggregation of these labels is not by itself the sought training algorithm.

### Bellman loss and value risk must not be conflated

With P=[[.1,.9],[.9,.1]], gamma=.9 and uniform state weighting, value-error
vectors [1,1] and [.1,-.1] have value MSEs 1 and .01 respectively. Their
squared mean Bellman residuals are .01 and .029584: the ordering reverses.
Norm bounds between these quantities do not imply identical pairwise
rankings of collaboration actions.

Nor can a symmetric Dirichlet surrogate silently replace general TD.
For stationary distribution mu, let A=diag(mu)(I-gamma P). The positive
Dirichlet construction equals (A+A^T)/2. It equals A for reversible chains,
but the three-state directed-cycle witness in the JSON has a nonzero
gradient of that symmetric surrogate at the true V. Its surrogate minimizer
has weighted value error .0854046. This does not refute valid TD gradient
splitting results; it refutes replacing splitting by ordinary gradients.

## 2. A static graph needs its own training history

The issue is not merely a statistical confidence interval. For two noiseless
scalar learners with targets (0,1), initialize both at zero and use local
update p_t=.5 x_{t-1}+.5 theta. Suppose actual training applies the averaging
graph J after every update.

Evaluating the local graph I on p_t from this J-history is not running local
training. Summing squared errors over T steps, the discrepancy is exactly

\[
\sum_t\{\|p_t^{J}-\theta\|^2-\|x_t^{I}-\theta\|^2\}
=T/8-(1-4^{-T})/6.
\]

At T=32, the same-history I cost is 4.166667, whereas I trained on its own
history costs .333333. The difference grows linearly even though the local
update is contractive. This is why ordinary online aggregation regret does
not establish a counterfactual fixed-training-graph guarantee.

The general distinction between ordinary regret and counterfactual policy
regret is established in Arora, Dekel and Tewari, cited as
[arXiv:1206.6400](https://arxiv.org/abs/1206.6400). The present witness is an
explicit calculation for our training interface, not a new generic
impossibility theorem.

## 3. Why a future-risk Lyapunov metric is relevant

The qualification uses actual Markov-jump TD matrices, not powers of a mean
TD matrix. Consider tabular TD with zero rewards (so V=0), current state s
and next state j:

\[
e^+=B_{sj}e,\qquad
B_{sj}=I-\eta e_s(e_s-\gamma e_j)^\top .
\]

Here e_s in B denotes a coordinate vector; e outside B denotes parameter
error. For H losses counted before successive TD updates, define Q_s^(0)=0,

\[
Q_s^{(H)}=I+\sum_jP_{sj}B_{sj}^{\top}Q_j^{(H-1)}B_{sj}. \tag{1}
\]

Conditioning on the first transition proves inductively that

\[
e^\top Q_s^{(H)}e
=\mathbb E[\sum_{k=0}^{H-1}\|e_k\|^2\mid e_0=e,S_0=s]. \tag{2}
\]

Thus (1) is the finite-horizon Markov-jump Lyapunov/value recursion. It does
not need an infinite-horizon stability assumption; such an assumption would
be needed for an infinite-horizon limit. We verify (1) against exhaustive
transition paths, not a Monte Carlo approximation.

For a one-time transfer d followed by local training, the exact future
excess cost is

\[
A_H(e,d,s)=2d^\top Q_s^{(H)}e+d^\top Q_s^{(H)}d. \tag{3}
\]

The Euclidean norm treats fast- and slow-learning error modes alike, while
Q weights them by their remaining learning cost. This is the concrete role
of a Lyapunov metric, rather than adding an unrelated debt queue.

### Exact sign-reversal and positive oracle qualification

Use P=[[.9,.1],[.4,.6]], stationary mu=[.8,.2], gamma=.9, eta=.5 and H=64.
The nontrivial Markov eigenvalue is .5. The stationary-averaged Q has
eigenvalues 8.3067607 and 28.3770473. A deliberately oracle-constructed
transfer moves unit error from the lower-cost eigenvector to .8 times the
higher-cost eigenvector.

- Current squared error decreases by .36 (36% from initial error one).
- Expected 64-step cumulative squared error increases by 9.8545496.
- Minimizing the future-risk quadratic along that donor direction instead
  chooses beta=.3138408 and reduces future risk by 2.6070005.
- Minimizing current Euclidean error chooses beta=.6097561 and reduces
  future risk by only .2893011.

These are exact known-model calculations, not a deployable controller,
heterogeneous-agent benchmark, or evidence of a new method's superiority.
The chosen vectors use privileged Q and true error specifically to test
the proposed objective. They are not fed into a purported online learner.

## 4. The counterfactual identity a final controller should exploit

Let Z_t include all current learner parameters, environmental Markov states,
and delay buffers. Let K_0 be the actual no-transfer training kernel and
ell(Z) the chosen personalized learning risk. Define the finite-horizon
baseline value by

\[
V_t^0(Z)=\ell(Z)+\mathbb E_{K_0}[V_{t+1}^0(Z')\mid Z],
\qquad V_{T+1}^0=0.
\]

A transfer acts first as R_a on parameters, then pays ell(R_a Z), followed
by K_0 from that changed state. Sampling policy remains fixed/exogenous.
For these actions, its baseline-relative training advantage is

\[
A_t^0(Z,a)=V_t^0(R_a Z)-V_t^0(Z). \tag{4}
\]

For any causal transfer policy pi, substitute the Bellman equation into
(4), take expectations along pi's actual trajectory, and telescope the
V_t^0 terms:

\[
J(\pi)-J(0)=\mathbb E_\pi\sum_{t=1}^T A_t^0(Z_t,a_t). \tag{5}
\]

This compares actual trajectories from the same initial state; it is not
the invalid same-history fixed-action comparison above. If true A_t^0<=0
at every chosen action, (5) guarantees expected finite-horizon no-harm
against actual no-transfer training. A confidence-based version needs
uniform adaptive coverage and a failure-event risk bound; neither follows
merely from computing a sample mean. This is standard Bellman telescoping,
not a claimed new performance-difference lemma.

The exhaustive-tree CPU test verifies both equality (5) and no-harm for a
small **oracle** policy that minimizes (3) at each node using remaining
horizon Q_s. This completes the known-model contract, not model-free safety.

### Affine extension and a small QP interface

For a known affine TD kernel v'=B_sj v+c_sj with deterministic rewards for
notational simplicity, and risk ||v-v_star||_M^2, write the local baseline
cost as v^T Q_s^h v+2p_s^h^T v+c_s^h. With all coefficients initially zero,
the exact backward recursion is

\[
Q_s^h=M+\sum_jP_{sj}B_{sj}^\top Q_j^{h-1}B_{sj},
\]
\[
p_s^h=-Mv_\star+\sum_jP_{sj}B_{sj}^\top
(Q_j^{h-1}c_{sj}+p_j^{h-1}),
\]
\[
c_s^h=v_\star^\top Mv_\star+
\sum_jP_{sj}(c_{sj}^\top Q_j^{h-1}c_{sj}
+2p_j^{h-1\top}c_{sj}+c_j^{h-1}).
\]

The symbol c_s^h is a scalar value coefficient, distinct from the
transition offset vector c_sj. Random reward offsets require their
conditional moments inside the same expectations. Q is positive
semidefinite for M positive semidefinite.

For a few delayed donor directions collected as columns of D, convex
transfer coefficients beta>=0, 1^T beta<=1 have oracle objective

\[
\beta^\top(D^\top QD)\beta+
2\beta^\top D^\top(Qv+p). \tag{6}
\]

This is a joint convex QP with beta=0 feasible, not an ad hoc graph scan.
The scalar oracle in the code solves its one-direction case exactly.
The baseline risk objective determines Q; it is not selected to make an
otherwise unrelated drift proof work.

## 5. One unresolved central implementation question

Can the small projected quantities D^T QD and D^T(Qv+p) be estimated and
upper-bounded from **fully charged observable data** cheaply enough that
nontrivial transfer remains beneficial?

They currently involve the unknown value target and transition law.
Constructing a full state/parameter Gramian and solving for V_star is an
audit oracle, not a low-complexity online solution. The current code must
never be presented as having answered this question.

Nor does the one-recipient qualification resolve joint heterogeneous
learning, source availability or strong-static-graph superiority. The
local-baseline contract (5) is distinct from matching the best static
collaboration policy. If sensing spends additional transitions or changes
the number of learning updates, remaining resources and elapsed sensing
must enter the baseline value/state; fixed-H (5) is not automatically a
dual-budget guarantee.

The next autonomous work is a single feasibility decision: derive an
observable, low-cost approximation/certificate for this advantage, with
its target-bias, Markov-dependence and omitted-tail terms explicitly
controlled. Reject the candidate if only oracle/full-model access or
permanent no-transfer can be justified. Only then freeze one CPU
development protocol. No frozen experiment is revived by this problem choice.

## Literature and evidence boundary

Verified primary sources establish the following inherited components:

- Lu and Giannakis, [arXiv:2112.00882](https://arxiv.org/abs/2112.00882):
  online TD function estimation with a weighted GP ensemble and cumulative
  error comparisons. This is not evidence of the same transfer controller,
  but rules out treating online value ensembles alone as our novelty.
- Ollivier, [arXiv:1805.00869](https://arxiv.org/abs/1805.00869):
  reversible-policy TD and Dirichlet-gradient geometry. Reversibility is a
  real restriction, not a property of all RL benchmarks.
- Liu and Olshevsky, [ICML 2021](https://proceedings.mlr.press/v139/liu21q.html):
  TD gradient splitting and finite-time analysis; splitting must not be
  silently identified with an ordinary symmetric gradient.
- Arora, Dekel and Tewari, cited above: history-sensitive policy regret.
  Their generic adversarial conclusions are not transplanted as lower
  bounds for this structured TD problem.

This is a bounded nearest-neighbor confrontation, not an exhaustive novelty
search. Source metadata, resolver failures and version distinctions are in
rl_collaboration_source_verification.json. No final bibliography was edited.
The numerical record is rl_collaboration_interface_results.json.
