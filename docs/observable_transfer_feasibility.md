# Observable transfer estimation: a complete reference and a failed compression route

2026-08-31. Internal AI-assisted feasibility package; parent 8a488b8.
No efficacy experiment, formal seed, or new experiment identifier is created.

## Decision

The direct coupled-probe estimator, robust controller and fixed-horizon
certificate can be specified without true values or a transition matrix.
However, this implementation is **rejected as the proposed inexpensive
long-horizon final method**. Its forward-propagation and sampling cost grows
with the prediction horizon and it requires conditional reset/simulation
access. Merely outputting a small projected quadratic does not compress that
work. Retain it as a correct observable reference and potential comparator.

An exact Markov-TD counterexample also rejects closing the recursion inside
the donor subspace without an invariance assumption or a leakage bound.
Neither conclusion is an impossibility theorem for all estimators, nor a
failure of the earlier known-model risk identity.

## 1. Observable sampling contract

Consider a finite fixed-policy Markov reward process with d states, discount
0<=gamma<1, and rewards bounded by |r|<=R. Let B=R/(1-gamma). Local tabular TD
uses step 0<eta<=1. At a decision, freeze the local value vector v, current
Markov state s, and k already received donor directions D. Every local/donor
value lies in [-B,B]^d. Beta belongs to C={beta>=0,1^T beta<=1}, so v+D beta
also lies in that box. Donor messages may be stale; the actual frozen message,
not a hypothetical fresh one, defines D.

Before new observations, fix H (remaining learning horizon), L (return-label
length), n (replicates), and delta. Each fresh replicate performs:

1. Draw K uniformly from {0,...,H-1}, independently of future trajectories.
2. From state s, simulate K local TD transitions, starting from v. Propagate
   the k columns of D on the same transitions as parameter sensitivities.
3. Draw an independent evaluation state U uniformly from the d states.
4. From U obtain an independent L-step discounted return Y_L.

The training and evaluation trajectories may each have arbitrary temporal
Markov dependence. Independence is across fresh replicates, and between the
training trajectory and its evaluation return conditional on U. This is a
conditional simulator/reset assumption, not an iid assumption about the steps
of a continuing chain. Cloning a random-number-generator state does not supply
independent replicas. Reset requests and every transition are charged.

The online encoder receives only transition records (state,reward,next_state),
the evaluation state and the frozen context. It never receives P or V*. The
exact qualification below uses P/V* outside the encoder to verify expectations.
Uniform evaluation over states and conditional resets are substantive access
requirements, not capabilities supplied by every RL environment.

## 2. Exact directional propagation and estimator

For observed transition (s,r,j), tabular TD is affine:

\[
v^+=B_{sj}v+\eta r e_s,\qquad
B_{sj}=I-\eta e_s(e_s-\gamma e_j)^\top.
\]

Let U_0=D and U_{t+1}=B_{S_tS_{t+1}}U_t. Training from v+D beta on this same
trajectory gives exactly v_t+U_t beta, by induction. Therefore the squared
prediction-risk contrast at state u equals

\[
(U_t[u,:]\beta)^2+
2(U_t[u,:]\beta)(v_t[u]-V^*[u]). \tag{1}
\]

One observable replicate outputs

\[
\widehat G=H U_K[U,:]^\top U_K[U,:],\qquad
\widehat g=H U_K[U,:]^\top(v_K[U]-Y_L). \tag{2}
\]

For V_L=E[Y_L|U], averaging (1) over K,U and training paths proves

\[
E[\beta^\top\widehat G\beta+2\widehat g^\top\beta]
=J_H^{V_L}(v+D\beta,s)-J_H^{V_L}(v,s). \tag{3}
\]

Both sides use the same executed local TD rule and sum H pre-update risks,
with uniform state-weighted squared prediction error. They do not replace TD
targets by unbiased infinite-horizon value labels. The remaining label bias is
explicit: ||V_L-V*||_infinity<=B gamma^L. If Delta=max|D_ab|, then

\[
|g_{L,a}-g_{*,a}|\le H\Delta B\gamma^L=:b_g. \tag{4}
\]

Proof: B_sj is nonnegative with row sums at most one, hence
||U_t[:,a]||_infinity<=Delta. Substitute V_L-V* into (2). The same TD update
keeps v_t in [-B,B]^d because its target r+gamma v[j] is bounded by B.

The scalar contrast identity (3) holds simultaneously for the whole finite
donor span; it is not a reward-free estimator.

## 3. Finite-data certificate and the executed QP

Let Gbar,gbar be n fresh-replicate averages. There are
m=k(k+1)/2+k upper-triangular Gram and linear coordinates. Set

\[
c_n=\sqrt{2\log(2m/\delta)/n},\quad
r_G=H\Delta^2c_n,\quad r_g=2H\Delta Bc_n.
\]

Each Gram coordinate lies in [-H Delta^2,H Delta^2], and each linear
coordinate in [-2H Delta B,2H Delta B]. Hoeffding's bounded-variable inequality
and a union bound give, with probability at least 1-delta,
max|Gbar-G*|<=r_G and max|gbar-g*|<=r_g+b_g. This probability is conditional
on the frozen context. Correlated steps inside a replicate are never counted
as separate independent observations. Ordered unique replicate IDs prevent
accidental double counting, but do not prove the required sampling law.

Specifically, for any coordinate X in [-M,M], the bounded-variable exponential
bound is E exp(lambda(X-EX))<=exp(lambda^2 M^2/2). Multiplication across n
independent replicas and Chernoff optimization yield
P(|Xbar-EX|>M c_n)<=2 exp(-n c_n^2/2)=delta/m. Apply this to each of the m
coordinates, then add the deterministic target bias (4). No union over a
discretized beta grid is needed because the coordinate bound implies (5).

Consequently every beta in C simultaneously satisfies

\[
A_H(\beta)\le U_H(\beta):=
\beta^\top\bar G\beta+2\bar g^\top\beta
+r_G(1^\top\beta)^2+2(r_g+b_g)1^\top\beta. \tag{5}
\]

The uniform statement permits selecting beta after observing this batch,
provided v,D,s,H,L were fixed before collecting it. Changing those inputs
invalidates reuse; the implementation rejects context-hash mismatches.
This is a fixed-n certificate. Adaptive stopping requires an additional
time-uniform construction or a prespecified union allocation across n.

The matrix Gbar is positive semidefinite, so (5) is a convex QP with matrix
Gbar+r_G 11^T and linear coefficient gbar+(r_g+b_g)1. The zero action has
exact upper advantage zero. The controller minimizes this upper bound and
executes only a feasible candidate with negative upper value; otherwise it
returns zero. For k=1 its exact solution is

\[
\beta=\operatorname{clip}_{[0,1]}
\frac{-(\bar g+r_g+b_g)}{\bar G+r_G},
\]

with the zero-curvature case handled separately. For general k the reference
uses I projected-gradient iterations from zero with step 1/L_Q, where
L_Q=2 lambda_max(Gbar+r_G11^T). Projection is onto C. The real-arithmetic
optimization error is at most L_Q/(2I), since ||beta*||_2<=1. This follows by
summing the projected-gradient descent inequality against beta*, then using
monotonicity to bound the final objective by the average gap. For L_Q=0,
linear minimization over C is solved by zero or a simplex vertex.

Safety depends on the checked upper value, not on numerical optimality.
On the coverage event the chosen action has A_H<=0. Since every risk is in
[0,4B^2], on its complement A_H<=4HB^2. Thus the one-decision statement is

\[
E[A_H(\widehat\beta)\mid\text{frozen context}]
\le4HB^2\delta. \tag{6}
\]

Under the matched-kernel timing of the existing full-trajectory identity,
conditional bounds of this form sum to an expected cumulative excess-risk
bound. This is not a pathwise risk guarantee or a terminal-error guarantee.
Nor does it prove per-agent safety from an aggregate guarantee.

There is a nontriviality condition: on coverage, 0<=U_H(beta)-A_H(beta)<=2eps
for eps=r_G+2(r_g+b_g). If some beta has A_H(beta)<-2eps-L_Q/(2I), the
computed QP must have negative upper value in exact arithmetic. This is a
sufficient, potentially conservative condition, not an information lower bound
or proof that it is affordable. Familiar concentration and convex optimization
are inherited tools; this conditional result is not asserted to be novel.

## 4. Full cost and the compression failure

For n replicas the expected environment transition count is

\[
n[(H-1)/2+L],
\]

and its maximum is n(H-1+L). The contract additionally requests 2n resets and
kd donor parameter scalars before probing. Expected forward TD propagation
requires n(H-1)/2 steps. The tabular encoder costs
O(n[dk+Hk+L+k^2]); dense-feature propagation costs O(n Hdk) plus label and
moment work. The small QP costs O(k^3+I[k^2+k log k]). These are separate costs.

The H=64,L=32,n=128 arithmetic example requires 8,128 expected transitions
(12,160 worst case), 256 resets and 4,032 expected forward TD steps per frozen
context. These are analytic counts, not a measured benchmark run. Its stated
R=1,gamma=.9,Delta=.5 confidence radii are r_G=4.89549, r_g=195.81975 and
b_g=10.98779. Loose bounds alone do not establish statistical impossibility.

The direct implementation still differentiates the local trajectory through
K updates: U_t is exactly the forward derivative with respect to beta. For
a fixed K, the contrast gradient is 2 U_K[u,:]^T(v_K[u]-Y_L). This is the same
chain-rule primitive as a matched unrolled/meta-gradient collaboration
comparator. Randomizing K estimates the desired sum; it does not eliminate
the work or establish superiority over that comparator.

The probe cost also reduces resources available to actual training. Formula
(6) compares the same H learning steps and excludes lost updates due to
sensing. It is NOT a same-dual-budget no-harm theorem. Resources would have
to enter the baseline state/value for that stronger claim. Likewise, a single
continuing trajectory is not made into fresh conditional replicas by splitting
or replaying it without an additional dependence/generalization argument.

## 5. Why projecting the recursion is not a shortcut

Take P=[[.9,.1],[.4,.6]], gamma=.9, eta=.5, initial environmental state 1
(zero based), zero rewards, donor subspace span(e_0) and H=2. The exact
conditional future-risk projection is

\[
e_0^\top Q_1^{(2)}e_0=2+P_{10}(\eta\gamma)^2=2.081.
\]

Projecting B_sj e_0 back to span(e_0) before accumulating risk gives 2.
The missing .081 is the second-coordinate error created by the TD update.
Thus compressed coefficient outputs do not imply closed compressed dynamics.
An invariant donor subspace or a certified residual/leakage term is needed.
This witness uses a genuinely mixing two-state chain and is not a claim that
all matrix-free or amortized methods are impossible.

## 6. CPU qualification and next decision

The code implements observed-data encoding, context/duplicate protection,
streaming moments, the uniform QP penalty, scalar and joint controllers,
complete transition/reset accounting, and exact finite-model qualification.
Twenty deterministic tests cover the entire interface rather than a new
parameter-sweep experiment. Exact mean contrast is -0.36471799447552 for
the finite-return target, versus -0.39415726692352 for the infinite target;
the .029439272448 bias is explicitly covered. These are algebraic expectation
checks, not performance or coverage results from independent scientific seeds.

Do not promote this direct-probe reference to a new efficacy pilot. The next
mechanism worth examining must reuse/amortize directional information with a
proved context-change/leakage error, or provide another genuine reduction in
work. It must confront the same short-unroll comparator. Repeatedly tightening
this particular worst-case bound or increasing n is not by itself the desired
new contribution. Preserve the current reference and stop decision.

## Source boundary

The concentration step is attributed to Hoeffding, *Probability Inequalities
for Sums of Bounded Random Variables*, JASA 58(301):13-30 (1963),
[publisher DOI](https://doi.org/10.1080/01621459.1963.10500830).
Crossref and OpenAlex agree on its metadata; publisher full-text access
returned 403, so no fresh full-text reading is claimed. The bounded-variable
derivation and each substitution used here are written explicitly above.

The related chain-rule/online validation principle is documented in Section 1,
equations (1)-(5), of Xu, van Hasselt and Silver,
[Meta-Gradient Reinforcement Learning](https://arxiv.org/html/1805.09801v1).
Its original algorithm adapts return parameters; our comparison is to a
matched transfer-weight instantiation, not a claim that the original paper
already supplies the same certificate. Source access and metadata records
are preserved in observable_transfer_source_verification.json.
