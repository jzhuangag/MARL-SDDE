# A complete calibrated-sharing construction, and why it is not the paper

2026-09-01. Parent: 47ea28c45f1b11b1c4b484906d3125c4d1801bd8.
Internal AI-assisted mathematical decision; not a preregistration or manuscript.

## Decision

**REJECT THIS CONSTRUCTION AS THE PAPER CANDIDATE; NO EFFICACY PILOT.**
The previously requested bounded construction is now specified and analyzed.
It does yield a strictly positive, fully charged learning benefit against
fixed-step independent TD on a nonempty class. However, that comparison does
not survive an elementary strong local learner available under the same model.
Moreover, its informative innovations are independent despite correlated
states, and the correction itself does not establish an advance over existing
personalized variance-reduction methods. No qualified successor has been found.

This is NOT a theorem that all corrected collaboration fails. In particular,
structured shared unknown dynamics and biased/adaptive shrinkage are not ruled
out. They would require a separately justified research problem, not another
experiment number attached to this failed qualification.

## 1. Observable model and full-state objective

There are two agents i,j. Agent i observes its own continuing, fully observed
two-state Markov reward process. Its kernel and public reward are

\[
P_i=\begin{pmatrix}1-p_i&p_i\\p_i&1-p_i\end{pmatrix},\quad
0<p_i<1,\quad r_i(s)=c_i(-1)^s,\quad c_i>0,\quad 0<\gamma<1.
\]

The transition probabilities are UNKNOWN to both learners. The symmetric
kernel family, discount and each recipient's reward scale are public to all
methods. No unknown reward is relabeled for free. The initial states can be
fixed; the transition flip X_(i,k)=1{S_(i,k+1) != S_(i,k)} is Bernoulli(p_i).
The flips are independent over time and across agents: this is an exact
consequence of this specified kernel/independent-innovation model, not an
assumption about arbitrary Markov TD. In stationarity, state signs have
correlation (1-2p_i)^lag. Thus correlated states here do NOT give temporally
correlated learning coefficients. This is a substantive scope restriction.

The true value is exactly V_i(s)=u_i^* (-1)^s, where

\[
a_i=1-\gamma+2\gamma p_i,\qquad u_i^*=c_i/a_i.
\]

Substitution in V_i=r_i+gamma P_i V_i proves this identity. The representation
is exact, not a projected-value substitute. With state weights (1/2,1/2),
full-state value MSE is exactly E[(u_i-u_i^*)^2]. The primary cumulative risk is

\[
J_T=\frac12\sum_{i=1}^2\sum_{t=0}^{T-1}
  \mathbb E[(u_i(t)-u_i^*)^2].
\]

Time t counts generated global environment ticks, NOT completed learning
updates. Initial waiting and calibration predictions remain in J_T. A fixed
policy generates each process; communication is training-only. This is not
partially observed or joint-action control, and not a standard MARL benchmark.

## 2. Executable protocol, cost and causal timing

Choose PUBLIC m>=1, fixed integer delivery delay D>=0, weight w, and step eta.
They are not selected using p_i, held-out outcomes or old formal data. Define

\[
\bar w=\min\{1/2,(1-\gamma)/(4\gamma)\},\quad
0\le w\le\bar w,\quad
0<\eta\le(1+\gamma+2\gamma\bar w)^{-1}.
\]

At each global tick both agents generate one transition and transmit their
flip in a timestamped/FIFO packet. Every packet, including one still pending
at the deadline, costs h+1 bits, with public h covering the actual header.
Updates are processed in birth order after D ticks; newer observations remain
buffered and are NOT used early in the value or correction. After T ticks:

\[
C_{\rm env}=2T,\quad C_{\rm msg}=2T(h+1),\quad H_T=\max(0,T-D).
\]

Both budget caps must allow these costs; the executable check rejects an
overspending tick before changing state. Independent local training consumes
the same 2T transitions and no messages, and updates immediately. It is NOT
artificially delayed or denied calibration-period samples. Message and
environment budgets are separate: unused message allowance does not purchase
extra transitions. This comparison uses an environment-binding budget ray.

For the first m DELIVERED pairs, both learners perform ordinary local TD:

\[
u_{i,k+1}=u_{i,k}+\eta[c_i-(1-\gamma+2\gamma X_{i,k})u_{i,k}],
\]

and simultaneously accumulate flip counts. The same samples serve learning
and calibration once each; there is no additional hidden probe stream. Freeze

\[
\widehat\delta_i=\frac1m\sum_{k<m}(X_{i,k}-X_{j,k}).
\]

Subsequent delivered pairs use

\[
Z_{i,k}=(1-w)X_{i,k}+w(X_{j,k}+\widehat\delta_i),\qquad
u_{i,k+1}=u_{i,k}+\eta[c_i-(1-\gamma+2\gamma Z_{i,k})u_{i,k}].
\]

Agent j uses the opposite correction. This replaces one sampled TD coefficient,
not a whole value vector. The drift term is not clipped; Z can lie outside
[0,1]. The public w/eta caps ensure every realized coefficient A=1-gamma+2gamma Z
satisfies alpha <= A <= A_max, with
alpha=(1-gamma)/2 and A_max=1+gamma+2gamma*bar_w. Therefore 0<=1-eta A<=kappa,
where kappa=1-eta alpha<1, on every path. For u_i(0)=0, all values remain in
[0,U_i], U_i=c_i/alpha. This is a pathwise stability statement, not no-harm.

Online work is O(1) arithmetic per agent per tick and O(D) buffered records
plus O(1) count/value registers for this two-agent construction. Bit width of
counts/timestamps is not claimed constant. No model, Hessian, covariance matrix
or inverse is given to the learner. The Fraction arithmetic and moment
enumeration in the audit are offline proof checks, not its implementation cost.

The proof filtration is the chronological delivered-data filtration, which
contains only pairs with earlier birth indices. The value/correction is a
function only of that history. Future flips are independent of it for THIS
model. They need not be fresh relative to the larger physical system history
at delivery. Fixed lag, birth-order processing and withholding newer data are
essential here; data-dependent delay, common noise and opportunistic reuse are
not covered by renaming this construction asynchronous.

## 3. Calibration error and the actual Lyapunov recursion

Put delta_i=p_i-p_j, Delta=hat_delta_i-delta_i and v_i=p_i(1-p_i). Directly,

\[
\mathbb E\Delta=0,\quad
\mathbb E\Delta^2=(v_i+v_j)/m,\quad
\Pr(|\Delta|\ge x)\le 2\exp(-mx^2/2).
\]

The last inequality follows by applying the bounded-sum exponential inequality
to m independent variables X_i-X_j in [-1,1]. It is fixed-m, not a confidence
sequence. The exact distribution is the difference of two independent binomial
counts. These are attainable calibration laws from the SAME paid stream.

The warm-up value u_(i,m) is correlated with Delta; it is not independent just
because future pairs are. For example, with m=2, gamma=1/2, eta=1/4, c_i=1,
p_i=1/4, p_j=3/8 and u_i(0)=0, Cov(u_(i,2),Delta)=-3/512 exactly.

Condition on the warm-up information and define

\[
a_\Delta=a_i+2\gamma w\Delta,\quad
s_A=4\gamma^2[(1-w)^2v_i+w^2v_j],\quad
b_\Delta=c_i-a_\Delta u_i^*.
\]

For e=u_i-u_i^*, the next post-calibration update has conditional mean
(1-eta a_Delta)e+eta b_Delta and conditional noise variance
eta^2 s_A(e+u_i^*)^2. Thus L(e)=e^2 has EXACT drift

\[
\begin{aligned}
\mathbb E[\Delta L\mid\mathcal G_k]
={}&[-2\eta a_\Delta+\eta^2(a_\Delta^2+s_A)]e^2\\
 &+[2\eta(1-\eta a_\Delta)b_\Delta+2\eta^2s_Au_i^*]e\\
 &+\eta^2[b_\Delta^2+s_A(u_i^*)^2].
\end{aligned}
\]

It controls full value error. Appending K*Delta^2 to L gives the SAME drift
after warm-up because Delta is frozen. It supplies no fictitious negative
calibration drift. In particular the linear e*Delta term cannot be set to
zero by citing E[Delta]=0; e depends on Delta. No SDDE approximation is needed.

For an exact finite-T calculation, retain joint unnormalized moments
(P,M,Q)=(Pr[counts], E[u*1_counts], E[u^2*1_counts]) for each count pair.
A warm-up pair with probability pi(x,y) and f=1-eta(1-gamma+2gamma x) maps it to

\[
\pi(x,y)\big(P,\ fM+\eta c_iP,\
f^2Q+2f\eta c_iM+\eta^2c_i^2P\big)
\]

in the incremented count bucket. After calibration, each bucket has fixed
hat_delta, and the recursion uses f_bar=E[f], f2_bar=E[f^2] in the same formula.
Its risk is Q-2u_i^* M+(u_i^*)^2 P, summed over buckets. Start at (1,0,0).
At resource time t use H_t processed updates, then sum over t<T. This proves
the finite-time identity by induction and retains warm-up/error covariance.
The deterministic evaluator knows p only for verification; it is NOT a
deployable risk certificate or the online algorithm.

## 4. A genuine positive result, with its exact comparator

The post-calibration recursion is uniformly contractive. Conditional on Delta,
its stationary mean is c_i/a_Delta. Solving its first and second moment equations
gives stationary full-state MSE

\[
R_i(w)=\mathbb E_\Delta\left[
  (c_i/a_\Delta-c_i/a_i)^2+
  \frac{\eta c_i^2s_A}
  {a_\Delta^2[2a_\Delta-\eta(a_\Delta^2+s_A)]}\right].
\]

The denominators are positive under the public stability caps. The expectation
is a finite binomial sum. At w=0, this is the ordinary constant-step local TD
stationary risk. Let s_0=4gamma^2 v_i. Differentiating the finite sum at zero,
the a_Delta derivative vanishes because E[Delta]=0, while s_A'(0)=-2s_0. Hence

\[
R_i'(0)=
-\frac{2s_0\eta c_i^2(2a_i-\eta a_i^2)}
 {a_i^2[2a_i-\eta(a_i^2+s_0)]^2}<0.
\]

Therefore, for any fixed interior p_i,p_j, finite m and stable eta, sufficiently
small positive w yields g_i=R_i(0)-R_i(w)>0. The same argument for both agents
gives a common nonempty interval of beneficial weights. This existence result
is NOT an observable adaptive choice of the largest useful w; the interval
depends on the unknown law. The arithmetic witness uses public w=1/100, not a
weight optimized from outcomes.

This also implies a FINITE RESOURCE-HORIZON gain, not only a limit. Couple
each stable recursion to its conditional stationary copy. Since both values
lie in [0,U_i], their risk difference after h updates is at most
2U_i^2 kappa^h. Let L=D+m. For T>L,

\[
J_{i,T}^{\rm corr}-J_{i,T}^{\rm localTD}
\le L U_i^2+\frac{4U_i^2}{1-\kappa}-g_i(T-L).
\]

Proof: the first L resource-time risks differ by at most U_i^2 each; thereafter
sum the stationary gap -g_i and both geometric transient bounds. Thus the
right side is negative at a finite T whenever g_i>0. Set budgets to at least
2T transitions and 2T(h+1) bits. Initial delay, calibration learning and pending
messages have not been erased. The guarantee is conservative; no claim is made
that its threshold is practical or that it dominates all static collaboration.

## 5. Why the positive result does not qualify the paper

### 5.1 Strong local learning removes the stationary floor

The SAME public model allows an O(1)-work/count-memory independent estimator:

\[
\widehat p_{i,t}=t^{-1}\sum_{k<t}X_{i,k},\qquad
\widehat u_{i,t}=\frac{c_i}{1-\gamma+2\gamma\widehat p_{i,t}},\quad t\ge1.
\]

This is an executable local baseline, not an oracle. Its value map is
L_i-Lipschitz on [0,1], L_i=2gamma c_i/(1-gamma)^2. Therefore

\[
\mathbb E[(\widehat u_{i,t}-u_i^*)^2]\le L_i^2v_i/t,
\qquad
J_{i,T}^{\rm localPlugin}\le (u_i^*)^2+L_i^2v_i(1+\log(T-1)),\quad T\ge2.
\]

This proves a FULL-STATE risk bound, not an inference from parameter variance
alone. The calibrated fixed-step learner has R_i(w)>0, so its cumulative risk
is R_i(w)T+O(1), whereas the plug-in learner's bound is O(log T). It eventually
loses at the same environment budget and with no additional communication.
The finite-T location of the crossover has NOT been estimated by a pilot.

This comparison rejects fixed-calibration/fixed-step correction as the desired
strong-baseline acceleration claim. It does not say that every finite horizon
favors the plug-in estimate or that changing all algorithms' steps would leave
this proof unchanged. Decreasing steps or increasing calibration windows would
be a different analyzed algorithm, not a post-hoc rescue of this one.

### 5.2 Full empirical correction can cancel the donor information

If the exact SAME window calibrates the flip means that are then pooled,

\[
(1-w)\bar X_i+w[\bar X_j+(\bar X_i-\bar X_j)]=\bar X_i.
\]

Data do not become additional target information by being counted twice. For
disjoint calibration/learning sizes m,n, let A,B be the respective means and
let Z=(B_j-B_i)-(A_j-A_i). Using ALL local samples gives
X=(m A_i+n B_i)/(m+n), and elementary covariance calculation yields

\[
\operatorname{Cov}(X,Z)=0,\qquad
\operatorname{Var}(X+wZ)=\frac{v_i}{m+n}
 +w^2(v_i+v_j)(1/m+1/n)
\]

for fixed w. This is a warning about generic unbiased mean correction, not a
full-value dominance theorem for every nonlinear or data-adaptive estimator.

More generally, for unrestricted independent p_i,p_j, the joint likelihood
factorizes. For any estimator of p_i that is unbiased for ALL parameter pairs,
the local score S_i=sum(X_i-p_i)/v_i has E[S_i^2]=T/v_i and
E[(hat_p_i-p_i)S_i]=1. Cauchy-Schwarz implies Var(hat_p_i)>=v_i/T.
Donor data do not change that bound. This elementary information calculation
does not rule out biased shrinkage, prior structure, finite-instance gains,
or aggregate Stein-type phenomena. It is not claimed as a novel lower bound.

### 5.3 Novelty and the Markov scope fail independently

AffPCL already studies personalized bias/importance correction and affinity-
dependent learning acceleration [1]. Its detailed reading boundaries from the
previous decision remain in collaboration_paper_level_sources.json; this turn
freshly verified its primary abstract/metadata, DOI and OpenAlex record. No
claim is made here to have audited its entire proof or to have reproduced its
algorithm. The generic correction mechanism above establishes no substantive
advance over that baseline. A new scalar drift identity is not sufficient.

In addition, the exact reduction to independent flips removes the hard
state-conditioned calibration issue. This reference cannot be promoted to
general delayed Markov learning merely because its raw states are correlated.
Together with the strong-local comparison, these fail the promised gate BEFORE
any new efficacy protocol or GPU experiment. These are separate reasons, not
failed experimental cells being excluded after seeing a performance curve.

## 6. Verification, scope and stop decision

The executable audit takes no true p in its online protocol, charges pending
messages, forbids over-budget ticks, and checks its finite-time moments against
all 1,024 paths of ONE declared five-tick arithmetic example. Equality is exact
rational arithmetic. It also checks calibration/warm-up covariance and the
equal-window cancellation. This is a proof check, not 1,024 new pilot samples,
an adaptation-value scan or a benchmark success.

The positive derivative/finite-horizon theorem is retained. It answers why a
basic collaboration method can beat constant-step TD without constituting a
strong research contribution. No theorem, appendix or result is relabeled as
an ICML-ready package. T-083A and every frozen failure remain unchanged.

The authorized bounded construction is complete and rejected. There is no
automatically queued successor pilot, formal study, GPU job or new experiment
number. Resuming performance work requires a NEW, substantive information/
structural premise and a mechanism that survives matched strong baselines;
the current record does not claim that such a successor has been found.

[1] Chenyu Zhang and Navid Azizan. Personalized Collaborative Learning with
Affinity-Based Variance Reduction. ICLR 2026; arXiv:2510.16232v3.
[Primary source](https://arxiv.org/abs/2510.16232).
Fresh citation checks, commands, hashes and read boundaries are recorded in
calibrated_flip_construction_execution.json. No manuscript or bibliography
is delivered in this task.
