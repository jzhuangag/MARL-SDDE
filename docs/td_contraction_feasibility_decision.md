# TD contraction certificates: a valid bound that is not yet a useful controller

2026-08-31; parent 938a7dc68587af325a1c063a56cd92ba59b1d598.
Internal AI-assisted derivation and outcome-free CPU qualification.
No scientific seed, efficacy run, new experiment number or formal protocol.

## Decision

The one-vector contraction certificate and its directional refinement have
correct fixed-law interfaces, explicit probability allocation, observable
implementations and full sensing costs. **Reject these implemented absolute
tail shields as the current low-cost algorithm proposal.** Do not respond to
this result by launching a performance pilot or increasing sample counts until
one of the existing diagnostic cells activates.

This is a scoped design decision, not a theorem that useful collaboration or
all long-horizon certificates are impossible. In the declared short-lookahead
diagnostic, removing every statistical uncertainty still leaves every action
at zero, even with a perfect donor. The deterministic tail envelope itself is
therefore an obstacle, not only sample noise. A future replacement needs a
different risk-control mechanism or a materially justified sharper envelope,
not another experiment identifier.

The paper-level problem remains useful collaboration during actual recipient
training, with a quantified cost of negative transfer. Its final mechanism
and ICML-level contribution are **not fixed or established** by this package.

## 1. Fixed-law contract and the observable Lyapunov quantity

Use the tabular fixed-policy MRP from reusable_transfer_cache_feasibility.md:
|r|<=R, B=R/(1-gamma), 0<=gamma<1, 0<eta<=1, and

\[
B_{sj}=I-\eta e_s(e_s-\gamma e_j)^\top.
\]

Both compared learners receive the same subsequent transition/reward path.
Their parameter difference x consequently evolves as x^+=B_sj x, even though
the individual parameter updates are affine and noisy. The matrices are
nonnegative and have row sums at most one. For T_m=B_(m-1)...B_0,

\[
Z_m:=\|T_m\|_\infty=\|T_m1\|_\infty\in[0,1]. \tag{1}
\]

Start z=1 and update only z_s=(1-eta)z_s+eta gamma z_j on each observed
transition. This computes Z_m with O(d+m) arithmetic and O(d) workspace,
including initialization and the final maximum. It needs neither P, V*, a
matrix inverse nor a d-by-d Gramian. One block consumes m transitions and a
reset request. Rewards do not enter this sensitivity calculation, but their
transitions are not free observations.

Fix m,n_c,delta_c before collection. Independently collect n_c fresh m-step
blocks from EVERY registered state; the registered set here is all d states.
Let bar z_s and bar z2_s be the first and second sample moments. With

\[
r_c=\sqrt{\log(2d/\delta_c)/(2n_c)},\qquad
\kappa_1=\min\{1,\max_s\bar z_s+r_c\},
\]
\[
\kappa_2=\min\{\kappa_1,\max_s\overline{z^2}_s+r_c\}, \tag{2}
\]

the one-sided bounded-variable inequality [1] and a union over 2d moments
give simultaneous bounds on max_s E_s Z_m and max_s E_s Z_m^2 with probability
at least 1-delta_c. Taking the minimum in kappa2 is valid because Z_m^2<=Z_m.
Distinct sample IDs prevent duplication, not dependence. Overlapping blocks,
adaptive stopping, omitted starting states and an unregistered changed law
are not certified by this construction.

If m<d, some state row is not updated on every path. That product row remains
the corresponding row of the identity, hence Z_m=1. More generally a long
block that leaves a state unvisited also has Z_m=1. Gamma<1 alone does not
establish finite-block contraction.

The Lyapunov function here is V(x)=||x||_infinity for the **difference between
two training trajectories**, not for the noisy learner's error relative to
V*. On the certificate event,

\[
E[V(x_{t+m})\mid\mathcal F_t]\le\kappa_1 V(x_t).
\]

This controls propagation of a transfer. It is not by itself a convergence
proof for general affine TD. The required conditional transition law given
the full history is the registered law given the current Markov state;
hidden cross-agent common noise may violate this assumption. No SDDE
approximation is used or needed for this discrete argument.

## 2. Tail risk and a structural no-activation condition

Submultiplicativity and conditioning at successive block boundaries imply

\[
E\|T_t\|_\infty^p\le\kappa_p^{\lfloor t/m\rfloor},
\quad p\in\{1,2\}. \tag{3}
\]

The successive blocks need not be independent. The uniform conditional bound
at their starting states permits iteration; an incomplete block has norm at
most one.

Let h be the actual remaining learning horizon, D the available donor
directions, beta>=0 with 1^T beta<=1, and d_j=||D[:,j]||_infinity. The executed
head estimator uses min(h,m) pre-update risks and its existing return-bias,
statistical and anchor-residual penalties. For h>m define

\[
\Phi_p(h,m)=\sum_{t=m}^{h-1}\kappa_p^{\lfloor t/m\rfloor}.
\]

The absolute omitted transfer advantage is bounded by

\[
|A_h(\beta)-A_m(\beta)|
\le\Phi_2(d^\top\beta)^2+4B\Phi_1 d^\top\beta. \tag{4}
\]

Indeed the current value error is bounded by 2B and the transferred
difference by ||T_t||_infinity d^T beta. Expanding the squared-error contrast
and applying (3) proves (4), without assuming independence of those two
quantities. Add Phi2 d d^T to the head QP matrix and 2B Phi1 d to its linear
coefficient. Both global and directional implementations execute only a
feasible negative certified upper value, otherwise zero.

Writing h=Jm+r, 0<=r<m, for 0<kappa<1 the O(1) tail calculation is
m kappa(1-kappa^(J-1))/(1-kappa)+r kappa^J when h>m. Kappa=1 gives h-m;
kappa=0 gives zero. The implementation handles these limits explicitly.

There is a stronger necessary condition than kappa1<1. For any bounded
actual query, its true m-step head advantage is at least -4Bm d^T beta.
On the head coverage event the certified head is no smaller. Therefore

\[
U_h(\beta)\ge4B(\Phi_1-m)d^\top\beta
                 +\Phi_2(d^\top\beta)^2.
\]

**If Phi1>=m, no nonzero action can have negative certified upper advantage.**
The algorithm can return zero immediately in this case. For an unbounded
remaining horizon Phi1=m kappa1/(1-kappa1), so kappa1>=1/2 triggers this
obstruction. Use Phi1, not that infinite-horizon shortcut, for finite h.
Phi1<m is only necessary, not sufficient: signal, head uncertainty and the
remaining tail can still prevent activation.

## 3. Directional refinement, with no free extra samples

The norm in (1) tracks the worst parameter direction. A few actual directions
may decay faster. Freeze the anchor basis E before the same contraction
blocks and propagate U_m=T_m E while propagating z. For every state and
column observe ||U_m[:,a]||_infinity and its square. These additional
calculations consume CPU work, not additional environmental transitions.

Allocate delta_c/2 to the global first/second moments and delta_c/2 to all
2dr directional moments. If Delta_a=||E[:,a]||_infinity, one-sided radii are
Delta_a r_E and Delta_a^2 r_E with
r_E=sqrt(log(4dr/delta_c)/(2n_c)). Denote the resulting upper moments p_s,a
and q_s,a. Valid additional caps are p_s,a<=Delta_a kappa1 and
q_s,a<=min(Delta_a^2 kappa2, Delta_a p_s,a). These are intersections of
valid upper bounds, not empirical clipping to obtain a desired answer.

For D=EA+F with nu_j=||F[:,j]||_infinity, define

\[
b_1=|A|^\top p_s+\kappa_1\nu,\qquad
b_2=|A|^\top\sqrt{q_s}+\sqrt{\kappa_2}\nu.
\]

Triangle and L2 Minkowski inequalities give E_s||T_m D beta||<=b1^T beta
and its second moment at most (b2^T beta)^2. These inequalities allow
dependent basis columns; no independence between their propagated values
is claimed. They also charge the out-of-basis residual F.

Now let Psi_p=sum_(r=0)^(h-m-1) kappa_p^floor(r/m). Conditioning after the
already propagated first block gives the directional tail bound

\[
|A_h-A_m|\le\Psi_2(b_2^\top\beta)^2+4B\Psi_1 b_1^\top\beta. \tag{5}
\]

The first term of Psi is one, not kappa: T_m was already included in the
terminal moments. Missing that term would undercharge the first omitted
risk. Add Psi2 b2 b2^T and 2B Psi1 b1 to the head QP. This remains convex.
The head cache and contraction basis/law/step/discount must match; the online
composition checks these contracts. Stored certificates cannot be altered by
mutating the returned arrays.

With head failure probability delta_G, the joint event has probability at
least 1-delta_G-delta_c. The previously derived baseline-value telescoping
then gives expected cumulative excess at most 4TB^2(delta_G+delta_c) for
T controlled updates AFTER collection and the stated fixed conditional law.
Neither variant proves benefit at an equal total sensing-plus-training budget,
nor superiority to a static graph. The optimizer is the same certified
short-unroll QP with an explicit tail, not a second algorithmic novelty.

## 4. Exact costs and diagnostic results

Separate head and contraction batches cost respectively

\[
C_{head}=d n_G[(m-1)/2+L],\quad C_{norm}=d n_c m.
\]

The full-H cache costs d n_G[(H-1)/2+L]. Thus shortening the rollout actually
saves transitions only if n_c m<n_G(H-m)/2. At equal n_c=n_G, this is exactly
**3m<H**. It is a cost identity, not equivalence of statistical accuracy.
Global plus directional contraction uses the same d n_c m transitions and
d n_c resets; the head uses 2d n_G resets. Propagating r directions adds
O(d n_c [dr+mr]) work and O(dr) per-block workspace. Donor messages and
coordinate construction remain charged as in the earlier cache reference.

The declared analytic diagnostic uses P=[[.8,.2],[.3,.7]], eta=.5,
gamma in {.2,.6,.9}, m in {2,4,8,12,16}, H in {64,512}, L=32,
n_G=n_c=128 and delta_G=delta_c=.005. These public parameters are not taken
from T-083A outcomes. Exact tree enumeration preserves within-block Markov
dependence; it does not produce a sampled certificate or scientific seed.

For a deliberately favorable falsification check, the reward is identically
zero, the declared bound remains R=1, the initial value is B(1,-1), and the
donor is the perfect zero value vector. That donor is an oracle DIAGNOSTIC,
not an allowed truth input to the online encoder or a proposed benchmark.
Transferring beta=1 would make all subsequent prediction errors zero, whereas
the local trajectory has strictly positive initial risk. A rejection here is
therefore a genuine false negative of the bound in this diagnostic.

| Diagnostic | Result |
|---|---:|
| Declared analytic cells | 30 |
| Phi1<m AND arithmetic saving | 5/30 |
| Global QP positive action, population moments plus stated radii | 0/30 |
| Global QP positive action, all uncertainty removed | 0/30 |
| Directional QP positive action, population moments plus stated radii | 0/30 |
| Directional QP positive action, all uncertainty removed | 0/30 |

The population-plus-radius evaluations are not confidence-coverage estimates.
The zero-uncertainty variants also remove return bias and use exact head
coefficients and exact contraction/terminal moments. They isolate the
absolute-tail envelope, and are not available algorithms. In the 5 cells
passing the necessary screen, the remaining tail is still too conservative.
This is why more samples alone cannot fix these declared cells.

No p-value, formal success rate, or universal impossibility conclusion is
computed. Larger m, a different objective, or a stronger risk model could
change the result; none is silently substituted to rescue this diagnostic.
All 30 rows, including costs and zero actions, are preserved in
td_contraction_qualification.json.

## 5. Research decision and next bounded work

Stop expanding this absolute-tail branch as the main algorithm proposal.
The earlier exact future-risk identity remains correct; the problem is that
the implementable upper envelopes discard too much useful sign information.
Do not present 25 passing implementation tests as 25 successful experiments.

The next question is whether cumulative negative-transfer control can use
the **executed learner versus an independently updated local shadow**, with
an explicit safe restoration action and correctly charged delayed evidence,
instead of proving each intervention's entire future advantage nonpositive.
This is a proposed change in the risk-control mechanism, not an already
established replacement theorem or ICML contribution.

Two existing baselines must be confronted first. T-071A already used a local
shadow reset and has immutable results: reset is NOT new. Conservative
Bandits [2] already imposes cumulative baseline constraints uniformly in
time: a credit ledger is NOT new either. The new package is justified only
if it closes the actual recursive-training, observable-Markov-feedback and
resource-accounting interface that those imported components do not by
themselves establish here. Do not assert that the published paper lacks a
result without reading that result.

Before proposing any run, determine the exact risk being certified (visited
state prediction risk versus full-state MSE cannot be silently exchanged),
whether labels remain conditionally valid when they arrive late or overlap,
how the last action is charged, and whether shadow restoration genuinely
restores the comparator's OWN training state. Specify initial risk allowance
explicitly: zero allowance with only worst-case positive reservations can
recreate the cold-start deadlock. A Lyapunov/credit invariant must control
this defined risk, not merely a retrospective proxy. If only generic
aggregation plus an inherited budget rule remains, reject it as insufficient
for the intended paper. Keep the research problem coherent; do not add a
new experiment number or revive a failed frozen run.

The historical readout reference in causal_collaboration_closure_decision.md
is distinct: it deliberately prevents collaborative outputs from affecting
future donor training. The next integration cannot claim recursive-training
safety merely by renaming that reference. No GPU is needed for this judgment.

## Verification and source boundary

Code: td_contraction_certificate.py; 25 deterministic tests cover exact
one-vector products, unvisited-state obstruction, conditional block bounds,
stable tail evaluation, both moments, complete state coverage, duplicate
rejection, physical charging, directional residuals and online interfaces.
The four-module dependency regression has 86 passing tests. Full regression,
source hashes, result replay and the stop decision are recorded in
td_contraction_execution_record.json. Frozen results/controllers are unchanged.

[1] Hoeffding, Probability Inequalities for Sums of Bounded Random Variables,
JASA 58(301):13-30, 1963,
[DOI record](https://doi.org/10.1080/01621459.1963.10500830).
The independent-bounded-variable ingredient is inherited; (1)-(5) are the
explicit specialization/derivation here, without a novelty-priority claim.

[2] Wu, Shariff, Lattimore and Szepesvari, Conservative Bandits,
ICML 2016, PMLR 48:1254-1262,
[official proceedings](https://proceedings.mlr.press/v48/wu16.html).
Only the cumulative-baseline constraint in its abstract is used here, not an
unread characterization of its full proofs. Fresh authoritative metadata and
resolver outcomes are in td_contraction_source_verification.json. No final
paper or bibliography is being delivered, and no acceptance claim is made.
