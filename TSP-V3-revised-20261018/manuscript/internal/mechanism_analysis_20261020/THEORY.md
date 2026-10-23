# Signed selection risk and when adaptation has decision value

Status: new analytical derivations and deterministic checks,2026-10-20.
Not a fresh empirical validation, priority certificate, or automatic TSP admission.
All formal scientific results/protocols remain unchanged. Main/supplement/PDF bytes
are not edited in this analysis phase. No new observations, seeds or bootstrap.

## 1. Exact identity: what the original correction actually protects

Use the manuscript's fixed-probe, two-frozen-linear-program Gaussian setting.
Under true hypothesis j, let m_j=E[C_T|H_d] have variance s_j², and write the
conditional squared risk of program a as

    E[Z_a² | H_d] = J_{j,a} + 2 d_a k_a m_j + k_a²(m_j²-s_j²).

Here J is its unconditional stationary-stream population risk with the PAID
remaining horizon; d is its initial-history coefficient; k transfers the probe-end
common state into the final estimate. These are the existing exact weights, not
new fitted quantities. Let E_j be the wrong-decision event, p_j=P_j(E_j), and let
c,w denote correct/wrong frozen programs. Exact subtraction gives

    R_j = J_{j,c} + p_j(J_{j,w}-J_{j,c})
          + 2(d_w k_w-d_c k_c) E_j[m_j 1_{E_j}]
          + (k_w²-k_c²) E_j[(m_j²-s_j²)1_{E_j}].                (1)

Thus the independent-mixture expression is generally NOT the selected risk.
The issue is statistical dependence between the decision event and continuing
learning noise, not merely the event's error probability. The old sqrt(delta)*Gamma
bound is a valid general Cauchy--Schwarz protection, but its absolute-value form
does not identify whether that dependence helps or hurts.

## 2. Proposition: symmetry removes the odd term exactly

Assume (a) the probe schedule is fixed before observations; (b) both probe laws are
centered nonsingular Gaussian; (c) the two programs and their weights are frozen
before the probe; (d) deployment uses their likelihood-ratio decision, possibly
with a fixed threshold; and (e) the conditional identity above holds.
Then E_j is invariant under changing the sign of the entire probe vector. Since
m_j is a linear function of that centered vector,

    E_j[m_j 1_{E_j}] = 0,
    R_j = J_{j,c} + p_j Delta J_j + A_j C_j,
    A_j=k_w²-k_c²,  C_j=E_j[(m_j²-s_j²)1_{E_j}].             (2)

Proof: sign reversal preserves the Gaussian density and event, and negates m_j.
Pair the two points in the expectation. This cancels the odd term, irrespective
of nonzero learner initialization d_a. Substitution into (1) proves (2).

This conclusion is not licensed for a sign-sensitive decision, adaptively stopped
probe, data-selected program weights or arbitrary non-centered hypotheses.

## 3. Proposition: an exact sign classification for ordered covariance hypotheses

Additionally suppose Sigma_1 >= Sigma_0 in Loewner order, with equal zero means.
The likelihood decision selects hypothesis1 when

    Y' (Sigma_0^{-1}-Sigma_1^{-1}) Y > log(det Sigma_1/det Sigma_0)

for the equal-prior zero-log-likelihood threshold used by the frozen study.
Under hypothesis0 the wrong event is the outside of a centered ellipsoid;
under hypothesis1 it is the inside. Then

    C_0 >= 0,                  C_1 <= 0.                       (3)

Proof: under either hypothesis whiten the Gaussian probe, then orthogonally
diagonalize its positive-semidefinite decision quadratic. Its coordinates G_i
are independent standard normal. The outside event is increasing in every G_i².
Writing m=sum_i c_i G_i, all cross products c_i c_l G_i G_l integrate to zero
by coordinate sign symmetry. For each diagonal term, conditional on the other
squares, Cov(G_i²,1_{outside})>=0 because both functions are increasing in G_i².
One elementary proof of this last fact uses an independent copy X':
2 Cov(f(X),g(X))=E[(f(X)-f(X'))(g(X)-g(X'))]>=0.
Sum with weights c_i². The inside event is the complement, giving the other sign.
No external Gaussian correlation theorem is needed.

Let a_0,a_1 be the two frozen hypothesis programs and D_k=k_1²-k_0².
Equation (2) and (3) imply this classification under BOTH true regimes:

    D_k > 0 : selection carryover is nonnegative (potentially harmful);
    D_k < 0 : selection carryover is nonpositive (beneficial);
    D_k = 0 : selection carryover is exactly zero.              (4)

Strictness additionally needs a nonconstant decision and nonzero aligned posterior
component. In the registered families the precision difference is positive
definite, lambda>0 and posterior variance>0; for the nonzero-weight differences
the sign is strict. The zero cases actually have identical frozen actions.

For the original covariance family Sigma_j=I+Q theta_j R_lambda, ordering is
immediate from theta_1>theta_0 and positive definiteness of R_lambda.

## 4. Proposition: a probability-sensitive, one-sided improvement of the bound

Let phi,Phi be the standard normal density/CDF. For 0<p<1 define

    h_+(p)=2 t_p phi(t_p),       t_p=Phi^{-1}(1-p/2),
    h_-(p)=2 a_p phi(a_p),       a_p=Phi^{-1}((1+p)/2).

For ANY event E of probability p and m~N(0,s²),

    -s² h_-(p) <= E[(m²-s²)1_E] <= s² h_+(p).                 (5)

Proof: among all events of probability p the largest integral of m² is attained
by the two tails |m|>s t_p; the smallest is attained by the central interval
|m|<=s a_p. A direct proof swaps any accepted smaller m² with any excluded larger
m², or conditions 1_E on m and optimizes a bounded acceptance function in [0,1].
Integration by parts yields the two displayed expressions. The argument also
covers events depending on additional Gaussian coordinates; it does not assume
E is measurable from m alone. The extremizers show the bounds are sharp for
this scalar moment problem, not necessarily sharp for a particular finite probe.

For p<=delta<=1/4, both h_+ and h_- are increasing on [0,delta]. Combine (2)--(5):

    U_0^sign = J_{0,a0} + delta_0[ J_{0,a1}-J_{0,a0} ]_+
               + [D_k]_+ s_0² h_+(delta_0),
    U_1^sign = J_{1,a1} + delta_1[ J_{1,a0}-J_{1,a1} ]_+
               + [D_k]_+ s_1² h_-(delta_1).                   (6)

These are valid population upper bounds under the stated ordered-Gaussian
likelihood conditions. If D_k<=0 the original independent-mixture UPPER bound
is already conservative for this special class: a positive carry allowance is
not needed. If D_k>0, the upper bound uses the correct harmful sign, rather than
charging beneficial carryover or an odd term that cancels.

For small delta, h_+(delta)=O(delta log(1/delta)) and h_-(delta)=O(delta).
This replaces the old generic sqrt(delta) penalty for this likelihood class;
it does not replace the original theorem outside that class. Limits at delta=0
are zero. This analytical refinement is not an executed new deployment policy.

## 5. Concrete necessity counterexample, without a new experiment

Let G,W be independent standard normals; take Y=sqrt(3)(rho G+sqrt(1-rho²)W).
The probe chooses the second program on E={|G|>t}. Freeze two learning programs:

    Z_0=1  (no-update program),        Z_1=(1+Y)/2 (one-observation running mean).

Both unconditional squared risks are EXACTLY1. Nevertheless,

    E[ Z_selected² ] = 1 + (3/4)rho² 2t phi(t) > 1           (7)

for rho!=0 and finite t>0. The independent mixture, even using the TRUE error
probability p=2(1-Phi(t)), incorrectly equals1. A delta*[Delta J]_+ correction
cannot repair it, because Delta J=0.

Derivation: condition on G. The conditional second moment of Z_1 is
1+(sqrt(3)/2)rho G+(3/4)rho²(G²-1). The odd term integrates to zero on E;
the remaining tail integral is 2t phi(t). At t=3,rho=1/2, p=0.002699796 and
the exact risk is1.004985829, confirmed by deterministic quadrature.
With delta=p the sign-aware upper bound in (6) attains this exact risk; its
tail-moment constant cannot be reduced for this scalar projection/event class.

This is a Gaussian Markov observation example: common variance2, private variance1
and lag coefficient3/4 give the required same-stream correlation1/2. A centered
variance likelihood has threshold t>1 by choosing a variance ratio R>1 solving
t²=R log(R)/(R-1). This proves necessity for the general frozen-linear-program
transport principle in Theorem5. It is NOT a new family from the registered grid,
not a resource-admission counterexample to the actual frozen gate, and not proof
that its hypothesis-specific selected programs or current allowance are violated.

The necessary object is correct accounting for selection dependence, not the
particular old absolute-value sqrt(delta)*Gamma penalty in every situation.
This example does not prove that omitting the term makes BOTH sides of the
registered AND-admission gate pass an unsafe action; that stronger counterexample
is not claimed. The stronger in-grid relevance is (4):44 actual public designs have the harmful
weight ordering. Their original gate margins still absorb the generic penalty.

## 6. Decision value, not blanket dominance over fixed policies

Fix a PUBLIC family, learner and budget. Let A be the ORIGINAL full-budget fixed
(q,b) catalogue, with the same deterministic learner/step and maximal affordable
update convention. A fixed action must be chosen without the unknown regime.
For an explicitly given prior pi in (0,1), define

    I_pi = min_{a in A} [(1-pi)J_0(a)+pi J_1(a)],
    O_pi = (1-pi) min_a J_0(a) + pi min_a J_1(a),
    V_pi = I_pi-O_pi >=0.                                    (8)

For a finite catalogue, V_pi>0 exactly when no catalogue action minimizes both
J_0 and J_1 simultaneously. Proof: the difference for each action is a weighted
sum of two nonnegative excess risks, zero iff both are zero; a finite minimum
is attained. This identifies the regime-only value of choosing among stationary
fixed commitments. Different hypotheses alone are insufficient to make THAT
opportunity positive. O_pi is a hypothesis-aware FIXED-program reference, not a
lower bound on arbitrary state-adaptive policies. Continuing-stream selection
may also create beneficial state-dependent carryover even when this regime-only
gap is zero; no general impossibility of adaptive improvement is inferred from
V_pi=0. The sufficient comparison in (9) remains valid in either case.

Let B_j be a valid population upper bound on the EXECUTED gate's risk under j,
including paid probe costs and selected-program carryover. On rejected designs
use the exact no-probe fallback J_j instead. Then

    G_pi = I_pi - [(1-pi)B_0+pi B_1] > 0                      (9)

certifies strict improvement over EVERY fixed catalogue action for that prior
and budget. It also beats every regime-independent random mixture of those
actions, since such a mixture cannot improve on the minimum linear objective.
This is a sufficient certificate, not a necessary condition for actual gains.
It does not compare against arbitrary stopping schedules, different steps, other
learner classes, or fixed actions chosen WITH the unknown true regime.

To separate costs, write L_j=J_post,j(a_j)-min_a J_full,j(a) and E_j=B_j-J_post,j(a_j).
Then (9) is equivalent to

    V_pi > (1-pi)(L_0+E_0)+pi(L_1+E_1).                      (10)

L is a signed opportunity difference unless monotone risk in affordable updates
has separately been proved; do not assume it is nonnegative for every affine
transient. This is the precise comparison: adaptation opportunity must outweigh
the paid remaining-horizon/decision-risk terms. It is not a new general value-of-
information principle; the potentially distinctive object is its explicit
continuing-stream, paid-learner instantiation.

A prior-weighted guarantee is not a minimax guarantee. For the latter one needs
min_a max_j J_j(a) > max_j B_j, which the existing certificates satisfy in ZERO
of128 designs. We do not promote (9) into distribution-free irreplacability.

## 7. Original-grid diagnosis and preservation

`complete/design_diagnostics.csv` preserves ALL128 designs:120 feasible probes,
8 infeasible;44 identical-program zero terms,32 beneficial terms,44 harmful terms.
Both the old and analytical sign-refined admission diagnostics admit113/128;
the new bound does NOT produce a new executed policy or new empirical effect.
All44 harmful-sign designs were already admitted by the old bound; the7 feasible
rejections are not rescued by deleting its penalty. This explains why more seeds
cannot reveal a gate-action difference on this unchanged deterministic grid.

For pi=1/2, the ORIGINAL executed-gate certificate satisfies (9) on29/128 designs:
running mean6/16 at scale512 and12/16 at2048; delayed affine2/16 at512 and9/16
at2048; neither learner at32 or128. Positive certified margins range0.1314%--
7.2032% of the best balanced fixed risk, not an average effect size. All99 other
designs are retained. `check_mechanism.py` separately enumerates every original
fixed catalogue action and verifies each sufficient inequality. Exact prior
interval calculations are retained in prior_frontier.csv; they are analytical
characterizations, not a searched new scientific grid or registered endpoint.

These are model-based, outcome-known diagnostic certificates, not29 independent
empirical discoveries. The new balanced objective does not replace the formal
regime-stratified primary endpoint. All original adverse results remain.

Existing sample table reanalysis gives gate lower mean risk than fixedq2 in11/16
learner/regime/budget strata, without claiming new significance. Averaging both
learners/regimes within scale, fixedq2-minus-gate differences are:

    scale32   -0.06674115;    scale128  -0.00179605;
    scale512  +0.00229956;    scale2048 +0.00071666.

The negative pooled fixedq2 comparison and these budget-specific positives are
compatible: small-budget absolute losses dominate that heterogeneous average.
This does not erase the pooled result, establish common-realized-cost dominance,
or justify selecting only large-budget cells as if prespecified.

## 8. What is established and what is still not established

Established by the supplied proofs: exact symmetry reduction, sign classification,
tail-moment bounds, a concrete failure of independent-mixture transport, and a
precise finite-catalogue criterion for prior-weighted irreplacability.
Deterministically checked on existing data/model:120/128 feasibility,44/32/44
sign partition, unchanged113 admissions,29 balanced certificates and0 minimax
certificates; nine low-dimensional quadrature checks agree with the moment signs.

Not established: incremental empirical benefit from the carryover penalty; every-
regime/unknown-prior or unknown-kernel dominance; non-Gaussian/nonlinear extension;
publication-level novelty of the refined theorem relative to all adjacent work.
The last item requires claim-level literature work before manuscript admission.
The old six-work comparison cannot certify absence across statistical decision
theory. No additional scientific run is presently requested or authorized here.
