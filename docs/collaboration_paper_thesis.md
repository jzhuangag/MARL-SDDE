# One paper, one problem: credit assignment for collaborative training

2026-08-31. Internal AI-assisted research design, based on 87a1dd9.
This is a prospective contribution contract, not a manuscript, preregistration,
new theorem claim, or declaration that the current method is ICML-ready.

## Decision and unifying question

Retain the research problem; do not freeze the current oracle construction
as the final method. The paper should answer one question:

> Can learners cheaply determine which transfers will improve their subsequent
> training, and use that determination to obtain useful collaboration with a
> quantified excess-risk cost relative to genuinely independent training?

The prospective thesis is **long-horizon credit assignment for collaboration**.
A transfer changes the recipient's future training trajectory. Its usefulness
depends on how the recipient learns, not just on the donor's current accuracy,
parameter similarity, or the immediate error reduction.

The proposed technical route is to compress those future consequences into
data-estimable, direction-specific Lyapunov quantities. The contribution, if
achieved, is the estimator-controller-guarantee as one mechanism. A graph, QP,
delay term, confidence bound and convergence lemma are not five contributions.

## Why this is a substantive multi-learner problem

The initial formal setting is collaborative fixed-policy learning: each learner
has its own Markov reward process and personalized value estimate. Donor models
must have a declared common representation or a fixed alignment; arbitrary
neural parameters from unrelated tasks are not automatically transferable.

Local training takes place at learners. A coordinator can route timestamped
donor messages; the recipient assesses a proposed transfer with its own data.
The initial claim concerns training communication, not online communication
between acting policies. This is not yet a joint-action cooperative MARL or
partially observable control theorem. Benchmarks and title must respect that
distinction; policy-control extensions require their own evidence.

The hypothesized transfer score is directed and state dependent: j helping i
need not imply i helping j, and the same donor can have different value at
different stages of the recipient's learning. Heterogeneity specifies whose
target matters; Markov dynamics specify how an intervention propagates; delay
specifies which donor information is actually available. These are mechanisms
inside one problem, not independent embellishments. Nonstationarity is not an
additional mandatory axis before the stationary heterogeneous case is solved.

## Closest work: what cannot be claimed as new

The following is a bounded confrontation, not an exhaustive literature review.
These are positive descriptions of verified sources, not claims that every
other method is myopic or lacks theory.

| Source | Established component | Consequence for our contribution |
|---|---|---|
| [FedFomo](https://arxiv.org/abs/2012.08565) | Client-specific weighted model combinations based on benefit to the recipient | Choosing who helps whom and learning aggregation weights are not new alone. |
| [pFedGraph](https://proceedings.mlr.press/v202/ye23b.html) | Collaboration graphs inferred using model similarity and dataset size | Learning a graph is not the main claim; compare against graph-based collaboration. |
| [FedAGHN](https://arxiv.org/abs/2501.16379) | Dynamic client-specific collaboration through attentive graph hypernetworks | Even stage-dependent collaboration is already an explicit objective. |
| [Meta-Gradient RL](https://arxiv.org/abs/1805.09801) | Online adaptation of RL return parameters through meta-gradients | Differentiating learning to improve learning is inherited, not our invention. |
| [Learning to Reweight Examples](https://proceedings.mlr.press/v80/ren18a.html) | Online weight updates through a training step and validation objective | A one-step lookahead weighting method is not enough to distinguish our method. |

The sought distinction is a **reliable long-horizon transfer estimate without
expensive long-horizon branching/unrolling**, under the actual sampled training
law. Neither that capability nor its novelty has yet been established here.
Full closest-method reading and matched implementations are required before
claiming an advantage over these families. Their existing learning guarantees
must not be dismissed merely because they use different terminology.

## The single mathematical object

Let Z be the full training state, including parameters, Markov states and any
delay buffers. Let J_h^loc(Z) denote expected personalized learning risk over
the remaining h training steps under independent local training from Z.
For an available transfer d, define its baseline advantage

\[
A_h(Z,d)=J_h^{\mathrm{loc}}(R_dZ)-J_h^{\mathrm{loc}}(Z).
\]

The existing note proves the exact full-trajectory telescoping relation for
this quantity with matched fixed-horizon training kernels. That identity is an
inherited Bellman argument. It is the interface for a theorem, not the novelty.

For known affine TD dynamics, the baseline value is quadratic in parameters v:
v^T Q v+2p^T v+c. With k available donor directions as columns of D and transfer
coefficients beta, the relevant advantage is

\[
A_h(\beta)=\beta^\top G_h\beta+2g_h^\top\beta,
\qquad G_h=D^\top Q_hD,\quad g_h=D^\top(Q_hv+p_h).
\]

Here Q is a finite-horizon stochastic Lyapunov risk metric determined by local
learning dynamics, not an arbitrary preconditioner. The ambition is to estimate
the k-dimensional projected quantities rather than construct a full Gramian.
Solving their QP is routine once they are available; estimating and certifying
them cheaply is the unresolved central step. No O(kd) claim follows from
writing the projected formula: forming it, estimating it and solving it must
all be included in complexity and sample accounting.

SDDE is optional. If a discrete Markov-jump argument directly establishes the
required result, adding a diffusion approximation without a controlled error
would weaken rather than deepen the paper. Lyapunov is essential only if it
actually enables the long-horizon compression and its guarantee.

## One theory chain and one evidence chain

The intended theorem chain is:

1. Derive an observable projected-advantage estimate with a valid finite-data
   error bound under specified Markov sampling, target estimation and delay.
2. Propagate that estimation error through the executed controller to cumulative
   excess risk against the baseline's own training trajectory, including the
   probability and magnitude of certificate failure.
3. Establish a nontrivial benefit condition after sensing/communication costs,
   and show when that condition can be detected at an affordable cost.

The elementary observation that a uniformly accurate estimate within epsilon
can detect a sufficiently separated negative advantage is not a new theorem.
The work is deriving an attainable epsilon from the permitted observations,
not assuming it. Aggregate risk control also does not imply per-agent safety;
the final statement must choose one and meet that scope explicitly.

The existing fixed-horizon identity does not yet establish a dual-budget
guarantee. If sensing changes available updates or clock time, remaining
resources enter the baseline state/value or the comparison is reformulated.

| Paper question | Decisive evidence, once a controller exists |
|---|---|
| Does recipient learning dynamics change the correct collaboration decision? | Analytically controlled Markov tasks varying propagation modes while holding immediate similarity/error cues fixed; compare current-risk and future-risk decisions. |
| Can an observable estimator recover useful long-horizon information? | Accuracy, calibration and activation versus horizon, correlation, delay and data cost, with truth used for evaluation only; oracle is a diagnostic, not the main method. |
| Does that information improve real learning at matched cost? | Independent trajectories for local, strong static graph, immediate-benefit and short-unroll/meta-gradient collaboration baselines; evaluate cumulative risk, endpoint risk and resource costs. |
| Is the mechanism useful beyond the analytic construction? | A predeclared standard RL-derived task family matching the claim, independent confirmation seeds and informative component ablations. |

This table is an evidence design, not a frozen experiment grid. Numeric gates,
budgets and populations must be specified before outcome collection. Main
figures should build this causal argument, not list experiment identifiers.
Appendix proofs use exactly the assumptions and algorithm executed in the
main experiments; implementation history belongs in internal records.

## Keep or change the idea: author decision criteria, not venue rules

Continue this candidate only if the same implementable mechanism satisfies
all three scientific obligations: a distinction beyond ordinary reweighting
or short unrolling; a finite-data training-risk guarantee; and meaningful
benefit with the estimator's cost included. This is not a prediction of review
scores or an official ICML 2027 acceptance criterion.

Change the core mechanism if the estimator needs privileged true values/full
models, if the certificate suppresses every useful transfer, or if a matched
simple/meta-gradient baseline supplies the same benefit and guarantee at
comparable cost. A mechanism counterexample without an attainable algorithm is
not sufficient for the intended algorithm paper. Do not rescue such a result
by adding more queue variables, a new experiment number, unsupported SDDE
claims or post-hoc favorable task selection.

The next deliverable is ONE feasibility package: a specified estimator and its
assumptions, a controller using only allowed observations, a derived error/cost
bound, and a bounded CPU qualification. Give a concrete pass/reject decision;
do not replace this deliverable with an expanding list of small audits.

## Current evidence and claim boundary

At 87a1dd9, exact Markov-TD oracle calculations support a mismatch between
immediate and future error and qualify the known-model risk identity.
They do not establish the prospective estimator or standard-RL superiority.
The previously reported 769 passing tests and 7 skips are implementation
regression evidence, not 769 successful experiments. Frozen T-083A outcomes
and every historical stop decision remain unchanged and are not tuning data.

Source checks, unresolved resolver availability and version distinctions are
recorded in collaboration_paper_thesis_sources.json. Bibliography metadata are
verified at the indicated source/version; no final paper bibliography is being
delivered and no unseen full-text proof is characterized as audited.
