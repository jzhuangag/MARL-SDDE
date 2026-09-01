# Autonomous CPU research plan — 2026-09-01

## Theorem-first replacement after scheduler failure

Read `clocked_async_mpg_mainline_decision.md`.  The project now stops creating
online scheduler variants.  The only active replacement candidate is a
finite-time wall-clock analysis of event-driven distinct-agent policy learning
in a Markov potential game.  Its intended Lyapunov--Krasovskii functional must
jointly handle potential gap, critic error and delayed block-update history.
The contribution is conditional on a genuinely new interaction-delay theorem;
a generic delay-adaptive step size is explicitly insufficient.

No scientific trajectory, seed registry, formal experiment or GPU benchmark
is authorized.  First close or kill four interfaces: delayed Markov-gradient
decomposition, one-event drift, finite-time game criterion and an analytically
nonempty fully charged wall-clock speedup region.  If the result is a direct
composition of existing delayed SA and synchronous MPG theory, stop it before
experiments.  All previous failures remain unchanged.

## Latest bounded decision: compatible-update scheduling stopped

Read `compatible_schedule_headroom_validation.md` before proposing another
asynchronous MARL scheduler.  The frozen 48-scenario deterministic ceiling is
complete and byte-exact reproducible.  Compatible parallel service achieves a
7.6147x geometric completed-update ratio over sequential service, but the
paper-critical comparison against a per-scenario strong compatible scheduler
fails: geometric queue-cost ratio 1.02676, only 18.75% of cells obtain 10%
improvement, and 20/48 cells are worse.  Three of six mandatory gates fail.

Stop `compatible_maxweight` before the stochastic potential-game pilot,
formal seeds or GPU benchmark.  Preserve the smoothness/independent-set and
queue-drift algebra as structural tools, not efficacy evidence.  Do not rescue
the candidate with a new potential weight, workload or relaxed gate.  The next
paper-level decision must identify a substantively different endogenous
learning constraint or information structure and pass novelty plus oracle-value
screens before a new efficacy protocol.  No stochastic or GPU run is queued.

## Latest bounded decision: Policy Backpressure construction stopped

Read `policy_backpressure_feasibility_decision.md` before proposing an
asynchronous cooperative-MARL successor.  A substantive new premise was tested:
agent-wise asynchronous policy proposals in a common-payoff Markov game, with
joint-policy path length as interaction staleness.  The stale unilateral-policy
performance lower bound passed 1,080 exact checks, but the 144-cell deterministic
development scan found no material dynamic scheduling value against a per-cell
best fixed trust radius and fresh serial learner.  Median non-myopic wall-clock
headroom was -1.3834%, the maximum was 2.2939%, and zero cells reached 5%.
The safe path-bound rule had no harmful update but was 47.78% below the strong
endpoint baseline at the median.

Stop this construction before a debt-queue theorem, sampled controller, formal
seed or GPU benchmark.  Do not rescue it by changing the completed grid,
threshold or metric.  This does not prove all asynchronous MARL impossible; a
future direction must supply a different intrinsic state or constraint and pass
its own bounded novelty and oracle-value screen.  T-083A and every prior result
remain unchanged.

## Current decision: bounded calibrated-sharing construction rejected; no pilot

Read calibrated_flip_construction_decision.md FIRST. The one concrete successor
construction requested below is now COMPLETE, and is rejected as the paper
candidate. Its fully charged, delayed two-state-MRP protocol has an exact
full-value Lyapunov/moment analysis and a nonempty gain region against
constant-step local TD. A same-information, O(1)-work local plug-in learner,
however, has O(log T) cumulative risk while this fixed-calibration/fixed-step
construction retains a linear risk floor. Its informative flip sequence is
iid despite correlated raw states; it does not solve the hard general Markov
calibration problem. Independent novelty against AffPCL is unqualified.

No new performance experiment, formal seed or automatically selected successor
is queued. This closes the bounded construction gate negatively, not the
entire possibility of collaborative learning. After completing verification
and writing the decision record, the hourly sdde-cpu follow-up was PAUSED at
this stop boundary. Resuming requires a substantive new information/structure premise
and research decision, not another weight/window sweep of this construction.
The original goal and every prior result remain preserved; no ICML-ready
algorithm or manuscript is claimed.

## Prior paper-level decision and preserved reference branches

Read collaboration_paper_level_decision.md for the preceding decision. It is
COMPLETE: the current generic safety-first parameter-mixing architecture is
not the final mainline. The selected research candidate is calibration-limited
collaboration: learn recipient-corrected update information from the same
delayed Markov stream and require a net benefit after calibration cost.
It remains a candidate, NOT an established new algorithm. In particular,
AffPCL (ICLR 2026) is a newly verified close baseline with personalized
correction, a TD application and its own Lyapunov analysis. A routine
Markov/delay extension or generic control variates do not pass novelty.
The original full-state risk goal remains. Exact all-prefix no-harm is no
longer a mandatory prospective claim; finite-time risk with an explicit
calibration penalty is the proposed replacement. This changes NO old gate.
The new decision supersedes the method ambition in collaboration_paper_thesis.md;
the latter, rl_collaboration_integration_decision.md and
causal_collaboration_closure_decision.md remain historical references.
The completed observable-estimator feasibility decision is now recorded in
observable_transfer_feasibility.md: the direct coupled-probe implementation
is a valid reset-access reference but is rejected as the inexpensive final
method. No efficacy run is authorized by that qualification.
The subsequent reusable_transfer_cache_feasibility.md closes fixed-law
parameter/direction/horizon reuse with explicit residual errors. Retain it as
a reference, not the final low-cost mechanism. Its up-front collection cost
and same-total-budget comparison remain unresolved; no pilot is authorized.
The bounded contraction implementation/qualification is COMPLETE in
td_contraction_feasibility_decision.md. Both global and direction-specific
absolute-tail shields return zero in all 30 declared perfect-donor analytic
cells, even with statistical uncertainty removed. Reject them as the current
inexpensive algorithm candidate. A valid contraction certificate alone is not
a useful collaboration controller; more samples do not resolve this example.
The subsequent delayed-training-risk interface is COMPLETE in
delayed_training_risk_decision.md. A fixed-law continuing-stream reference now
controls cumulative PRE-UPDATE VISITED-STATE risk with actual shadow restoration,
fully reserved delayed labels and a chronological martingale argument. Reject
the generic ledger as the final algorithm: it does not certify full-state MSE,
does not establish reflected-debt/Lyapunov convergence or useful learning gain,
and budget/rollback/delay primitives already have close prior counterparts.
No new efficacy protocol is authorized by this narrower reference theorem.
This integration plan supersedes the chronological audit-only to-do list;
complete historical versions remain in Git.

Stop treating the frozen post-hoc debt penalty as a proved safety controller.
The independent-local-bank, predictable-readout reference has an exact
post-update contrast identity, finite-time comparison, composite Lyapunov
learning bound and realized all-prefix allowance under explicit known-mixing
scalar AR assumptions. The integration gate is now complete: retain that
reference as a baseline, reject readout alone as the final candidate. Direct
TD-target substitution is biased; same-history graph comparisons cannot
certify counterfactual training trajectories.

The selected problem is useful transfer during personalized TD training,
judged by its effect on subsequent learning risk relative to actual local
training. Exact Markov-jump Lyapunov risk recursions and baseline advantage
telescoping close the known-model/oracle contract. They do NOT supply a
model-free, low-complexity controller or a new performance-difference lemma.

## Completed bounded integration package and conditional future stages

The user requires one coherent problem-mechanism-theorem-evidence narrative.
The candidate contribution is net personalized learning acceleration after
learning the correction required to reuse heterogeneous data. Its proposed
Lyapunov risk bound and calibration cost must explain that same mechanism.
The new nearest-method/source record is collaboration_paper_level_sources.json;
earlier graph/meta-learning comparisons remain historical. Neither novelty
nor an end-to-end affordable learning benefit has yet been established.

1. The direct-probe branch has been resolved: exact observed-data coefficients,
   finite-return bias, fixed-n uniform confidence, scalar/joint robust QP and
   full cost are derived and implemented. Its O(n H k) tabular propagation
   and conditional reset access do not solve inexpensive long-horizon credit.
   A projection-only TD recursion also fails due to off-subspace leakage.
   Do not rerun this branch seeking a passing performance result.
   That fixed-law reuse question is now mathematically resolved: one
   precollected all-prefix anchor cache supports changing v,D,s,h under a
   joint coverage event, with explicit e/F residual penalties and an actual
   local-trajectory finite-T guarantee AFTER collection. It does not justify
   nonstationary-law reuse, unobserved cross-agent common noise, or giving
   the baseline fewer resources. Caching alone is rejected as a sufficient
   final mechanism; do not run it against only an uncached weak comparator.
   The one-vector and directional contraction package is now resolved too:
   first/second-moment bounds, Markov block iteration, convex head-plus-tail
   QP, residuals and full costs are implemented and derived. The necessary
   activation-and-cost screen passes 5/30 cells, but both finite-radius and
   zero-uncertainty perfect-donor QPs activate in 0/30. This is a deterministic
   diagnostic, not a sampled certificate or efficacy experiment. Preserve all
   rows. Do not rerun the declared grid looking for a pass or claim that this
   proves impossibility of every long-horizon estimator. The absolute-tail
   design is also the matched certified short-unroll baseline, not a distinct
   algorithmic novelty. No new pilot is authorized by this package.
   The executed-risk/credit question has now been resolved at its attainable
   scope. T-071A/T-072 source and immutable outcomes were read; shadow rollback
   and dual-use data are already present. Conservative UCB's algorithm and
   safety proof, and Prudent-Banker's displayed delay/baseline mechanism were
   checked. Do not claim generic credit, rollback or delayed safety as new.
   The new reference freezes actual predictions before rewards, colors
   overlapping finite returns, retains previous valid certificates when noisy
   feedback arrives, and reserves every pending/final action. Its scalar
   admission is closed form. It is not a general affine-TD Lyapunov proof.
   Deterministic counterexamples establish that visited risk does not imply
   full-state MSE safety and cumulative credit does not bound reflected debt.
   These are design boundaries, not a new negative-results paper mainline.
   The paper-level decision is now COMPLETE, not another pending memo.
   The requested finite-MRP construction is now COMPLETE in
   calibrated_flip_construction_decision.md. Observable updates, fixed-lag
   birth-order timing, fully charged dual-use calibration, its exact error
   law, the real value-error Lyapunov drift and finite-resource-horizon gain
   against constant-step TD are specified. A strong local O(log T) cumulative
   risk bound rejects that comparator-limited gain as the desired mainline.
   Same-window empirical correction cancels donor information; disjoint-window
   unbiased correction does not evade full-data local variance accounting.
   These are scoped mathematical results, not an impossibility theorem for
   every adaptive collaboration mechanism. No qualified successor was found.
   Do not enqueue a new wrapper or experiment identifier. No formal-data tuning.
2. Only if this replacement interface AND novelty/feasibility pass: freeze ONE CPU
   development protocol, its source/public config, fresh development seeds,
   accounting and decision rules. Include actual local and strong static
   training histories, AffPCL/corrected-update and other matched baselines,
   nonzero temporal correlation, heterogeneity/change, delay, safety cost
   and scaling. A guarantee relative to local training is not automatically
   best-static-graph matching. Tests are not an efficacy gate. Preserve failures.
3. Only if unchanged development succeeds: freeze final implementation and
   unused independent confirmation seeds. Preserve all comparison scopes.
   Do not use T-083A formal outcomes for tuning or recycle them as evidence.
4. Then design matched standard RL validation, requesting GPU/HPC4 handoff
   only if necessary. Write a unified paper and proof appendix around the
   verified mechanism, not the historical participation manuscript.

No efficacy preregistration or scientific trajectory is authorized by the
reference qualification alone. No new experiment number bypasses failure.

## Completed integration qualification

- Earlier reference: causal timing, restricted finite-time Lyapunov and
  all-prefix allowance; 24 tests. Its controller and provenance are unchanged.
- New interface decision: ordinary-TD bias, recursive-comparator mismatch,
  current versus future value-risk distinction, exact Markov-jump risk metric
  and full-trajectory oracle advantage identity; 20 deterministic tests.
- Positive oracle qualification is conditional on privileged model/error
  access. It is not an observable controller or benchmark efficacy result.
- Earlier interface regression: 769 passed, 7 skipped, 108.86 seconds.
- Exact audit CLI output reproduced byte for byte in two executions.
- Four source records cross-checked with primary records, DOI redirects and
  OpenAlex; Semantic Scholar returned 429. See the new source-verification
  JSON for version boundaries. No final bibliography or novelty claim.
- No sampled scientific trajectory, old formal regeneration, GPU or remote
  storage operation. No new efficacy protocol or formal seeds registered.
- Observable reference qualification: 20 additional deterministic tests;
  then-current full regression 789 passed, 7 skipped in 117.95 seconds. Exact CLI
  replay is byte-identical. General QP has a stated optimization error bound;
  confidence is fixed-n and conditional on fresh reset-access replicates.
- Known-model moment identity now has a correct observable, fully charged
  reference estimator. It still lacks the computational advantage required
  by the paper thesis, and same-dual-budget no-harm is not proved.
- Reusable cache: all-state/all-prefix normalization, exact affine reuse,
  out-of-anchor residual and convex robust QP are derived and implemented.
  Its global once-built-cache event supports a conditional fixed-law
  expected cumulative comparison, not per-query conditional coverage.
  Cached short unrolling is explicitly included as a reference; its omitted
  tail is charged. Current qualification uses deterministic enumeration only.
  Full current test/replay results and source hashes are recorded in
  reusable_transfer_cache_execution_record.json.
- Contraction/tail package: 25 deterministic tests, 86 dependency tests;
  full final-source regression 835 passed, 7 skipped in 107.82 seconds.
  Exact qualification replay is byte-identical; all 30 rows are preserved.
  Reject the implemented absolute-tail controller, not the fixed-law moment
  identities. Its Lyapunov quantity controls a difference of coupled training
  trajectories; it is not a general affine-TD convergence theorem. See
  td_contraction_execution_record.json for commands, JUnit and SHA-256 records.
- Delayed actual-risk ledger: 19 deterministic tests; correct fixed-law
  continuing-stream visited-risk allowance, explicit tail-label reservations,
  one-use accounting and cold-start condition. Full-state MSE and reflected
  queue implications both have explicit counterexamples. Retain as a reference,
  reject as a sufficient paper mechanism. Full regression before the last
  numerical guard: 853 passed, 7 skipped in 746.96 seconds. The bounded
  rounding guard and its nineteenth test were subsequently verified by 105
  final-source dependency tests and a 19-test standalone run; do not label
  the earlier full regression as testing this last change. Two final CLI
  executions are byte-identical. Commands, version boundaries and hashes
  are recorded in delayed_training_risk_execution.json.
- Paper-level decision: retain personalized full-state learning, reject a
  generic exact-safe mixing wrapper as the final algorithm, and select the
  calibration-cost construction for one bounded feasibility decision. Three
  primary sources were freshly cross-checked, including AffPCL as a close
  personalized/Lyapunov baseline. No completed novelty or algorithm theorem
  is claimed. Final-source full regression now closes the previous numerical-
  guard version gap: 854 passed, 7 skipped in 752.63 seconds. No experimental
  source changed in this decision turn. Exact scalar drift/covariance checks
  are algebra, not new efficacy evidence. See collaboration_paper_level_decision.md,
  collaboration_paper_level_sources.json and collaboration_paper_level_execution.json.
- Calibrated flip construction: complete observable two-agent reference,
  exact full-value moments including warm-up/calibration dependence, fixed-lag
  resource clock, and strictly positive finite-horizon gain against fixed-step
  local TD. Strong local learning and the iid-innovation reduction reject it
  as the paper candidate. One exact rational audit checks 1,024 enumerated
  five-tick paths, not pilot seeds. Two normalized stdout replays are identical;
  independent variance/derivative checks pass. Full regression: 854 passed,
  7 artifact-dependent tests skipped, 729.07 seconds. The standalone proof
  audit runs separately; it is not counted as an extra pytest test. See
  calibrated_flip_construction_execution.json. No qualified successor or
  performance protocol is automatically queued at this stop boundary.

## Non-negotiable preserved results

T-083A endpoints/cells, original/reproduction summaries and frozen gates
remain untouched. Reproduction F12 timeout and strict F13 summary mismatch
retain FAIL. Primary temporal correlation was zero; stationary versus local
had a cost. Continuous-AR fixed-baseline covariance omission and duplicate
NoiseCertificate evidence remain documented, not silently repaired.

Audit trail: validation_t083a_formal_end_block_confirmation.md,
end_block_certificate_audit_20260831.md,
end_block_debt_theory_audit_20260831.md,
markov_baseline_law_and_target_coverage_audit.md,
fixed_end_block_moment_integration_audit.md,
shield_activation_scan_validation.md,
paired_innovation_risk_certificate_audit.md.

## Execution rules

Hourly heartbeat sdde-cpu is periodic continuation, not a continuously
running experiment. Check Git/processes/logs before CPU work; never duplicate
a running job. Record commands, hashes, outcomes and stop decisions, then
commit/push reviewed changes. No GPU/HPC4, remote storage operation or
destructive cleanup is authorized. No guarantee of a one-day finish,
positive experiments or acceptance.

## 2026-09-01 asynchronous-MARL successor checkpoint

The generic calibrated-collaboration construction remains stopped.  The only
active successor is now the distinct-agent asynchronous CTDE problem in
`asynchronous_policy_update_backpressure_scope.md`:

- unilateral agent-policy proposals are perishable because other agents'
  updates change the joint occupancy and their gradients;
- cross-agent freshness debt is the single state variable linking the problem,
  Lyapunov drift, online admission, step size and ready-proposal ordering;
- the universal 144-cell value screen failed unchanged at `7.2183%` median and
  `54.1667%` directional coverage, so universal superiority remains stopped;
- an outcome-independent phase theory now proves low-load static
  near-optimality, a high-load `O(1)` versus `Omega(M)` wall-clock separation,
  one-event descent and conditional finite-time full-gradient stationarity;
- 14 current package tests pass, including 405 capped/unclipped event-algebra
  instances and exact separation witnesses; these are algebra, not efficacy;
- a phase-conditioned disjoint CPU study is only conditionally allowed after
  the Stage-1 checkpoint.  It must implement the closed-form controller and
  may not use favorable old cells as confirmation data;
- no GPU/HPC4, standard MARL benchmark or paper efficacy claim is authorized.

See `policy_update_backpressure_stage1_checkpoint.md` and
`perishable_update_phase_theory.md`.  The next unresolved paper risk is an
executable conservative cross-sensitivity certificate for neural policies;
the episodic bounded-trajectory radius is proved, while continuing-chain and
critic/off-policy terms remain explicit obligations.

## 2026-09-01 policy-inventory successor checkpoint

The phase PUB pilot and the unconditional cross-agent HVP-transport headroom
scan have now both stopped under their frozen gates.  The latter completed all
17,820 deterministic trajectories but passed only 2/7 gates; full-HVP high
phase regret was 1.3524 times the strong comparator.  It does not authorize a
stochastic or GPU experiment.

The only active theory candidate is now `policy_inventory_control_mainline.md`.
It treats paid in-flight joint rollouts as perishable policy inventory.  The
factorized joint importance-ratio second moment is the physical risk state; a
current distinct-agent update improves the policy while changing the future
variance of every outstanding rollout.  A composite Lyapunov function prices
that causal externality and yields a continuous scalar update in `O(P)` work,
with no HVP, preconditioner, finite step catalogue or participation scan.

Exact Gaussian factorization, RMS risk scaling, post-step inventory geometry,
convexity and scalar optimization currently pass outcome-free algebra tests.
The event-time completed-risk bound is conditional; the general Markov
trajectory interface, nonvacuous full-gradient conversion, wall-clock theorem
and fixed-controller separation remain open.  No efficacy pilot is authorized
until both the proof interface and a separately frozen outcome-free oracle
headroom gate pass.

### Policy-inventory headroom decision

The separately frozen outcome-free headroom scan at commit `31ed16c` completed
155,520 evaluations but failed 2/6 mandatory gates.  The exact continuous
action improves certified utility over the strong ray-level causal baseline by
only 1.5039% in aggregate, and only 4.1667% of rays reach 5%.  Structural
validity, low-load recovery, nonzero high-load updates and pointwise dominance
pass.  The stochastic pilot, formal study and GPU benchmark are forbidden.

Accordingly, the policy-inventory controller is no longer an active positive
mainline.  Its exact risk identities remain valid supporting theory.  No
further controller variant is authorized merely by changing a step, graph,
participation, transport or shrinkage parameter.  A new successor must change
the task-level research premise and demonstrate paper-level oracle value before
any sampled experiment.

## 2026-09-01 compatible-update scheduling gate

A substantively different premise is now frozen for an outcome-free gate in
`compatible_async_updates_mainline.md`.  HAPPO/HATRPO/HAML obtain monotonic
improvement through sequential heterogeneous-actor policy updates; the new
question is whether a sparse certified interaction graph permits compatible
ready actor updates to be committed concurrently, while Lyapunov MaxWeight
scheduling removes the sequential barrier under heterogeneous completion.

The current algebra verifies additive gain on independent blocks, a harmful
pair of individually beneficial conflicting updates, exact path
maximum-weight independent-set scheduling and queue-drift bookkeeping.  The
pending static ceiling must beat both sequential service and a strong
best-color compatible scheduler.  No stochastic learner or GPU benchmark is
authorized unless every frozen scheduling gate and the later Markov causal-cone
theorem pass.
