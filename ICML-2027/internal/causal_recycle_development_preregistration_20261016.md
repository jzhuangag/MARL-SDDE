# ICML27-CAUSAL-RECYCLE-CPU-DEV-001

## Material Passport

Prospective native complete-learning DEVELOPMENT, not confirmation, publication
admission or ICML readiness. Execution is conditional on the complete outcome-
free qualification/replay decision. Preserve all older failures and sources.
No development return exists or was inspected when this protocol was prepared.

## Research question and complete method

Does causal temporal recycling with joint Lyapunov receipt/probe control improve
finite-budget team return over strong fresh and delayed-update controls in BOTH
native MPE tasks, with exact native/message/optimizer/probe charges?
The current engine is the immutable all-method/two-task qualified source.
No additional score bank is sampled: actions and m are chosen before a training
bank and that bank becomes eligible for gradients only on later receipts.
The theory concerns exact expected team-return contrasts and a fixed-mixture
finite-menu/queue comparison along visited states. It is not a theorem of
HAPPO trajectory dominance or nonconvex gradient stationarity.

## Design, freeze and cost

- Two fixed native tasks: simple_spread_v3 (three actors) and
  simple_reference_v3 (two distinct actors). These are two tasks in ONE benchmark
  family, not broad cross-family MARL confirmation.
- Three untouched seeds: 79311001, 79311002, 79311003. No old qualification,
  A1, variance validation or TSP outcome is a training/selection label here.
- All ten methods: fresh sequential HAPPO at LR multipliers .5,1,2; immediate
  latest-gradient receipt; version-delay step adaptation; largest-gradient
  and cyclic one-actor schedulers; full method; no-dual and no-hint ablations.
  Baselines do not build unused shadow proposals or wait for score banks.
- Forty rounds, sixteen native episodes per bank, horizon25, one actor backward
  pass per actor/bank and five centralized critic steps. Equal 16000 native
  training transitions per cell; no artificial credit burning.
- Frozen independent transport ranges1--4 then1--8 rounds, hashed by per-actor
  launch ordinal independently of the scheduler clock. This models delayed
  communication, not real-world compute speedups or variable action durations.
- Full joint {0,.5,1}^N action catalogue, all m in {4,8,16}, all qualified
  controller constants unchanged. No per-task catalogue or score tuning.
- 32 independent paired evaluation episodes at frames0,8,16,24,32,40, never
  available to the learner. This prospectively chosen development cadence
  lowers evaluation-only cost; it does not alter training or controller feedback.
  Six checkpoints suffice for this bounded screen, not a polished convergence
  claim. Future confirmation needs its own precision and seed registration.
- Exact worker gradient bytes/messages, trajectory uplinks, advantage downlinks
  (weighted-advantage downlinks for sequential HAPPO), accepted-policy broadcasts,
  source/critic backprop, Adam proposals, potential fits and score/hint forwards.
  Logical tensor payloads exclude identical initialization and protocol headers;
  actual CPU duration is separately measured. No sunk work is refunded.

Accounting interpretation: `source_actor_steps` in the qualified payload counts
source backward evaluations, not worker Adam steps. Executed actor Adam calls
are `proposal_adam_steps` plus N times frames for fresh HAPPO (zero additional
source Adam calls for raw-gradient methods). `committed_adam_steps` counts
logical accepted actor updates and must NOT be added again to executed proposal
calls. Critic Adam calls are reported separately. Potential rows and policy
forward rows are exact operation counts, not claims of identical FLOP cost.
The two queues are virtual resource debts, not physical link-service backlogs.

Outcome-free qualification measured about384 active CPU seconds for twenty
six-round cells. The 60 scientific cells plus six selected replay cells contain
1,372,800 native training/evaluation transitions in total. The registered whole
workflow CPU limit is12000 seconds, checked at completed-cell boundaries; one
CPU worker, single-thread numeric libraries. This is a pre-outcome technical
budget, not an extension after a failed run.

## Frozen comparison and decision

For each task, choose the strongest NON-recycle method by mean native-transition
return AUC over all three development seeds; report every other control and both
ablations. Compute paired seed relative differences using denominator
max(abs(baseline),1). Require in EACH task:

1. Full method mean native return-AUC relative gain >=1%.
2. At least two of three paired seed AUC gains >0.
3. Mean final-return relative change >=-2%.
4. Selected strong baseline mean final-vs-initial learning gain >=2%.
5. Actual nonzero joint-version staleness in every full-method seed, common
   independent service streams, complete finite causal payloads and exact costs.

The baseline-learning requirement is fixed BEFORE outcomes, to prevent ranking
several nonlearning policies from being called successful learning evidence.
Always report total logical-communication AUC on the common observed range,
source/proposal/probe/fit cost and both ablations; native AUC is not a universal
Pareto dominance claim. Three seeds are a mechanism/development decision, not a
significance or cross-environment generalization claim.

After all sixty cells complete, replay seed79311001 on BOTH tasks for HAPPO1.0,
async_latest and the full method in one fresh root. Require semantic payload
equality except measured CPU. Run the unchanged qualified analyzer with the
prospective stage configuration, add the registered baseline-learning gate,
and require byte-identical metrics.csv, integrity.json and decision.json from
two disjoint analyses. Plot all ten methods against native transitions and
charged communication; shading is one standard error across three seeds, not
a 95% confidence interval. Curves connect actual evaluation checkpoints.

## Execution and stop

Freeze source hashes and this complete plan in a separate commit BEFORE the
scientific command. The driver refuses execution without a complete20-cell
qualification and replay pass, checks the exact qualified source lock, and
rechecks source hashes after each scientific cell. No overlapping worker or
duplicate root is allowed.

    conda run --no-capture-output -n ust2 python ICML-2027/experiments/causal_recycle_development/workflow.py --lock ICML-2027/internal/causal_recycle_development_lock_20261016.json --qualification-decision ICML-2027/results/causal_recycle_qualification_analysis_a_20261016/decision.json --output ICML-2027/results/causal_recycle_development_20261016

Any payload failure stops this execution without retry. Any mandatory scientific
failure preserves all tasks, seeds, controls, raw and replay files and stops this
frozen design. Then diagnose complete-algorithm opportunity, stale-gradient
damage, score transfer, queue overpricing or learning-signal failure before a
substantive correction. No threshold relaxation, outcome-based task deletion,
estimator-only success label, GPU/HPC4, or manuscript change is permitted here.
A pass permits a separately frozen confirmation/theory-extension decision,
not immediate ICML readiness or experimental publication admission.
