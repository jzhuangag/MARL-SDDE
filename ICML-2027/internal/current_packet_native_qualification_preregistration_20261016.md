# Native current-packet integration: outcome-free operational preregistration

## Material Passport

ARS experiment-agent/run; prospective operational qualification, not scientific
performance development or confirmation. No new native rollout has been run
before this source/plan freeze. Original A1, temporal-recycling failures, all
old seeds/gates/roots, and TSP manuscripts remain unchanged.

## Purpose and exact change

Qualify a complete native MARL loop, not a standalone variance estimator. The
new source is `experiments/current_packet_native/`. The candidate update is five
full-batch epochs of the pinned HARL HAPPO actor's actual `update` method, with
new loss and backpropagation at each candidate parameter state. Server policy
and Adam moments are cloned before each proposal; only the chosen full server
state is committed. Source-worker optimizer states are never imported.

This requires a declared PAYLOAD CHANGE: delayed immutable local trajectory,
source-log-probability and advantage records replace delayed raw gradients.
There is no gradient payload in this adapter: source backward rows, gradient
bytes and gradient messages are exactly zero and remain reported as such.
Server proposal backward rows and optimizer calls are explicitly charged.
Do not call repeated Adam steps on one stored gradient multi-epoch PPO.

Distinct actors remain CTDE: public centralized state/reward supports a common
critic; actors use local observations. Asynchronous means logical transport
receipt of actor-specific update programs, with fixed per-actor launch ordinal
service streams. This is not measured concurrent compute-worker execution or a
hardware speedup claim. Pending programs are fully reserved at launch, including
ones superseded at receipt or still in transit at the terminal frame.

## Causal protocol and controls

Before each bank, receive all currently due programs and retain the newest for
each actor. Freeze server-owned candidates and unique joint-policy hashes,
potential from past public banks, full cost catalogue and probe choice. The
current_value method then opens its bank, receives a paid early probe slice of
private actor records, forms unclipped actual-return contrasts, chooses its
CURRENT eligible joint update and commits it. The bank generates only future
programs, with its captured precommit joint policy/critic versions.

Three methods, on BOTH native tasks, share initialization and sampling roles:

- fresh_happo: strong ideal zero-delay anchor; five epochs per actor, deterministic
  actor order, multiplicative sequential HAPPO factor; fresh full bank.
- eligible_all: matched eligible delayed-data PPO reference; the same post-bank
  receipt timing, all latest-ready programs, unit learning-rate multiplier,
  server-owned five-epoch proposals; no current-return scoring.
- current_value: full eligible joint catalogue with alpha in {0,.5,1,2} per ready
  actor, exact policy deduplication, adaptive probe m in {4,8,16}, and joint
  communication/computation Lyapunov prices. No task-specific action removal.

The delayed adapter uses local PPO factor=1 rather than pretending that actors
whose latest programs come from different banks share a valid HAPPO sequential
prefix. Actual-return importance scoring treats their combined candidate as a
whole policy and does not require unbiased PPO gradients. The mathematical
comparator is the eligible menu at the learner's visited state, not fresh HAPPO's
independent learning trajectory. No monotonicity or stationarity is claimed.

The same-world frozen-initial-policy control accompanies every evaluation
checkpoint, using identical initial states and per-episode action uniforms.
Its role and evaluator outputs are inaccessible to controller decisions.
Qualification evaluation returns are checked ONLY for finiteness and equality
at initialization. No AUC, final-return ranking or gain admission is allowed.
Training rewards are used only by the prospectively fixed learning algorithm;
this does not permit an analyst to tune from qualification outcomes.

## Frozen catalogue and resource accounting

The complete configuration is `current_packet_native/plan.json`. Fresh seed
79416017, eight banks/task/method, 16 episodes/bank, horizon25; six original cells
and all six semantic replays. The seed is disjoint from prior registered seeds.
Pinned NumPy2.2.2, torch2.6.0+cpu, HARL b1af98b and local PettingZoo1.24.3 runtime.
The source lock records all dependency hashes, not just the new runner.
Small 32x32 tanh actors are operational profiles, not a standard full-scale
benchmark claim. Actor/critic epochs5 match the pinned configuration; one full-
batch minibatch per epoch. No new package/environment installation.

Communication counts logical array payload bytes, NOT observed network wire
bytes or protocol headers. Complete modeled data paths per bank are:

1. Public state(float32) and team reward(float64) uplink, one message.
2. Advantages(float32) downlink to each actor, N messages.
3. Local observation(float32), action(int64), behavior log probability(float64)
   and advantage(float32) program uplink, N messages. Its entire eventual bill
   is reserved at launch; supersession/rejection does not refund it.
4. For current_value only, an additional early probe uplink of local observations,
   actions and behavior logs for m episodes, N messages, even if those bytes
   recur in the later program. Public states/rewards already paid are reused.
5. Accepted actor parameter broadcasts, actual tensor bytes, one message per
   changed actor. All proposed/server optimizer work is charged regardless of
   acceptance. No worker optimization/gradient message is invented.

Raw counters additionally retain native transitions, evaluation/control
transitions, actor/critic backprop rows, optimizer calls, source-gradient zero
counters, score forwards, potential fit/prediction rows, HAPPO factor forwards,
copied local-program bytes, and actual process CPU time. Each forward/backward
or potential-fit row has unit weight in the registered COMPUTE ROW PROXY.
It is not a FLOP estimate or a hardware CPU-time certificate; actual CPU time
includes Python bookkeeping, parameter/moment copies and ridge solves too.

Byte normalization is the analytically known fresh anchor's per-bank logical
public+advantage+program+all-policy bill. Compute normalization is its sampling,
critic-value, actor/critic-backprop and sequential-factor row count. Queue
targets are 1.25/2.0 in those fixed units. V=sqrt(8), gain scale=1 episode-return
unit, meta step=.1, common probe catalogue (m,lambda) with lambda in {.01,.1,1}.
Every probe has BOTH communication and compute cost; the controller prototype
was extended accordingly before this freeze. These constants are operational
design choices, not demonstrated optimal settings or tuned positive results.

## Gates, replay and hard stopping

All gates in plan.json are mandatory. Each complete cell is checked before the
next starts. The audit checks actual five-epoch gradient/parameter changes,
unchanged original server optimizer during proposals, immutable program hashes,
exact service times independent of action choice, single-use receipts, source
version lineage, same initial policies, no evaluation leakage, same-world frozen
control identity, exact ledgers and two-queue recurrences. The full method must
have genuine nonzero joint-version staleness; a delay label alone is insufficient.

One Python worker and one numerical-library thread. Total original+replay+
analysis active-process CPU limit1800 seconds, checked throughout proposals and
at bank/cell boundaries. No overlapping worker or retry. A timeout or payload
failure preserves partial root and stop decision and stops this registration.
Only after all six originals pass may the one registered complete replay run.
All non-timing cell fields must reproduce exactly; two disjoint analyzers must
produce byte-identical integrity.json, decision.json and resource_counts.csv.

Source/plan/lock are committed and pushed before execution. Command:

`D:/anaconda/Scripts/conda.exe run --no-capture-output -n ust2 python
ICML-2027/experiments/current_packet_native/run.py --lock
ICML-2027/internal/current_packet_native_lock_20261016.json --output
ICML-2027/results/current_packet_native_qualification_20261016`

Fresh sibling replay and analysis roots are owned by this driver. Never launch
them separately. Process identity and progress may be inspected while running;
incomplete return values may not be inspected.

## Bounded decision and precision boundary

Operational pass establishes executable timing/accounting/multi-epoch learning,
not better returns. Selected scores after argmax are biased; sample variance is
not an upper confidence bound. The selection-regret term and finite-probe error
floor in current_packet_value_design_20261016.md remain explicit.

The generalized probe cost is now QC*c_probe_j+QF*f_probe_j; replace QF*f_j by
this term in the existing proof. Costs remain predictable and cancel from the
selection error. The variance feedback identity and OMD proof are unchanged.

Useful discrimination at affordable cost is still OPEN. This qualification may
not certify it by ranking its returns. A complete-loop decision must distinguish
operational GO from permission for performance science. Only a justified useful
precision/cost interface permits separately preregistered fresh-seed learning-
response development; no large sweep, confirmation or manuscript admission
follows automatically from an operational pass. No GPU/HPC4/storage cleanup.
