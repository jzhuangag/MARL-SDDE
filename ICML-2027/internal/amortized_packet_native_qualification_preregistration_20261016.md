# Amortized complete native loop: outcome-free qualification

## Material Passport

ARS experiment-agent/run and validate; prospective operational test, not
performance development, useful-score validation, or manuscript admission.
Source/plan/hash lock must be committed and pushed before execution. This is
the one selected amortization correction, not a retry of prior failed science.

## Frozen experiment and roles

Plan: experiments/amortized_packet_native/plan.json. Fresh seed79516031;
Spread and Reference; fresh sequential HAPPO, matched latest-ready eligible-all,
and amortized current_value. Eight banks of16 independent25-step episodes;
five genuinely recomputed PPO epochs, five critic epochs. All6 cells and all6
registered semantic replays. One worker;1800 active CPU seconds including replay
and analysis. No return ranking, baseline selection, tuning, or learning gate.
Qualification evaluator returns may be checked ONLY for finiteness and exact
unchanged-policy equality at initialization. Training/evaluation/controller
roles912/913/911 are disjoint. No previous qualification return used.

## Causal algorithm

Independent actor/launch-ordinal service ticks1--2. Immutable local data programs
contain observations/actions/behavior logs/advantages, no worker optimizer.
Each latest-ready actor receives one server-owned five-epoch unit PPO proposal.
Alpha in{0,.5,1,2} is a post-PPO PARAMETER line search, not a learning-rate sweep.
Idle keeps old server moments; any positive alpha uses unit proposal moments.
Full policies, coefficients, single-actor unit basis and past-only MC potential
freeze before the bank. Joint meta option(m,lambda,geometry) freezes before data;
m in{4,8,16}, lambda in{.01,.1,1}, full-vector/response-basis geometry. Full
unclipped prefix-IS actual-return scores select the CURRENT receipt. Neither
selected score nor empirical variance is an unbiased selected gain/UCB.
Both geometries supply all-option sample variance/remainder feedback. The basis
keeps the exact nonadditive interaction remainder; no post-bank minimum penalty.
Fixed return scale R=1; own-return telescoping assumptions remain explicit.
Current bank creates only future delayed programs, after the receipt decision.

## Complete costs and static feasibility

All native training/evaluation/control transitions, sampling/critic/PPO/factor/
candidate forwards, potential fitting, unit program copies, tensor-state copies,
parameter interpolation and geometry representation entries are recorded.
Early private probes contain ONLY eligible changing actors. Public states/rewards,
advantage downlinks, every full future program (including duplicated probe rows),
eventual receipt/drain and policy broadcasts remain paid. No refund is allowed.
Initial frozen-control actor tensor copy is separately recorded as fixed startup
work; per-decision resource queues include candidate and program materialization.
Logical array payload bytes are not wire bytes. Work units are explicit proxies:
one native row plus copy bytes/4096 and interpolation/geometry entries/64.
These conversion constants do not assert equal hardware cost/FLOPs; actual CPU is
separately recorded. Initial fixed startup cost does not affect action comparisons.

Communication target1.25 and compute target1.5 are prospective reference-scaled
design budgets, not hard caps. For an all-ready workload, worst past-potential
fit is4BH. With the smallest m=4 and idle, the row-only compute charge is at most
28/30+(.08334) for N=3 and22/22+(.07955) for N=2 before copy/scalar work.
Using the architecture's small32x32 actor tensors, conservative12 times the
parameter-byte total for proposal/variant state copies plus two full-program
copies and interpolation/64 keeps this below1.5 in both task shapes. The source
audit must verify this numerical bound from observed shape metadata WITHOUT
ranking returns. Idle communication plus eligible m=4 private probes is below
1.25 for both qualified task shapes. Thus a minimum-probe idle reference is
analytically feasible; nonzero update quality/precision is NOT established.

## Mandatory gates and stop

Require exact6-cell mapping/hashes/source/runtime lock, initial-policy equality,
all finite artifacts, five distinct recomputed epoch states/gradient traces,
unit-server moment ownership, unique packets/lineage/receipts and independent
service streams, actual version staleness, all charge/queue recurrences,
pre-bank basis/meta and post-score receipt, evaluation exclusion, six exact
non-timing semantic replays and two byte-identical analyzers. Full interaction
statistics remain finite and all options accounted. Costs and gate definitions
are frozen before data. Any payload failure stops this registration without
retry, source/gate change, or a new automatic qualification attempt.

Command after push:
`conda run --no-capture-output -n ust2 python ICML-2027/experiments/amortized_packet_native/run.py --lock ICML-2027/internal/amortized_packet_native_lock_20261016.json --output ICML-2027/results/amortized_packet_native_qualification_20261016`

Only a complete operational pass and a useful expected-risk/full-cost feasibility
judgment permit a separately committed fresh-seed learning-response development.
Practical precision must be tested jointly with learning and cost in that study,
not inferred from this operational pass. No GPU/HPC4, cleanup, prior evidence
changes, or TSP manuscript edits are authorized here.
