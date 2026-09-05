# ICML 2027 mainline and evidence plan

Date: 2026-09-02.

## Current unique candidate

The sole current ICML candidate is **Two Clocks, One Game**: barrier-free
centralized training with decentralized execution for distinct policy blocks in
one Markov potential game. Each agent owns at most one fixed-horizon policy
gradient packet in flight. A completed packet is applied immediately with a
single common pathwise-certified step, then its owner launches the next packet
from the current joint-policy version.

The paper asks:

> When does heterogeneous packet completion yield genuine wall-clock learning
> value in an interacting Markov game, and when must teammate-policy staleness
> erase that value?

The intended answer is a rate--coupling phase. The favorable quantity is
aggregate useful packet completion under heterogeneous service. The adverse
quantity is the interaction-weighted teammate-policy motion while a packet is
in flight. Single-flight ownership makes a packet self-fresh in its own policy
block; it remains strategically stale in teammate blocks. A slow essential
policy block remains a wall-clock bottleneck for full-policy or Nash guarantees.

The recommended title is **“Two Clocks, One Game: Rate--Coupling Limits of
Barrier-Free Policy Learning in Markov Potential Games.”** It must not claim
universal asynchronous no-harm, generic linear speedup, or that aggregate
throughput eliminates an essential slow actor.

## The theorem package that must appear in the paper

1. **Event-time upper bound.** Give the finite-time stationarity result for
   arbitrary bounded packet-completion order using an off-diagonal
   Lyapunov--Krasovskii history energy. The common certified step must be the
   declared positive-root condition based on the interaction envelope; the
   online block update remains ordinary `O(dim(theta_i))` policy gradient.
2. **Wall-clock conversion and barrier separation.** Convert the event-time
   result through bounded service intervals or rates, exposing aggregate
   completed work and the essential-block coverage window. Show why a fully
   utilized frozen-policy barrier cannot replace adaptive query depth with
   duplicate packets sampled at one frozen policy.
3. **Strategic-staleness obstruction and essential-agent lower bound.** State
   the stale-direction safe-progress boundary and the stochastic
   essential-agent clock lower family. Delimit the result honestly as an
   instance-sensitive phase picture, not a globally minimax-matched theorem.
4. **Finite-horizon Markov-packet interface.** Account for the exact
   likelihood-gradient identity, truncation bias, bounded packet second
   moment, and conditional independence of completion from packet innovation.
   Give the factorized-softmax cross-sensitivity instantiation and distinguish
   a theorem-facing uniform certificate from any practical measured proxy.
5. **Conditional stationarity-to-Nash conversion.** Prove or explicitly
   delimit the MPG interiority/distribution-mismatch constant; no uniform Nash
   conclusion may be silently inferred where this condition is absent.

The central technical distinction from black-box delayed SGD is structural:
the owner block is self-fresh, so the delayed history pays only off-diagonal
cross-agent coupling while diagonal curvature remains local.

## Evidence status

The exact multi-state CPU confirmation passed all ten frozen gates. Across 12
heterogeneous primary cells, certified asynchronous learning was faster than
the fully utilized frozen-policy barrier in all 12 cells, with geometric
target-time ratio `0.4445572803`.

The independent stochastic Markov-packet confirmation passed all twelve frozen
gates and reproduced byte-for-byte. Across 64 fresh seeds and the same 12
heterogeneous cells, its certified-asynchronous/barrier ratios were
`0.4313525930` in wall-clock time and `0.4296873191` in charged transition
work; all 12 cells had lower median wall-clock time. Higher service
heterogeneity helped and stronger coupling hurt in every registered direction.
Homogeneous high-coupling controls retain the predicted negative side of the
phase. Raw asynchronous learning is a speed reference, while the certified
learner is the theorem-facing method; the reported certificate-cost ceiling is
not a superiority claim over raw async.

These are fully charged CPU results for the theorem-facing model. They do not
establish performance on standard nonlinear MARL benchmarks.

## Standard-MARL evidence plan

After CPU/theory closure, freeze a separate outcome-free GPU preregistration
for the single-flight common-step learner. It is pathwise certified on the
finite-policy theorem class and an explicitly empirical instantiation on the
unconstrained neural actor. It must retain CTDE/decentralized execution and
compare against a fully utilized barrier, raw single-flight async, a strong
delay-adaptive asynchronous reference, and official HAA2C/HAPPO/MAPPO
baselines where applicable. The planned task layers are a CPU integration
smoke, continuous heterogeneous MAMuJoCo control, and partially observable
SMACv2 cooperation.

Report return versus actual wall-clock and transitions separately, together
with charged partial/cancelled work, utilization, actor idle fraction,
policy-lag events, teammate KL drift, proposal scale, and final/lower-tail
return. Standard-MARL evidence is an empirical extension beyond the tabular
MPG theorem and must not be represented as theorem-covered PPO, Adam, learned
bootstrapped critic, or recurrent-policy behavior.

No GPU job, seed registry, threshold, or formal standard-MARL result is
authorized by this document. Those items require a later frozen
preregistration after the CPU/theory gates close.

## Superseded historical route

The reward-free participation / T-063 route is **superseded** as the ICML
mainline. Its historical experiments, frozen gates, seeds, diagnostics,
results, and conclusions are preserved unchanged in their existing records;
they are not evidence for the current Two Clocks candidate and are not
reinterpreted or repaired here. The supersession does not authorize changing
any historical threshold, comparator, result, or formal-status decision.

## Remaining gates and stop rules

The publication upper/lower chain and dependence audit are now consolidated.
The architecture-specific audit closed negatively for the unconstrained HARL
actor, so no neural theorem-coverage claim is allowed. Before GPU work, freeze
the external-checkout overlay, dependency lock, non-scientific resource
preflight, task budgets, seeds, baselines, analysis hashes and stop rules in a
separate preregistration. Complete a fresh primary-source novelty and
citation-integrity audit before manuscript delivery.

Stop the mainline before GPU if the lower result collapses to generic delayed
SGD after removing distinct policy blocks and the joint Markov trajectory, or
if the neural implementation cannot show a charged, reproducible asynchronous
advantage over both the fully utilized barrier and a strong asynchronous
reference without outcome-selected delay injection.
