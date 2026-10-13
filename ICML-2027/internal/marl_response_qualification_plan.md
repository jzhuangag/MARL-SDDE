# Next boundary: actual MARL learning response

## Status

Design plan, not a frozen GPU performance preregistration.
The confirmed Gaussian CPU result establishes a cost-aware observable-risk
interface and online Lyapunov gains in its declared model. It does not identify
PPO contraction or certify a moving-policy return bound.

## One research question

Can a fully charged, observable learning-response model choose parallel data
acquisition from the full q catalogue more effectively than strong tuned fixed
and adaptive controls under communication and TOTAL actor-transition budgets?

Dependence, per-update efficiency, usable horizon, and acquisition cost belong
to that one finite-budget decision. Additional queues, graph mechanisms, or
SDDE approximations should not be attached unless the actual decision requires
them and the discrete theorem remains valid.

## CPU interface work before performance jobs

1. Qualify both actor and critic on fixed pinned HARL buffers. Start from exactly
   matched parameters, optimizer state, normalization state, and random-number
   streams. Distinguish duplication, genuinely independent additional data, and
   intermediate dependence. Keep q={1,2,4,8}; count every model evaluation.
2. Measure observable held-out critic tracking loss, actor policy movement/KL,
   gradient variability, and predicted versus realized update response. Do not
   relabel training reward, actor KL, or critic loss as a team-return certificate.
3. Compare fixed optimizer steps with fixed processed-row work. Audit how q
   changes epochs, minibatches, normalization, and update count. Include fixed-q
   learning-rate schedules and development-selected stopping checkpoints; a
   long-horizon baseline must not be made to overtrain merely to help a selector.
4. Use disjoint calibration/validation buffers and subsequent-state transfer
   checks. An empirical response fit is labelled empirical until an appropriate
   bias/confidence/regularity argument is established. Known Gaussian whitening
   intervals cannot be transplanted to unknown neural Markov data.
5. Charge acquisition, validation, candidate trials, transitions skipped for
   decorrelation, messages, and model/optimizer work. Log parallel ticks and
   wall time separately from total transitions. No sensor can inspect benchmark
   evaluation return; development outcomes may tune one frozen rule only.

## Prospective performance package, conditional on qualification

Candidate families are pinned MPE discrete coordination and MaMuJoCo continuous
control, with a second continuous task proposed before final confirmation.
Native independent data, an intermediate unchanged-marginal coupling, and a
shared extreme should test the mechanism rather than supply environment counts.
Exact tasks, versions, budgets, effects, seeds, and decision gates must be
frozen after interface qualification and before scientific development.

Required controls are all fixed q, correlation-only selection, a genuinely
replanning myopic controller, a specified adaptive batch rule, and acquisition
ablation. Every fixed-q schedule is tuned on an equal declared development
allowance. The strongest learned-static and online comparators remain in
confirmation; no task-specific q exclusion is allowed.

Report return versus total transitions and communication as primary axes;
wall-clock and model/control overhead are supplementary. Robust population
gain and defined regime-wise safety are different questions, so neither an
all-cells-first rule nor an aggregate gain may silently substitute for the other.

Use development to choose and freeze the single algorithm. Use independent
confirmation seeds for claim admission. Preserve all tested tasks and scientific
failures; do not rescue a design by removing an environment, weakening gates,
or presenting the same pilot as a fresh confirmation.

## Concrete current stop boundary

The actual HARL critic replication audit is complete and nearly invariant.
The actor response, nonlinear score-to-risk link, and state-transfer guarantee
are not yet qualified. No new multi-environment return result or GPU job is
reported in the CPU package. The next work item is those operational/model
checks, not another unqualified task sweep.
