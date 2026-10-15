# ICML 2027 learning-value upgrade: persistent plan

## Material Passport

- Origin: academic-research-suite / experiment-agent, authorized autonomous implementation.
- Date: 2026-10-13.
- Status: independent acquisition-aware efficient-learner confirmation and byte replay complete; fresh stochastic learning curves and independent accounting audit verified. Actual MARL actor/critic response qualification remains open.

## Decisions

1. Preserve TSP/, TSP-V2/, TSP-V3/, all historical gates and results.
2. Use the full q={1,2,4,8} catalogue; forbid task-specific q removal.
3. Avoid V2's clipped one-block reward, nonshrinking sliding-window bonus, and
   repeated validation tax. Pay one independent calibration stream.
4. Count both messages and TOTAL actor transitions, with parallel ticks logged
   separately. Never label parallel-clock ticks as total environment work.
5. First establish an observable score/true-risk interface in a tractable
   Markov TD model. The selector cannot access theta or rho.
6. Use known-mixing Gaussian whitening with exact t/chi-square intervals;
   do not label a normal approximation as a finite-sample certificate.
7. Freeze the CPU design in a separate commit before calibration outcomes.
8. If development passes, freeze a separate confirmation lock with fresh seeds.
9. After CPU confirmation, qualify the actual MARL learning-response interface
   before any multi-environment performance run. A controlled CPU pass is not a
   nonlinear MARL success claim.

## Required CPU outputs

Full-context risk/decision tables, coverage and interval widths, oracle regret,
interior-q selection, strong fixed comparators, ablations, source/config hashes,
an independent byte replay, and a complete decision preserving every cell.

## GPU evidence requirements

One frozen controller and complete q catalogue; discrete coordination and
continuous control; native independent, intermediate, and shared coupling;
full fixed catalogue plus correlation-only, myopic, and a specified adaptive
batch baseline; return versus total actor transitions and messages; separate
wall time and calibration cost; development before fresh confirmation.

No new manuscript claim is admitted until it is supported at the declared scope.

## Completed on 2026-10-13

- Preregistration source: f584d76.
- Ten targeted tests passed.
- Development: 360 contexts, 16 fresh seeds, 46,080 method rows; all gates pass.
- Five scientific files reproduce byte for byte.
- A separate 64-seed confirmation lock retains the original implementation,
  population, complete catalogue, and all gates.
- The myopic comparator is a one-step-then-commit ablation. A strong repeatedly
  replanning adaptive baseline is still required before an ICML algorithm claim.

## Strengthened learner and acquisition design

- Implemented causal scalar gains from minimizing exact quadratic Lyapunov risk.
  They attain the Gaussian linear-unbiased information bound in the declared model.
- Efficient fixed-q opponents use that same stronger learner. A global-single-q
  positive result was insufficient: the public-budget-wise diagnostic exposed a
  6.61% loss due to acquisition cost. This diagnostic is preserved.
- Added a uniform, outcome-free perfect-information futility theorem. It permits
  skipping acquisition when even the paid oracle cannot improve the full-budget
  public fallback. Every q remains available.
- CPU-003: 360 contexts, 16 fresh seeds; all original gates and byte replay pass.
  The budget-wise efficient-fixed risk ratio is 0.947742; interior selection is
  99.7024%; 80 contexts correctly skip acquisition without reading the sensor.
- CPU-003 confirmation: 64 independently locked seeds, 184,320 method rows;
  every frozen numerical gate and scientific byte replay pass. The budget-wise
  efficient-fixed risk ratio is 0.947742415 [0.947667792,0.947817043]; interior
  q selection is 99.7396%. Eighty contexts skip acquisition before sensing.
- Fresh learning verification: 18 method paths, 2,048 replicas per path,
  378 checkpoints; exact-risk diagnostics and byte replay pass. Figure checked
  visually; value-estimation MSE is not relabelled as team return.
- Independent recurrence audit verifies all actual total charges and horizons.
  Preserve the diagnostic-oracle prefix-label erratum in a derived sidecar;
  do not alter raw scientific outputs, frozen code, or any threshold.
- Relevant regression: 173 passed; existing TSP/V2/V3 evidence preserved.
- The pinned HARL critic replication audit runs on CPU. TensorBoard logging is
  unnecessary. A missing PyYAML import was resolved by reading the already
  installed pure-Python package from the base environment; no environment was
  installed or upgraded and no experiment payload was retried.
- HARL critic duplication changes parameters by at most 2.98e-8. The next
  interface must measure actor/critic response, optimizer work, and transfer,
  rather than attribute a shared-stream q=2 optimum to critic duplication.
- A separately frozen 32-payload CPU MAPPO actor audit covers two seeds,
  discrete/continuous action spaces, tanh/relu, and every q. All payloads are
  finite and replication-invariant within 1e-5 (maximum 3.28e-7).
  The audit JSON also reproduces byte for byte in a disjoint execution.
  Actual independent-data and evolving-policy response remain the next boundary.
- Next-stage requirements are in marl_response_qualification_plan.md. No new
  GPU performance study, nonlinear return claim, or manuscript migration is
  authorized by the Gaussian qualification alone.

## Decision after real-MAPPO response development (2026-10-14)

- The frozen `ICML27-MAPPO-OBSERVABLE-RESPONSE-CPU-001` cross-stage gate failed
  and is preserved at `marl_response_development_decision.md`; it is not to be
  rerun, tuned, or presented as return evidence.
- Do not immediately repeat q-specific fresh sensing: the existing shadow
  score observes candidate displacements after q-specific data and training,
  so it is not an executable pre-action controller and probe cost may consume
  the oracle gap.
- The candidate has been sharpened after checking the HAPPO/HAML literature:
  it is not generic asynchronous actor-block scheduling, but certified
  parallel batching of HAPPO-like sequential updates. One already charged
  joint rollout supplies per-agent gradient packets; the prospective design
  variable is the subset of policy blocks applied jointly plus their step
  sizes. Lyapunov drift would jointly account for message/environment budgets
  and a certified lower bound on return gain.
- This is only a candidate. Async MARL, stale-gradient scheduling, gradient
  conflict, and generic candidate-descent scoring are established prior work.
  HAPPO/HAML already provide sequential heterogeneous-agent updates and
  monotonic-improvement theory. The specific novelty hypothesis is a
  certified condition for parallelizing subsets of those updates under
  cross-agent interactions and stale packets, with resource-aware drift.
- CPU kill tests and proof obligations are recorded in
  `async_policy_block_scheduler_candidate.md`; no new seed, scientific run,
  experiment number, or GPU task is authorized until the observable path,
  confidence/certificate, oracle headroom, and charged compute are fully
  specified outcome-free.
- An exact small-team optimizer for the proposed concave drift surrogate is
  implemented in `experiments/async_policy_block_scheduler.py`; its six unit
  tests and the full 27-test ICML-2027 suite pass in the existing `ust2`
  environment. This is software verification only, not evidence for MARL
  return improvement or a valid gradient/curvature certificate.

## HAPPO return-bound audit (2026-10-14)

- Derived a sufficient bridge from the HARL Lemma-19 sequential return bound
  to a frozen-prefix simultaneous candidate: subtract a uniform pairwise
  conditional-advantage oscillation remainder
  `(1-gamma)^(-1) sum_{j before i} omega_ij TV(pi_j, pi'_j)`, a uniform
  post-selection confidence radius, and a separately justified stale-data
  bias term.
- The bridge only certifies non-decrease relative to the old policy when its
  RHS is nonnegative; it does not certify superiority over HAPPO's resulting
  policy. A monotonicity lower bound for each method is not a paired method
  comparison.
- Exact binary two-agent counterexample: frozen local gains sum to 2 while
  true simultaneous return change is `2-kappa`; the interaction correction
  recovers the exact HAPPO prefix term and rejects harmful updates.
- Novelty audit found existing concurrent/joint monotonic-update work
  (CoPPO, MAT) and a parallel-pipeline monotonicity proposal (FP3O preprint).
  The candidate cannot claim first parallel monotonic MARL optimization. The
  remaining hypothesis is budgeted online selection of policy blocks under a
  nonvacuous stale-aware return certificate; direct comparison and novelty
  remain open.
- Current step is still theorem-interface feasibility, not a scientific run.
  Do not start MARL learning or GPU experiments until (i) exact finite-sample
  confidence is uniform over the candidate family, (ii) interaction and
  staleness terms are computable without evaluation leakage and are
  nonvacuous, (iii) a relative-to-HAPPO comparison protocol is frozen, and
  (iv) an outcome-free independent-data CPU headroom screen is preregistered.

## End-to-end CPU decision experiment (2026-10-15)

- The receipt-time subset model was rejected before scientific execution:
  packets already computed and transmitted have sunk cost, and a subset of
  agents in a joint game does not remove the other agents' environment steps.
- `ICML27-HAPPO-ASYNC-BATCH-CPU-001` therefore makes the decision before actor
  packet launch and charges every native joint trajectory, optimizer row,
  gradient packet, proxy forward pass, and outcome-aware search trajectory.
- Stage A compares fresh sequential HAPPO, full-cadence HAPPO, fresh parallel,
  stale, age-first, causal history-surrogate, and an outcome-aware finite-menu
  diagnostic on two cooperative MPE tasks. The latter is an empirical
  opportunity diagnostic rather than a deployable method or formal upper
  bound.
- Development and holdout seeds, all actions, costs, replay checks, and gates
  are fixed in `happo_batch_cpu_preregistration.json`. A task must also show
  at least 2% fresh-HAPPO learning from initialization so that lack of signal
  is not misread as lack of allocation value.
- The final one-worker, 14-payload operational qualification passed every
  mapping, finite-value, hash, controller-separation, trajectory-identity,
  and exact-accounting check. A prior two-worker operational attempt stopped
  on host memory pressure and is retained; no scientific seed was involved.
- A complete Stage-A pass authorizes only a separately frozen, fresh-seed
  observable-score and confidence qualification. A failure requires a causal
  design change tied to the failed mechanism; gates, tasks, and failed seeds
  remain unchanged.
- Before reading any scientific return, Amendment 1 corrected a deterministic
  full-cadence credit-bank defect found during live ledger inspection. The
  first two unopened payloads are ineligible; every scientific seed was
  replaced. The amended one-worker qualification passed 14/14 payloads,
  including exact equal-budget exhaustion by every fixed schedule.

## Final A1 CPU decision and causal redesign (2026-10-15)

- All 80 payloads completed; frozen integrity and supplementary per-block
  resource recurrences passed. Six registered payloads reproduce all 18
  scientific files byte for byte, and two complete analyzers produce identical
  CSV/JSON files.
- Stage A **fails**: 10/16 scientific gates pass. Against the selected strong
  control, spread AUC gain is 0.7115% and reference is -0.0334%; both miss the
  frozen 2% threshold. Stage B and GPU performance work do not follow from A1.
- Search ranking transfers poorly to independent checkpoints and costs
  46.17x/11.58x the training joint transitions. The noisy finite-menu selector
  is not a true oracle ceiling, so this is not a proof that all allocation
  opportunity is absent.
- Reference's rolling age stream aliases with its cyclic selection order;
  nominal stale-cyclic, fresh parallel, age-first, and cyclic HAPPO coincide.
  Exact accounting passed but did not detect this missing mechanism separation.
- A substantive successor interface is implemented in
  `causal_packet_ledger.py`: unique launch jobs, service streams independent of
  selection clocks, single-use received raw gradients, server optimizer
  ownership, and permanently charged work/transmission after rejection.
  The 32-test suite verifies software only.
- `happo_batch_cpu_a1_decision.md` records the complete decision.
  `causal_packet_redesign_after_a1.md` derives the conditional receipt-step
  lower bound and exposes the remaining launch/receipt gap: one-step queue
  drift does not price a launch's delayed value. A successor scientific
  preregistration requires causal score, feasible paired precision, uniform
  return confidence, and multi-step drift together. No task/seed/gate rescue,
  narrative-only change, or unregistered successor run is authorized by A1.
