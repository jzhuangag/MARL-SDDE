# Acquisition-aware learning value: independent confirmation

## Material Passport and decision

- Origin: academic-research-suite / experiment-agent; autonomous implementation.
- Date: 2026-10-13.
- Status: VERIFIED for the stated Gaussian Markov critic model.
- Frozen development design: commit `1d92557`.
- Independent confirmation lock: commit `26355e5`.
- Confirmation seeds: 52000001--52000064; no overlap with development.

All ten frozen numerical gates and the separate scientific byte-replay gate
pass. An independent recurrence audit verifies every executed method's exact
message and TOTAL actor-transition charges and maximal usable horizon.
No GPU performance study or nonlinear MARL claim follows automatically.
TSP/, TSP-V2/, and TSP-V3/ are preserved.

## Algorithm and theory that were actually implemented

1. Before seeing calibration, compare every paid (q,spacing) action with the
   full-budget public fallback at rho=0 and rho=1. Affine risk differences
   certify acquisition futility over the entire rho interval, not just the grid.
2. When acquisition is potentially useful, pay for 65 observations from eight
   actors. Known-persistence whitening yields independent Gaussian innovations.
   Exact Student-t and chi-square intervals give joint parameter coverage and
   simultaneous candidate-risk bounds. The selector receives observations,
   not the evaluator's target or rho.
3. Select from the COMPLETE q={1,2,4,8}, spacing={1,4,16} catalogue by fitted
   finite-budget risk, with a true-risk selection bound on the coverage event.
4. Update the critic with the causal gain minimizing
   V_next=(1-kappa)^2 V+kappa^2 W. The resulting constant-memory innovation
   estimator attains the Gaussian linear-unbiased information bound. Fixed-q
   controls receive this same efficient learner, not a weaker constant-step rule.

Participation and spacing are chosen before the learning segment; the update
gain changes online. This is not a demonstrated moving-policy online-q law.
The exact Gaussian confidence/GLS machinery is established statistical machinery;
its use alone is not claimed as an ICML novelty result.

## Registered population and results

The complete 360-context population and eight methods give 184,320 method rows.
The factors are target, coupling, persistence, parameter staleness, overhead,
and actor budget. In this efficient scalar subclass target and recorded
parameter staleness do not alter terminal risk, leaving 90 distinct risk
configurations. All registered controls remain in the analysis.

| Comparator | Proposed/comparator geometric MSE | Paired-seed 95% interval |
|---|---:|---:|
| Public-budget-wise efficient fixed q | 0.947742415 | [0.947667792, 0.947817043] |
| Always acquire, same efficient learner | 0.985856605 | [0.985834522, 0.985878688] |
| Same-cost admissible oracle | 1.010125095 | [1.010045561, 1.010204636] |

The primary reduction is 5.2258% against a fixed-q comparator that may depend
on every public cost/persistence configuration. The comparator is the declared
prior-geometric-risk-optimal fixed q, not an arbitrary q or a task-selected
subset. The oracle comparison is a diagnostic, not a deployable method.

- 80 registered contexts (20 distinct risk configurations) skip acquisition.
- 84 interior-qualified contexts (21 distinct risk configurations); correct
  full-catalogue q selection is 99.7396% over confirmation calibration draws.
- Probed joint parameter coverage: 99.0011%.
- Probed simultaneous all-action risk coverage: 99.6373%.
- No omitted seeds, cells, methods, or failed payloads; all totals are finite.

These uncertainty intervals quantify calibration randomness for the declared
finite population. They do not estimate generalization over unobserved tasks.
They are Student-based intervals for paired seed-mean log ratios, not exact
distribution-free performance guarantees. The exact Gaussian parameter
coverage theorem is a different claim.
The primary endpoint integrates future learning noise exactly; rows are not
independent training trajectories. Risk improvement is aggregate, not a
per-cell no-harm guarantee.

The prespecified population is not uniformly improved. For example, ratios
aggregated within rho={0,0.2,0.5,0.8,1} are respectively
{0.833052,0.989791,1.008077,0.981580,0.937166}; the intermediate rho=0.5 group
has a 0.81% acquisition-related cost. By actor budget {8192,32768,131072},
ratios are {1.001903,0.951319,0.893140}. These groups remain in the primary
aggregate and explain why the result is not a universal safety guarantee.

## Fresh stochastic learning verification and figures

The separately frozen validation (`ff5ecc6`) generates actual new Gaussian
learning paths for three specified coupling contexts and six methods. Each of
the 18 paths uses 2,048 fresh replicas; all 378 checkpoints agree with exact
risk within the declared five-Monte-Carlo-standard-error diagnostic tolerance.
CSV, PNG, and validation JSON reproduce byte for byte in a disjoint run.

The figure has Times New Roman text, equal subplot geometry, shared charged
dual-budget axis ticks, and labels below each subplot. Its bands are 1.96
Monte Carlo standard errors conditional on one fixed confirmed calibration
decision, not between-seed confidence bands for the entire controller.
The visualized metric is value-estimation MSE, not MARL team return.

## Accounting metadata erratum

The raw diagnostic admissible-oracle rows incorrectly label their acquisition
prefix fields as zero when the selected oracle action is paid. Their total
charges and reduced usable horizons already include acquisition correctly.
The independent audit reconstructs prefix labels without altering any source
artifact, scientific risk, selection, total, frozen seed, or gate.

There are 8,192 affected diagnostic-oracle rows in confirmation and 2,048 in
development. Every actually executed controller/comparator prefix is exact.
Corrected derived ledgers and compact audit JSON are preserved separately.
This is a recorded metadata erratum, not a silent scientific rerun or repair.

## Reproduction and checks

Scientific confirmation SHA-256:

| File | SHA-256 |
|---|---|
| endpoints.csv | 2f2f544e546d76753aee511f9067b5a496641e40c9d42c1116a4dfe1c82ae641 |
| certificates.csv | 1c2aacce92ab3bb99896d634c5edb98ae853cd0a1cd07bd00bf229c7ec1ba696 |
| summary.json | fd3cca6fd61e53c612f2c1040cb20c662087e81bd541c652cd2af481ab596419 |

Reproduction commands, from the repository root, using new output names:

```powershell
D:/anaconda/envs/ust2/python.exe ICML-2027/experiments/run_acquisition_cpu.py --confirmation-lock ICML-2027/experiments/acquisition_confirmation_lock.json --out ICML-2027/results/acquisition_confirmation_new_reproduction
D:/anaconda/envs/ust2/python.exe ICML-2027/experiments/audit_acquisition_results.py --source ICML-2027/results/acquisition_confirmation_20261013 --replay ICML-2027/results/acquisition_confirmation_new_reproduction --out ICML-2027/results/acquisition_new_audit
D:/anaconda/envs/ust2/python.exe ICML-2027/experiments/validate_acquisition_trajectories.py --source ICML-2027/results/acquisition_confirmation_20261013 --out ICML-2027/results/acquisition_new_trajectory_check
```

Relevant regression: 173 tests pass across TSP, TSP-V2, TSP-V3, and ICML-2027.
The 41 warnings are existing np.trapz deprecations. No remote job or storage
operation was performed for this upgrade.

## Statistical and transfer audit

The ARS statistical fallacy scan covers 11/11 categories:

| Category | Check and interpretation |
|---|---|
| Simpson's paradox | Ratios examined by rho, persistence, actor budget, and overhead; subgroup costs above are retained, not called uniform improvement. |
| Ecological fallacy | The claim concerns the declared aggregate; no individual-task or unseen-task guarantee is inferred. |
| Berkson's paradox | No outcome-based filtering of the primary population; the interior subset is analytically specified and secondary. |
| Collider bias | No post-outcome conditioning or regression adjustment defines the primary comparison. |
| Base-rate neglect | Interior selection includes its 84/360 registered-context denominator; probed coverage has a separate denominator. |
| Regression to the mean | Independent confirmation uses no selection of extreme development seeds or cells. |
| Survivorship bias | All 360 contexts, 64 seeds, and eight methods remain; no failed payload is omitted. |
| Look-elsewhere effect | One primary budget-wise comparison; auxiliary intervals are descriptive and are not simultaneous superiority claims. |
| Garden of forking paths | Source, independent seeds, population, and gates were committed before confirmation; prior designs and diagnostics remain preserved. |
| Correlation versus causation | Model experiments manipulate coupling with unchanged actor marginals; no observational or general MARL causal claim is inferred. |
| Reverse causality | Calibration precedes selection; evaluation truth cannot enter the controller. |

Additionally, independent seed-level aggregation prevents endpoint-row
pseudoreplication; fixed target/delay controls do not multiply inferential
sample size. The primary comparison and gates were frozen; seed-level log
ratios define inference. Auxiliary intervals are descriptive, not a familywise
claim of simultaneous superiority. Joint parameter coverage is a separate
theorem-backed event. Monte Carlo tolerance is a numerical diagnostic.

The previous constant-step CPU-001 confirmation passes its own original
gates. The efficient CPU-002 global-single-q result is not the headline:
its stronger budget-wise diagnostic loses 6.61%, and is preserved. CPU-003
addresses that identifiable acquisition-cost mechanism prospectively, then
confirms it on independent seeds. No prior threshold or result was changed.

The actual pinned HARL full-batch critic audit is near replication-invariant
(maximum parameter difference 2.98e-8). A further 32-payload MAPPO actor audit
also passes duplication invariance over discrete/continuous actions and
tanh/relu networks (maximum difference 3.28e-7). Therefore a shared-stream interior
q optimum cannot be explained by duplicated critic batches alone. Actual
independent-data actor/critic dynamics, optimizer work, horizon, and calibration-to-training
transfer must pass the next interface qualification before multi-environment
return confirmation or an ICML-ready manuscript is claimed.
