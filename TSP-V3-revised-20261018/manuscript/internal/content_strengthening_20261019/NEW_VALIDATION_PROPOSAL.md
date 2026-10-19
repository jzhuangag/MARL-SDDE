# V3 risk-gate direct validation — separate budget request

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent (plan mode).
- Origin Date: 2026-10-19.
- Verification Status: NOT RUN; REQUIRES HUMAN APPROVAL OF SEPARATE COMPUTE.
- Version: proposed_v1. Existing outcomes were known when this design was proposed.
- This is NOT part of the sealed 140 CPU-hour MARL study and cannot change it.

## Question and scope

Does the explicitly risk-corrected gate provide the predicted population-risk
shield and a useful admission/safety tradeoff when actually executed with new
seeds, rather than reconstructed from old branches? This tests the controlled
Gaussian running mean and averaged delayed affine learner, not neural MARL.
Success cannot establish unknown-kernel calibration or TSP acceptance readiness.

## Fixed design proposed for approval

- 16 public families: Q in {8,32}, theta_low=.05, theta_high in {.5,2},
  lambda in {.2,.8}, D in {0,8}; h=8, private variance=1.
- Four public scales s in {32,128,512,2048}; B_msg=(h+Q)*s, B_env=4*s.
  Thus 64 family/budget designs, each with BOTH regimes and BOTH learners.
- New seed list: all 128 integers 20571001 through 20571128. No development
  seed selection; verify non-overlap with all earlier V3 seed registries before start.
- Running mean and averaged FIFO: unchanged weight/initialization definitions;
  affine mu=.5, eta=.1/[mu*(D+1)], pay all scheduled updates and withhold D.
- One fixed probe candidate per family: q_d=Q, b_d=1, smallest integer
  n_d<=4096 whose exact Gaussian affinity is <=.025. If none, or budget infeasible,
  fall back and preserve the infeasibility. No outcome-based probe search.
- Public downstream catalogue: q powers of two up to Q, b in {1,2,4,8}.
  Under each hypothetical instance minimize its exact stationary commitment risk
  after the paid probe, with lexicographic (q,b) tie breaking. Freeze both programs
  before probing. This improves the low-instance program constructively, rather
  than requiring it to repeat the all-agent action after losing the probe budget.
- Primary allowance epsilon=.1 fixed across families; zero allowance is a
  separately labelled stress target, not a replacement selected after results.
- Six policies: corrected gate; identical gate without carryover; information-only
  always-probe-when-feasible; no-probe all-agent; no-probe fixed(q=2,b=1);
  known-hypothesis full-budget catalogue oracle (evaluation-only).
- Exactly 16*4*2*2*6*128 = 196,608 policy trajectories, including infeasible
  candidate/fallback cells. Shared potential observations for paired comparisons;
  independent seed streams. The actual unknown regime never enters deployment decisions.

## Primary analysis and failure interpretation

1. Gate feasibility/admission and actual terminal squared-risk versus fallback,
   separately for every learner/regime/budget. Report all adverse cells and all costs.
2. Compare corrected versus no-carryover admissions and low-risk penalties. A
   correction need not reduce empirical risk uniformly. No-admission is a result.
3. Report certified population upper risk, empirical mean and bound/mean tightness;
   finite-sample means above a population bound are not automatically proof failures.
4. Paired seed-level mean differences, family-macro means and paired bootstrap
   95% intervals (2,000 resamples, analysis RNG seed 20579901). Use descriptive
   cellwise intervals; no claim of familywise significance or equivalence from a null.
5. Check likelihood directional errors, same-stream weights/conditional identity,
   exact paid ledgers, initialization, no true-regime leakage, source/runtime hashes,
   finite values, and deterministic replay byte equality after removing only runtime fields.
6. Predeclared negative outcomes: all fallbacks; certified but practically vacuous
   bounds; no gain over original controls; safety/improvement uncertainty; replay
   or ledger mismatch. No automatic tuning, seed extension, retry or cap increase.

## Explicit requested CPU envelope: AT MOST 8 additional CPU hours

| Phase | CPU allocation |
|---|---:|
| Build/identity checks and cost-only admission profile | .25 h |
| All 196,608 direct policy trajectories | 4.00 h |
| One full semantic replay of the same fixed suite | 3.00 h |
| Double analysis, integrity checks, tables/plots | .75 h |
| Total separate hard ceiling | 8.00 h = 28,800 CPU s |

These are allocations, NOT measured runtime forecasts. The exact workload has not
been benchmarked, so completion within 8h is not guaranteed. The cost-only profile
is included, must not inspect scientific returns, and must stop for human direction
if the fixed workload is projected to exceed this ceiling; do not shrink it to fit.
The cap includes failed/partial work and replay/analysis. No silent retries.

To avoid competing with the current sole native worker, launch only AFTER its
140h workflow reaches a terminal decision, unless the human explicitly supplies
another noncompeting resource and authorizes its use. Until approval, no new
simulation driver is launched. Literature, proofs and saved-evidence work do not wait.

## Required before execution

After approval: implement an isolated driver, validate it without native scientific
payloads, freeze exact source/config/runtime/seeds/protocol and output root, commit
and push admission before data, then run one worker with a cumulative CPU guard.
The final entry command and hashes will be recorded then; no runnable command is
claimed for an unimplemented driver. Preserve every output and any stop/failure.
