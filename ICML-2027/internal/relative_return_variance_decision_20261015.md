# Relative-return estimator: completed decisions and substantive continuation

## Material Passport

- Original research, native CPU estimator qualification, and deterministic replay.
- Status: A1 and shared-bank qualifications completed; no algorithm/GPU performance claim.
- Registration commits: `a9c9916` original; `532c5de` exact-identity amendment; `b76914d` shared-bank extension.
- Frozen TSP manuscripts and all earlier failed experiments remain untouched.

## Completed evidence

The first-disagreement estimator is structurally unbiased for actual finite-
horizon team-return differences. Exact two-agent Markov enumeration verifies
the identity. Its second moment is no greater than ordinary maximal coupling;
that proof does not automatically compare it with every common-random-number
scheme or establish learning improvement.

All eight A1 contexts and 24 candidate policies completed. All 24 candidate
sample means are nonzero (the paired baselines also have nonzero means).
Compared with the better of the two observed paired estimators, task geometric
mean transition-cost-adjusted variance ratios are **0.632125** for Spread and
**0.259178** for Reference: reductions of **36.79%** and **74.08%**. Seven of
eight context averages improve. These are qualification estimates from two
seeds per task, not universal variance-per-second or return-superiority claims.

Eight of nine A1 gates pass. The positive independently validated selected
score gate fails: **0/8**, versus the registered minimum 4/8. This qualification
therefore remains **NO-GO** overall. In particular, the original validation
uses inverse-CDF paired native episodes, and only **2/24** validation effects
exclude zero by the diagnostic one-sided normal-SE threshold. Its precision
is insufficient to determine most signs; this is not evidence that every
candidate's true mean is zero or that the new estimator is biased. A fitted
score and a noisy validation mean cannot be treated as exact ground truth.

All scores agree with independent validation within the registered 4-SE
diagnostic. That broad compatibility is not proof of small bias or predictive
accuracy: first-disagreement and validation means have the same sign on only
12/24 candidates, while most validation signs are themselves unresolved.

The conservative, known-support simultaneous empirical-Bernstein interface
has **0/24 positive lower bounds**. Low observed variance does not eliminate
its support term. No Gaussian approximation or sample maximum was relabelled
as a rigorous certificate. Full stage-4 probing remains nine times the paid
training prefix for the branch estimator; variance efficiency alone is not
affordable per-update control.

## Shared-bank result

The shared-reference extension was registered before opening A1 outcomes.
It reconstructs the identical frozen policies, uses a new independent bank,
and estimates real return via exact unclipped likelihood ratios with a frozen
state-value control variate. It is unbiased for any fitted potential, with
independence at episode level. All four fixed estimator variants are retained;
the TD-control-variate primary is not switched after results.

Its native score sampling is **12,800** transitions versus **115,200** for
the three-candidate branch banks, a ninefold reduction. But its primary
amortized cost-variance ratios to first-disagreement are **0.543423** for
Spread and **2.156551** for Reference. Thus the primary cost-variance gate
fails on Reference; positive independently validated selected scores are
again 0/8. The shared-bank qualification is **NO-GO** overall. Lower raw
sampling cost is not an algorithmic success when estimator variance increases.

## Integrity and operational records

- Clean pinned HARL and preserved MPE runtime hashes match the original locks.
- 1,024 unique training seeds, 4,608 score seeds and 2,048 validation seeds
  are disjoint by role. Validation never enters candidate construction or fitting.
- Every native common path, replayed prefix, suffix, forward row and training
  optimizer-row recurrence is verified. Parameter payload counts are logical
  simulator payloads, not measured physical network messages.
- A1 registered context replay is byte-identical. The shared-bank full replay
  reproduces all eight context files byte-for-byte.
- Both analyzers' complete metrics and summaries replay byte-identically.
- Original A1 operational stop: equal categorical laws had a spurious tiny
  disagreement due to summation roundoff. Old partial outputs stay unopened
  and ineligible. Exact-equality handling was isolated, all scientific seeds
  replaced, and no scientific gate changed.
- Shared-bank analyzer initially failed only while serializing a NumPy boolean.
  Both partial analysis roots are preserved. A separate serialization-only
  entry point converts scalar types without changing formulas, samples or gates.

## Mechanism-specific continuation, not a gate rescue

The learner's discounted, time-homogeneous critic is not necessarily an
effective control variate for undiscounted, finite-episode return differences.
This is a plausible cause of the Reference variance, not an experimentally
isolated causal conclusion. Test it with a **separate training-only Monte Carlo
potential**, including remaining horizon and native undiscounted return-to-go.
Its parameters are fitted on already paid training episodes, never score or
validation returns. Its error cannot change the importance-estimator target.

The separately frozen new qualification retains both tasks and all three
candidate policies, replaces every context seed, compares with independent
first-disagreement samples, and improves validation using a disjoint shared
reference bank. No previous decision is recomputed as a pass. No GPU follows
unless observable precision, cost, causal launch/receipt transfer and a useful
theory interface are actually established.

## Fresh finite-horizon-potential qualification: completed

The new study registered at `8ba05cb` is complete on eight entirely fresh
task/seed/stage contexts, keeping all three candidates. The potential is a
fixed ridge fit of remaining-horizon features to native undiscounted Monte
Carlo returns-to-go from already paid training episodes. No score/validation
labels enter its fit. Separate score and validation roles use reference banks;
an independent first-disagreement sample checks the target by a different
estimation construction.

Both task cost-variance gates pass: geometric mean ratios to the new independent
first-disagreement comparator are **0.151188** (Spread) and **0.432552**
(Reference), reductions of **84.88%** and **56.74%**. These use fresh contexts;
they must not be multiplied by the earlier reductions to claim a compounded
improvement on the same data. The shared score bank still costs 1,600 native
transitions per context, plus fitting and forwards; it is not zero-cost.

Six of seven recorded gates pass. The frozen positive independently validated
selection gate remains failed: **2/8**, below 4/8. **Both positives are Spread
contexts from the same seed** (stages 4 and 16), not robust positive return
evidence across both tasks or seeds. Every score and direct-branch mean is
compatible with independent validation under the registered 4-SE diagnostic.
No realized positive score is promoted to a rigorous simultaneous certificate.

Context-0 replay and both analyzer CSV/JSON outputs are byte-identical.
Additional role reconstruction verifies 1,024 training, 512 score, 2,048
validation and 768 direct-branch episode seeds, all disjoint. All source locks
remain intact, and **43 relevant tests pass**. The complete machine-readable
validation, all three decisions, sources and replay digests are preserved in
`relative_return_complete_validation_20261015.json`.

The bounded empirical conclusion is therefore: **a substantively more efficient
actual-return estimator exists and has some independently reproduced positive
scores, but stable positive action selection and an affordable end-to-end
asynchronous learning advantage remain unestablished**. All three overall
qualification decisions remain failed. No GPU experiment, paper migration,
sample-budget increase or threshold rescue follows automatically.

Continue with the causal decision interface, not another estimator bake-off:
define a receipt-time, server-owned update candidate and its eligible reference,
price the probe budget before opening its data, and carry policy motion during
probe/receipt plus already sunk work explicitly. Decide whether a finite-budget
expected-risk guarantee is useful before preregistering any learning benchmark.
The expected-risk interface below is only a starting lemma; it is not that theorem.

## Expected-risk interface worth deriving next

A per-decision, always-nonnegative confidence lower bound is not the only
possible performance target. For a fixed eligible action family and unbiased
relative-return estimates, let their episode-level covariance be Sigma and
their means be mu. With m independent shared episodes, the selector maximizing
estimated utility `mu_hat[a] - Q*c[a]` obeys

`E[(best_true_utility - selected_true_utility)^2 | frozen history]
 <= 4 trace(Sigma)/m`.

Proof: pointwise selection regret is at most twice the maximum estimation
error, hence its square is at most four times the sum of squared errors.
Conditional unbiasedness and iid episodes identify their sum's expectation
as `trace(Sigma)/m`. Cross-candidate correlation is allowed. Known future action
costs may be subtracted without changing the error; the probe cost is still
charged separately. A sample covariance is an unbiased *estimate* of Sigma,
not a realized upper confidence bound or a per-step safety certificate.

This lemma is an expected-risk interface, **not** queue stability, stationarity,
an async-MARL convergence theorem, or a novelty claim. In particular, it does
not make a candidate known only after computation into an observable pre-launch
action value. Any use in Lyapunov design must explicitly change the guarantee
from per-step no-harm to finite-budget expected regret and still solve that
causal timing problem. The current experiments do not establish ICML readiness.

## Statistical fallacy scan

11/11 checked: task/stage splits retained (Simpson); no agent-level inference
from task aggregates (ecological); no outcome-selected environment removal
(Berkson); no outcome-conditioned covariates (collider); all 24 candidates and
8 contexts counted (base rates); independent validation prevents treating
selected extremes as truth (regression to mean); all stopped runs preserved
(survivorship); full methods reported (look-elsewhere); preregistered constants
and gates unchanged (forking paths); variance comparisons are not claimed to
cause downstream learning gains (causation); bank separation fixes temporal
precedence (reverse causality). Normal-SE gates are empirical diagnostics,
uncorrected for simultaneous inference, and are not publication confidence claims.

## Citation and novelty scope

Fresh authoritative records and resolver outcomes are preserved in
`relative_return_citation_verification_20261015.json`. Classical paired policy
search, measure-valued estimators, sequential off-policy control variates and
recent cost-aware coupling selection already exist. The present estimator
checks support continued research, not a first-coupling or ICML novelty claim.
Academic-research-suite was used for execution, replay and statistical scope.
