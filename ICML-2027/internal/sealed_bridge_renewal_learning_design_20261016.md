# Complete native learning backend and the paid-exposure correction

## Material Passport / bounded status

ARS autonomous complete-algorithm implementation,2026-10-16. This is a NEW
unfrozen science backend under experiments/sealed_bridge_learning. The qualified
0558f08 implementation, six+six original/replay cells, all stopped studies and
every TSP manuscript remain untouched. No new native environment was created,
reset or stepped in this implementation turn. No native scientific result,
registration, learning superiority or ICML-readiness is claimed.

Implemented one callable complete-cell backend with13 learning methods,
exact native exhaustion, independent launch/service streams, one-use anchors,
real evaluation and same-world frozen-policy controls. The source-locking
driver, cost-only profiling gate, deterministic replay owner, complete analyzer
and plotting package still need integration BEFORE scientific registration.
Do not invoke benchmark.run_cell on native data without that separate freeze.

## A substantive finite-budget problem caught before science

The earlier unfrozen draft priced work/bytes per receipt and applied the same
SGD step to receipts with different m. A smaller bridge then buys more policy
updates from a fixed native budget. Minimizing per-receipt variance plus per-
receipt resource costs does not by itself optimize learning per native budget.
This is not an old scientific failure being retried: no native data exist for
the science backend, and the qualified earlier package is preserved unchanged.

Define a receipt's paid exposure n_j=N+m_j, including its independent source
cohort even though that cohort was paid at launch. A fresh update's exposure
is its complete paid episode count. Source reservations and drain prevent any
source cohort from disappearing at the terminal horizon. Native transitions
are n_j*H for a cell's fixed episode horizon H.

Let a_t be a reference SGD step chosen from PAST gradient RMS and a declared
cap, and n_ref the public reference episode count. Execute

    alpha_t,j = a_t*n_j/n_ref.

The reference cap is reduced by n_ref/n_max before data, so every candidate's
actual alpha is at most the declared cap. In a fixed-a phase, the sum of actor
step exposures is proportional to paid native episodes rather than receipt
count. Calibration episodes have no actor commit and remain paid; the matched
fresh-PG diagnostic uses the SAME calibration and exposure scaling.

## Predictable actor scale and its real theorem interface

A disjoint charged calibration bank initializes v_0, the mean per-coordinate
squared episode gradient; its policy is NOT updated from that bank. Critic
fitting follows the calibration gradient role. Later receipts use only past
v_t to set a_t=min(a_cap,a_base/sqrt(epsilon+v_t)); the CURRENT corrected gradient
updates v for the NEXT receipt. All actors share this predictable scalar.
No current-gradient normalization, imported Adam moments or clipped SGD
direction is silently introduced. The constructor and begin/close transaction
are implemented in updates.PredictableActorScale.

Under the previously declared smoothness/integrability/independence assumptions,
the prefix transport remains conditionally unbiased. With conditional variance
r_t,j, the exact useful inequality is

    E[Delta J | F_t,j] >= alpha_t,j*(1-L*alpha_t,j/2)*||grad J_t||^2
                         - L*alpha_t,j^2*r_t,j/2.

If the declared cap is <=1/L, every signal coefficient is positive. At fixed
or predictably budget-stopped horizons, retain the SUM of these weighted
coefficients and variances; do not divide by a random receipt count. The RMS
scale is not an estimator of L. Without the L/cap condition, the smoothness
inequality is still valid but its signal coefficient can be nonpositive.
This is NOT a high-probability no-harm, separate-HAPPO superiority, global
optimality, ordinary-Adam guarantee or practical queue-stability theorem.

## Resource control in native time rather than receipt time

For each frozen option compute exact declared training work-row and logical-
payload costs k_j BEFORE opening source/bridge data. Neural work rows are

    W_j = H*((3*A+1)*N + (5*A+3+2*E_critic)*m_j),

where A is the number of distinct actors. The terms count source+bridge actor
sampling, source plus dual bridge forward/backward gradients, both frozen value
baselines, and target preparation plus every critic epoch. Logical bytes are

    B_j = (N+m_j)*episode_program_bytes + 3*joint_policy_bytes.

The3 policy copies are the source rollout snapshot and BOTH policies sent to
the bridge rollout group. Gradients/critics are server-owned; no imaginary wire
gradient payload is charged for a server-only backward. Retained gradient
arrays, neural copies, all-option variance scalar entries, optimizer calls,
program/policy messages and actual CPU are reported separately. Work rows are
declared proxies, not FLOPs; logical bytes are not physical network telemetry.

For each resource, normalize by its maximum catalogue cost PER PAID EPISODE.
An outcome-free equal-exposure mixture of the shortest and longest options
has a known resource-rate vector r_mix. Set b=(1+r_mix)/2. This constructs an
explicit randomized Slater witness with strict slack whenever both resource
rates vary. If strict slack is absent, registration is forbidden; it is not
repaired using outcomes. These derived rate targets belong to this NEW renewal
controller, not the frozen qualification's per-receipt target constants.

Let d_j=k_j/scale - b*n_j, Q_next=[Q+d_j]_+, Bq_j=||d_j||^2/2, and
L_t=V*(Jmax-J_t)+||Q_t||^2/2. Then

    E[Delta L | F_t,j] <= -V*alpha_t,j*(1-L*alpha_t,j/2)*||grad J_t||^2
                          + V*L*alpha_t,j^2*r_t,j/2 + Q_t.d_j + Bq_j.

The variable-cost control must preserve this native clock. Maintain an
EXPOSURE distribution w on the finite catalogue; sample receipts with

    p_j = (w_j/n_j)/sum_l(w_l/n_l), Cbar=1/sum_l(w_l/n_l).

Consequently E[n_selected|F]=Cbar and
E[n_selected*l_selected|F]=Cbar*sum_j w_j*l_j. The normalized loss is

    l_t,j = rho*a_t^2*n_j*r_t,j/(2*n_ref^2)
            + (Q_t.d_j+Bq_j)/n_j.

Choosing V=rho/L in the ANALYSIS matches the noise coefficient without claiming
that code knows L. Code declares a positive rho with a shape/rate normalization;
the theoretical interpretation uses the actual finite L. The gradient signal
per exposure is at least V*a_t/n_ref*(1-L*alpha_cap/2)*||grad J_t||^2 under the
stated cap condition, independently of j. Source and terminal effects stay paid.

All-option unbiased sample-variance feedback supplies lhat only AFTER receipt.
The OGD gradient is Cbar*(lhat-mean(lhat)); post-feedback AdaGrad projection
updates w for the NEXT receipt. The earlier finite-sequence projection proof
gives, for any fixed exposure comparator u,

    sum_t Cbar_t <w_t-u,lhat_t> <= 1.5*sqrt(2)*sqrt(epsilon+sum_t||z_t||^2).

Combining the conditional identities with telescoping drift yields the
corresponding expected weighted-gradient/resource inequality plus this regret
term. It compares options on the algorithm's visited policies, not another
method's trajectory. Unbiased variance feedback needs conditional fourth
moments; it is not a confidence upper bound. Unbounded queue-dependent losses
mean the displayed self-normalized regret bound is not automatically sublinear
in T. Do not claim a queue rate until a complete self-bounding argument closes it.

## Exact finite-budget and causal implementation

- Reserve source N and the MAXIMUM bridge menu at launch, while retaining the
  true minimum feasible bridge. This keeps the same statistical catalogue
  affordable at receipt and avoids an unproved changing-domain comparator.
  Releasing unused unsampled reservation is not refunding collected data.
- Choose actor reference scale, receipt option and all policies/baselines before
  opening the one-use source gradient and current bridge. Full likelihoods use
  all teammates; prefix weights use one complete-policy coin per episode.
- Service delay is a fixed hash stream of seed and launch ordinal, independent
  of update counts, reward, scores and the selector. Source metadata include
  launch/due ticks and actual joint policy versions.
- After all bridge receipts finish, exhaust the remaining whole episodes with
  DECLARED fresh PG. Never truncate a trajectory to cosmetically fill a budget.
- Benchmark/frozen-policy evaluation uses separate nontraining roles and common
  world/action seeds between those two policies. The final return is evaluated
  only AFTER all in-flight source programs drain, not on the first receipt after
  the last source sampling payment. Evaluation cannot influence the controller.
- Gaussian source programs preserve PER-COORDINATE latent log likelihoods for
  HARL PPO. Joint likelihoods sum those coordinates. Tanh Jacobians cancel in
  ratios. Sending a scalar joint log into HARL's coordinatewise PPO ratio would
  raise it to the action dimension; the NEW backend explicitly avoids this bug.
- Source snapshots copy network/baseline tensors, not optimizer moments. The
  server owns every fresh/delayed PPO/Adam state. Repeated PPO epochs recompute
  losses/gradients on the declared data rather than reapplying a stored gradient.

## Complete learning methods implemented

Nine strong/reference controls: fresh HAPPO learning-rate factors0.5/1/2;
fresh MAPPO; delayed HAPPO; online joint-KL HAPPO; a joint MA-Trace-style adapter;
a defensive PAGE-PG adapter with prospective refreshes; and matched fresh PG.
Full prefix/renewal control and no-transport/fixed-allocation/no-price ablations
bring the total to13. The two published-method adapters are labelled adaptations,
not official reproductions or novel algorithms. HAPPO/MAPPO use real recomputed
PPO; the V-trace route uses joint clipped ratios and its OWN PG/critic targets.
The PAGE reference uses recursive full-trajectory defensive transport, not
our prefix correction relabelled as a competing method.

Primary comparisons remain the strongest registered control in EACH task.
Keep the two native MPE tasks and official MaMuJoCo HalfCheetah; do not claim
that three tasks are three families or silently import old MuJoCo-v2 results
into the new official runtime. Meaningful task horizons, total budgets, new
seeds, role catalogue and every performance/cost gate still require a separately
pushed source/config/preregistration lock before any native data.

Fresh primary checks used for adapter design: the MA-Trace record
https://arxiv.org/html/2111.11229v1 and the existing authoritative/arXiv source
verification report; the recent defensive-VR-PG record
https://arxiv.org/html/2610.13692v1 (Appendix I, actual experimental settings).
The latter uses100-step HalfCheetah-v4 episodes and a long trajectory budget;
this is not evidence that25-step MaMuJoCo qualification established learning.
Its batch/refresh constants need explicit adaptation to the fresh native
multiactor comparison and cannot be falsely described as a full reproduction.

## Verification and next bounded decision

All13 routes across3 categorical/continuous tensor fixtures pass the complete
backend regression with REAL HARL networks/losses/server updates. No environment
call is allowed: make_env is patched to raise; evaluation fixtures are constant
zero, never ranked or persisted as learning evidence. The updated renewal
backend completed these39 synthetic routes in19.841 wall seconds reported by
unittest. This timing is NOT a native runtime forecast or performance result.

The next deliverable is the source-locked complete driver/analyzer/replay and
cost-only feasibility calculation, not another estimator or native outcome-free
qualification. Keep the unchanged task-level AUC>=1%,>=2/3 positive seeds,
final>=-2%, strongest-control learning>=2%, <=1.5 declared work/logical-payload
ratios, all-cost curves and deterministic replay requirements. A prospective
one-worker whole science/replay/analysis cap<=12000 CPU seconds must include
cost profiling. If meaningful native learning cannot fit, report that resource
decision; do not shrink horizons into an inert positive-looking benchmark.

No new GPU/HPC4, cleanup, manuscript edits or scientific retries occurred.
This engineering/theory progress is not admitted to V3 as a positive experiment.
