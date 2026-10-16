# Full-horizon learning backend: prefix transport, scale-adaptive risk and reservations

## Material Passport / present decision

ARS experiment/research design; 2026-10-16. This is a separate science backend,
not an amendment of qualification0558f08, not a new registered study and not
positive MARL learning evidence. No new native data have been collected.
Existing six+six integration cells and all TSP manuscripts remain unchanged.
The complete multi-method learning benchmark is NOT implemented or registered
yet. The correction below closes specific interfaces needed by that backend;
software regression is not a substitute for its development decision.

## Full-trajectory overlap is the wrong long-horizon default

The qualified estimator uses one entire-trajectory importance weight on every
gradient term. Even though each defensive weight is at most2, later policy
divergence can reweight rewards received much earlier in the episode. Bounded
weights do not imply useful covariance or equal-cost improvement.

For a half/half mixture, write p and u for target/source trajectory densities.
For f=(p/q)h_theta and c=(u/q)h_mu, their covariance trace is

    kappa = integral (p*u/q) <h_theta,h_mu> - <g_theta,g_mu>.

Nearly separated trajectory supports can make this quantity nonpositive even
with perfectly finite weights. In the exactly separated limit, beta=0 is
optimal when <g_theta,g_mu>>=0, and

    sigma_f^2 = 2*sigma_theta^2 + ||g_theta||^2.

Paying N source episodes then gives variance at least sigma_f^2/m, worse than
sigma_theta^2/(N+m) for a matched fresh PG batch. This is a structural limiting
case, not an outcome measured on qualification gradients, and not a theorem
that all nonzero delays fail. It rules out using weight boundedness alone as
the entry argument for a long-horizon science study.

## Exact prefix correction implemented in the NEW backend

The mixture still selects ONE COMPLETE JOINT POLICY for the entire episode.
Let P_{theta,t},P_{mu,t} be probabilities of history through action a_t, and
Q_t=(P_{theta,t}+P_{mu,t})/2. Transition/initial factors cancel in all ratios.
For joint score s_{theta,k}=sum_i grad log pi_theta^i(a_k^i|o_k^i), define

    F_theta = sum_t gamma^t (P_{theta,t}/Q_t) R_t sum_{k<=t}s_{theta,k}
              - sum_t gamma^t (P_{theta,t}/Q_t) b_theta(s_t)s_{theta,t}.

Each reward term changes measure only through its own observed history.
The state baseline is frozen before data and independent of the current action.
Therefore E_Q[F_theta]=grad J(theta); the baseline terms each integrate to zero.
Define C_mu analogously, with E_Q[C_mu]=grad J(mu). The one-use independent
source mean A remains unchanged in role, and

    d = mean(F_theta) + beta*(A-mean(C_mu))

is conditionally unbiased when beta,m precede the anchor and bridge.
Use on-policy Monte Carlo PG for A; its representation as reward-to-go or as
the prefix reward-score sum is algebraically identical when all weights are1.
The qualified native code is not silently relabeled as this prefix variant.

Implementation: experiments/sealed_bridge_learning/prefix.py. It evaluates
real categorical/tanh-Gaussian actor networks on supplied complete histories,
uses every teammate's log likelihood, detaches ratios and frozen baselines,
and constructs only two per-episode backward graphs. Prefix scalar operations
are O(H), not H separate full backwards. A tempting alternative multiplying
reward-to-go by the single prefix weight at its starting action is generally
biased; it is explicitly not implemented. Later divergence no longer changes
earlier reward terms. There is NO asserted total-variance dominance: reward
terms can correlate, and late-prefix overlap can still be poor.

This is an application of established per-decision importance-sampling/control-
variate ideas, not a claim of a new DR identity or first variance-reduced PG.
[Huang and Jiang, ICML2020](https://proceedings.mlr.press/v119/huang20b.html)
is an authoritative primary reference for the differentiated off-policy value
estimator connection. [Cesani, Papini and Restelli, October2026 preprint]
(https://arxiv.org/abs/2610.13692) already combines recursive PG and defensive
sampling. The existing eight-reference verification report remains preserved;
these two authoritative records were freshly checked for this design note.

## Scale-adaptive meta update with an explicit regret interface

Current option probabilities p_t and queues precede all current data.
After the transaction, all-option unbiased variance feedback yields

    lhat_t,j = risk_weight*rhat_t,j + Q_t.c_t,j.

On the simplex, subtract the common mean without changing comparator regret:
z_t=lhat_t-mean(lhat_t). With D=sqrt(2), S_t=epsilon+sum_{s<=t}||z_s||^2,

    eta_t=D/sqrt(S_t), p_{t+1}=Proj_simplex(p_t-eta_t*z_t).

The CURRENT observation changes the NEXT probability vector only; there is no
same-bank minimization. For any fixed simplex comparator u, the standard
projection inequality telescopes with nonincreasing eta_t to

    sum_t <p_t-u,z_t> <= D^2/(2*eta_T) + .5*sum_t eta_t||z_t||^2
                       <= 1.5*D*sqrt(S_T).

The last inequality uses x/sqrt(a+x)<=2*(sqrt(a+x)-sqrt(a)). No bounded-loss or
clipped-feedback assumption is hidden. Finite second moments of lhat suffice
for its expected version; unbiased sample-variance feedback requires the
declared fourth moments of episode gradients. Queue constraints and risk
guarantees still need their own Slater/moment assumptions. The loss coefficient
is not a known neural L merely because it is named risk_weight.

Actor-step scaling is NOT solved by this meta result. Before registration,
declare a fair prospective actor learning-rate/normalization rule and its
fresh-PG controls; only steps satisfying the actual smoothness assumptions
support the earlier positive stationarity weights. A tuned rate is not a
certificate of an unknown neural smoothness bound.

## Exact-budget reservation / terminal behavior

Implemented NativeBudget in experiments/sealed_bridge_learning/control.py:

1. At launch reserve the source cohort AND its minimum future bridge.
2. Debit source sampling irrevocably before it starts; a failure never refunds it.
3. At receipt restrict bridge counts to those affordable after honoring ALL
   other in-flight reservations. Consume each source ID once.
4. Stop launches when a full source-plus-receipt cannot be reserved; drain all
   in-flight sources. Source costs never disappear at the terminal horizon.
5. Only then exhaust the remaining whole episodes with an explicitly declared
   terminal fresh-PG update. Do not truncate episodes or leave a method with
   less training data to fake equal-budget exhaustion.
6. Benchmark and same-world frozen-policy episodes have distinct additional
   charges. They never alter source/controller seeds or choice probabilities.

This ledger is only the native reservation component. Full proposal/critic/
copy/payload/CPU ledgers, launch-service scheduling, all strong learning methods,
real evaluation and the driver/analyzer/replay must still be integrated before
any scientific freeze. The ledger does not claim wall-clock asynchronous speedup.

## Bounded next decision, not another auxiliary qualification

Proceed with one COMPLETE learning backend incorporating the prefix primitive,
adaptive meta update and reservations. Do not run another estimator-only or
outcome-free native study as a claimed scientific improvement. No science
preregistration is issued from these software checks alone.

The next actual science package must retain two MPE tasks and official native
HalfCheetah, meaningful task-specific horizons, the declared fresh HAPPO LR grid,
MAPPO, delayed/online-KL, joint MA-Trace and accurately labeled defensive-VR-PG
adaptations, fresh-PG diagnostic, substantive ablations, all native/work/logical-
payload costs and the same registered task-level performance gates. A cost-only
runtime forecast must fit the existing12000s one-worker ceiling; otherwise
report the resource-feasibility decision, not tiny inert locomotion evidence.

This development direction remains separate from V3's confirmed TSP contribution.
Its current engineering/theory progress cannot be added as a positive return
result. No GPU/HPC4, manuscript edit, old-source edit or scientific retry occurs
in this backend implementation step.

## Verification in this implementation step

Nine new/previous software regressions pass (4 new+5 frozen). Exact finite
two-actor/three-step path enumeration checks target and source gradient means,
including an action-independent state baseline; it is algebra, not native data.
The remaining checks cover prefix causality, the stated finite-sequence meta
regret bound, constant-offset invariance and in-flight/native-budget exhaustion.
Every one of the seven original qualification source fingerprints is unchanged.
No qualification driver, replay or analyzer was restarted. The current app
terminal has no attached session and no MARL native worker is live. An unrelated
work/monitor_job.py process remains untouched. Full native neural integration,
learning performance and scientific cost replay of this NEW backend are untested.
