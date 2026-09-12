# T-032 preregistration: exact fresh-diversity full-risk ceiling

## Status

This commit freezes the exact analytic scan before any T-032 scientific
outcome exists.  `validate` is permitted; `run` is permitted only after this
design is committed.  No sampled trajectory, pilot seed, GPU, HPC4 job, or
formal claim is created.

## Question

Does equal-count identity-aware selection have at least 15% complete
finite-time learning-risk value on a delayed jointly Markov subclass, after
strong single-factor baselines and all three budgets are charged?

This is deliberately stricter than a variance-resource scan.  For every
candidate subset, the runner propagates the exact mean contraction and the
exact covariance induced by the delayed parameter recurrence and stationary
AR(1) common/private factors.  The output metric is parameter-MSE AUC and
terminal MSE.

## Frozen design

- pools: 16 and 32 agents;
- equal-count decisions: `m in {4,8}`;
- four-agent dependency blocks;
- within-block long-run dependence levels `rho in {0,0.9}`;
- temporal Markov coefficients `lambda in {0,0.8}`;
- zero-delay and heterogeneous-delay populations;
- three frozen assignments preserving group sizes and the same delay
  histogram: balanced, clustered, and hash-permuted;
- target horizons 64 and 256;
- message-, total-environment-, and wall-binding rays;
- 576 analytic cells and zero scientific trajectories.

Every single-agent innovation has stationary variance one and the same AR(1)
autocorrelation regardless of its dependency group.  Changing the layout
changes only the joint law and staleness assignment, not the marginal target
operator.

## Policies and ceiling

The strong baseline is selected cellwise from diversity-only,
freshness-only, fixed-ID, and the mean of four frozen uniform subsets.  This
is intentionally favorable to the baseline.  The conservative oracle ceiling
selects from those policies and a frozen family of joint greedy subsets with
delay tradeoff values `{0,0.25,1,4,16}`.  It is a template ceiling, not a
deployed algorithm or formal evidence.

The exact block-selector theorem and a theorem-derived delay multiplier are
not claimed by T-032.  They are authorized for development only if every
static gate passes.

## Mandatory gates

The machine-readable versions are in `t032_exact_full_risk_manifest.json`.
The decisive gates are:

- at least 15% aggregate geometric MSE-AUC improvement in the active
  high-dependence, heterogeneous-delay population;
- at least 70% of active cells improve by at least 5%;
- homogeneous controls stay within 2% of the strong baseline;
- at least three distinct oracle structures;
- every resource ray contains a 5% improvement path;
- median count-only separation at least 1.10;
- exact finite values, zero resource violations, no old-outcome input, and
  CPU-only execution.

Any failed mandatory gate stops the selector theorem/pilot progression under
this identifier.  Gates, grids, layouts, and thresholds may not be amended
after the run.

## Provenance placeholders

- configuration SHA-256:
  `826bc4fde0231a017a719ddd76e937916400a1018e5815a69b97769b37511c25`;
- preregistration commit: populated by the commit itself;
- result directory: absent at preregistration;
- old EXP-017A--019A outputs: prohibited inputs.
