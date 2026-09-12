# EXP-016A preregistration: finite-budget learning-value threshold

## Material passport

- Evidence class: CPU mechanism pilot, not formal evidence.
- Preregistration parent: `41bd4c696b49c9876cff537d3fb03c571393a7b2`.
- Frozen configuration SHA-256:
  `bb3ab51bc64d4ee334e7c5da6b6e7a4e7ffd303692abb6a5e48d06e48bb9baf5`.
- Pilot seeds: 64 fresh seeds specified in `exp016a_seed_registry.json`, all
  permanently excluded from formal use.
- Formal seeds: 128 separately specified seeds inaccessible to the pilot
  entry path; no formal runner is authorized by this commit.
- Scientific outcomes present at preregistration: **none**.
- Recommended hardware: local CPU. HPC4 and GPU are neither required nor
  authorized.

## Frozen theory boundary

EXP-015A remains an honest 7/8 failure: its `0.80` gate and observed
`0.777778` are unchanged. AC-8 closes unrestricted unknown mixing
negatively. Positive cells require public \(\lambda\leq1-\gamma\), here
\(\gamma=0.05\). AC-9 supplies only a compact-separated finite-budget
threshold sandwich; it does not match the entire controlled-belief occupation
optimum. Generic controlled-sensing change-of-measure machinery is inherited.
No SDDE-to-discrete convergence theorem is claimed.

## Research questions

1. Below the necessary threshold \(B_N\), does the theorem-derived controller
   execute its deterministic fallback?
2. Above the sufficient threshold \(B_S\), does it improve high-regime
   empirical downstream learning risk while satisfying low-regime safety?
3. Does the first empirical positive-gain budget fall inside the frozen
   \([B_N,B_S]\) bracket?
4. In delayed dual-budget cells, does a learning-risk-aware rule beat an
   inherited information-per-control-cost rule on downstream oracle regret?
5. Do no-delay, single-budget, and no-mixing-correction ablations fail in the
   prespecified directions?

RQ4 is the novelty gate. Failure of G8 stops the adaptation-cost ICML route.

## Frozen Gaussian experiment

Individual observations are

\[
X_{i,t}=\mu+C_t+\epsilon_{i,t},\qquad
C_{t+b}=\lambda^bC_t+
\sqrt{1-\lambda^{2b}}\,\xi_t,
\]

with stationary \(C_t\sim N(0,\theta)\), unit private variance, and evaluation
coordinate \(\mu=0\) without loss of generality for the frozen
translation-equivariant sample-mean learner. The numerical value of \(\mu\)
is not a controller input and hard-coded zero output is prohibited.
Controllers receive only actual requested individual
observations, registered hypotheses, public \(\lambda\), actions, and
remaining resources. The downstream learner uses the committed action's
actual observations and harmonic mean update (`eta=1.0` multiplier). It does
not receive hidden regime, true \(\theta\), latent \(C_t\), or oracle action.

The positive manifest freezes:

- \(Q\in\{8,16,32\}\);
- \((\theta_0,\theta_1)\in\{(.05,.5),(.05,2)\}\), so
  \(\Delta_{\min}=.45\);
- \(\lambda\in\{.2,.7,.94\}\leq.95\);
- \(D\in\{0,4,12\}\), overhead \(h\in\{4,16\}\);
- balanced, message-limited, and environment-limited positive dual-budget
  rays;
- `epsilon_safe in {0.10,0.20}`;
- finite catalogue `q in {2,4,8,16,32}` clipped to \(Q\),
  `b in {1,2,4,8}`, `eta=1.0`;
- relative oracle coefficient gap at least `0.03` (actual manifest minimum
  `0.16`).

There are 54 positive base scenarios. Each has low/high transition sides and
five budget points: `floor(.5 B_N)`, `floor(.9 B_N)`, the floor midpoint of
`ceil(B_N)` and \(B_S\), `ceil(1.1 B_S)`, and `ceil(2 B_S)`. Message and
environment budgets are independently `floor(scale × beta)`. In the frozen
manifest \(B_N\in[2.5,40]\) and \(B_S\in[29,572]\); every midpoint is truly
gray-zone.

## Frozen thresholds

For every `(q,b)` probe, both directional KLs determine the necessary sample
count and the Bhattacharyya bound determines the sufficient count at
`delta=0.025`. The raw necessary scale is

\[
B_N=\min_{q,b}\max\left\{
\frac{n_{LB}(h+q)}{\beta_m},
\frac{n_{LB}b+D}{\beta_e}\right\}.
\]

\(B_S\) is the first integer scale at which a frozen sufficient probe, real
delay, both costs, wrong-commit term, low-regime relative safety bound, and
at least `0.005` analytic high-regime gain all qualify. Thresholds,
qualification margins, probes, oracle actions, budget points, and rounding
are stored per scenario in `exp016a_scenario_manifest.json`. They may not be
changed after outcomes are viewed.

The manifest also freezes \(B_{oracle}\): the first known-high-instance
integer scale at which a sufficient probe plus real delay can be amortized by
the correct high-regime action, before wrong-commit and low-regime safety are
charged. Every scenario satisfies
\(B_N\leq B_{oracle}\leq B_S\).

## Policies

Ten policies are frozen:

1. infeasible true-instance oracle, evaluation only;
2. always-all (`q=Q,b=1`);
3. fixed-small action `(q,b,eta)=(2,1,1.0)`;
4. EXP-015A paid ETC, unchanged;
5. theorem-derived learning-aware controller;
6. information-only controlled-sensing baseline, which cannot access
   downstream risk, wrong-commit loss, safety slack, or delay amortization;
7. no-delay decision ablation: threshold/qualification receives `D=0`, but
   execution and charging use the true public delay;
8. ignore-message-budget ablation: planning treats message budget as
   unbounded, then a deterministic prefix projection stops before either real
   budget would be exceeded;
9. ignore-environment-budget ablation: the symmetric planning error and the
   same deterministic legal-prefix projection;
10. no-mixing-correction ablation: planning and likelihood use `lambda=0`,
    while trajectories retain the public true mixing value.

Only the oracle may access the true instance. No non-oracle API accepts true
\(\theta\), true regime, latent state, or oracle action.

## No circular validation

Empirical outcomes must be generated from fresh common-factor trajectories,
actual individual observations, actual tests and actions, actual downstream
updates, and trajectory-level resource use. Analytic catalogue risks may not
be copied into observed MSE/CVaR. Hidden parameters may not select actions.
No pilot result may change thresholds, seeds, gates, or gray-zone membership.

The CRN key aligns potential observations by physical time and agent, not by
policy step. Thus action-dependent paths do not reuse unobserved future data.

## Negative controls

Two five-scale families are frozen outside the positive theorem population:

- `lambda=0.9999`, illustrating the near-nonmixing AC-8 trend;
- `theta gap=.005` and target oracle gap `.001`, illustrating AC-9
  degeneration.

They are explanatory only and cannot empirically prove either impossibility
theorem or contribute to G2--G10 success.

## Gates, statistics, and progression

All G1--G12 in `exp016a_gate_table.json` are mandatory. There is no “k of n”
override. The cell rules, 20,000-resample paired max-t procedure,
Clopper--Pearson/Holm directional bounds, CVaR90 estimator, zero-event upper
bound, 2%/3% effect thresholds, and 75% break-even proportion are frozen in
`exp016a_analysis_plan.md`.

EXP-015A cannot be reinterpreted. Formal work is prohibited unless every
pilot gate passes and a separate formal authorization commit is made.

## Static workload estimate

- expanded cells: 550 (540 theorem-scope plus 10 negative-control cells);
- policies: 10;
- pilot seeds: 64;
- estimated trajectories/rows: 352,000;
- single-process CPU wall time: 1.17 hours;
- peak memory: 1.5 GB;
- disk: 0.493 GB.

These are static operation/row estimates, not timed scientific outcomes.
They are below all HPC4 triggers (6 hours, 32 GB, 20 GB), so the registered
recommendation is local CPU.

## Authorized commands in this commit

Only static commands were executed:

```bash
python experiments/dependence_delay_linear/run_exp016a.py validate
python experiments/dependence_delay_linear/run_exp016a.py estimate
python -m pytest experiments/dependence_delay_linear/test_exp016a.py -q
```

`emit` and `freeze-manifest` are deterministic configuration tools. This
preregistration-stage runner intentionally exposes no pilot or formal
scientific-run command. The next agent may implement/execute the pilot only
after this independent preregistration commit, without changing frozen JSON
or analysis rules.

No scientific trajectory, row, aggregate, outcome, or result directory was
generated during preregistration.

## Preregistration verification

- deterministic `validate`: passed with the frozen SHA-256 above;
- static `estimate`: passed and reported `scientific_outcomes_generated=false`;
- EXP-016A static audits: 15 passed;
- complete repository suite: 171 passed in 8.91 seconds;
- all seven documentation JSON files parse successfully;
- no EXP-016A result directory or scientific output exists.
