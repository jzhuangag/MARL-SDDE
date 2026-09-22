# TSP-MARL-CONF-001 preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-10
- Verification Status: PREREGISTERED-DESIGN
- Version Label: confirmation_v1

## Purpose

This experiment confirms the positive DEV-002 mechanism on disjoint seeds.  It
tests whether a fully charged, return-free Lyapunov probe preserves the strong
fixed `q=8` action under independent rollout innovations and switches to the
`q=1` action under shared innovations, improving equal-resource MAPPO learning
without sacrificing the regime-specific fixed action.

DEV-002 was used only to decide that confirmation was worth running and to set
the sample size and frozen practical-effect floors.  No DEV-002 seed appears in
confirmation, and no confirmation result may change the design below.

## Frozen methods and data

- Task: discrete PettingZoo MPE `simple_spread_v2`, MAPPO, CTDE.
- Upstream HARL commit: `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- Methods: frozen Lyapunov probe-then-commit controller, fixed `q=1`, fixed
  `q=8`; actor and critic rates are `0.0005` for every method.
- Regimes: marginal-preserving independent and shared rollout innovations.
- Controller: 128 non-learning `q_probe=8` blocks, one-sided correlation upper
  proxy, Lyapunov score over `{1,8}`, one irreversible commit.
- Budgets: 5,000,000 messages and 1,000,000 environment ticks; server overhead
  100 and rollout length 25.  Probe and training are both fully charged.
- Training seeds: `96001--96008`.
- Disjoint probe seeds: `97001--97008`.
- Total: 2 regimes x 8 seeds x 3 methods = 48 runs.
- Primary outcome: unsmoothed deterministic team-return AUC over 21 equally
  spaced points of the maximum charged dual-budget fraction.

The normal upper proxy is an empirical plug-in sensor for this nonlinear
benchmark.  The experiment makes no claim that it is a time-uniform Markov
confidence sequence.  The theoretical TSP guarantee remains the affine
delayed Markov result under its stated assumptions.

## Co-primary estimands

For seed `s` and regime `r`, define

```
g_r,s(q0) = (AUC_controller - AUC_fixed-q0) / |AUC_fixed-q0|.
```

With eight pairs, the registered one-sided 95% Student lower bound is

```
mean(g) - 1.894578605061305 * sample_std(g) / sqrt(8).
```

All conditions form an intersection-union success rule:

1. independent `g(q8)` lower bound is greater than or equal to -2%;
2. shared `g(q8)` lower bound is nonnegative and its mean is at least +4%;
3. shared `g(q1)` lower bound is greater than or equal to -2%;
4. the per-seed equal-weight mean of independent/shared `g(q8)` has a
   nonnegative lower bound and mean at least +2%.

Each co-primary component is evaluated one-sided at alpha 0.05.  Since success
requires all component nulls to be rejected or all margins to be met, this is
an intersection-union rule rather than a choose-the-best multiple comparison.
No failed component may be dropped.

## Operational gates

- Exactly 48 unique finite records; exact accounting and clean pinned upstream.
- At least 7/8 independent controller runs choose `q=8`.
- At least 7/8 shared controller runs choose `q=1`.
- Probe messages are at most 1% of the message budget.
- Scalar selection overhead is at most 5% of training time.
- The frozen marginal-preserving coupling audit passes.

Any mandatory gate failure yields `stop`, retains all outcomes, and prohibits
replacement confirmation seeds or a manuscript return claim.

## Power and precision rationale

The four-seed development shared effect relative to fixed `q=8` was 6.34% with
sample standard deviation 2.86 percentage points; the independent cost was
0.413% with standard deviation 0.044 percentage points.  Eight new pairs give
an expected shared standard error near 1.01 percentage points if that
development variance transfers, while doubling the number of unseen seeds and
keeping the GPU allocation bounded.  This is a planning calculation, not a
guarantee of a positive confirmation.

## Frozen provenance

- Machine-readable design SHA-256:
  `536a613468b14f067448b9744f30c5e42e154983adeabecb9401baedb0885e3c`.
- Controller SHA-256:
  `14e1f3941f734a7652726251aacb7f59fb21485b9e1387cf29100722d0c2b9c9`.
- Runner SHA-256:
  `732b859408d4116729b81323280f842a5676f79de4ed69111d15d4d8b8be0877`.
- Confirmation analyzer SHA-256:
  `f09be47a526b1222472c2e6d961a0ad95df160404a27f046c4278328e0107966`.
- DEV-002 gate SHA-256:
  `0f12063dc9725308a2d1c24174360ece9ef6cd295a31a1bcbfdf0a710d585c04`.

Execution belongs exclusively under
`/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001`.  No output is written to
`/project` or `/home`.  The 48-cell array may be submitted in smaller batches
to satisfy QOS limits, but cell indices and all scientific content remain
unchanged.

