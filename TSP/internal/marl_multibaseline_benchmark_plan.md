# Multi-baseline, multi-task policy-learning plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-14
- Verification Status: FROZEN-STAGED-DESIGN
- Version Label: multibaseline_v1

## Question

The policy-learning experiment must test whether an observable Lyapunov
controller is close to the best participation level under each dependence
regime and better than a deployable, task-specific static choice under equal
physical budgets.  It must not test a preassigned claim that either small or
large participation is always preferable.

Accordingly, every admitted task uses the complete catalogue
`q in {1,2,4,8}`.  The four fixed actions are scientific baselines, not merely
controller ablations.  A per-regime fixed-q envelope is reported as a
descriptive, outcome-aware upper bound and is never labelled as a deployable
method.

## Comparator hierarchy

1. fixed `q=1` (local/small-batch endpoint);
2. fixed `q=2` and fixed `q=4` (interior participation baselines);
3. fixed `q=8` (broad-participation endpoint);
4. the task-specific strong fixed `q`, selected on development seeds and
   frozen before confirmation;
5. an uncertainty-blind plug-in ablation that uses the point dependence
   estimate rather than its upper certificate;
6. the fully charged Lyapunov controller;
7. the per-regime fixed-q envelope, shown only as an oracle diagnostic.

The main return plot will display all four fixed-q curves with thin lines, the
strong fixed baseline with a heavier line, and the controller with the
dominant line.  The oracle is summarized in a table or a short dashed marker
to avoid turning a nondeployable diagnostic into a visual baseline.

## Gates that match the scientific claim

No gate requires a particular `q` to win.  For each task and dependence
regime, the controller's return AUC must be within two percent of the complete
fixed-q envelope.  Across the registered regime mixture, it must exceed the
development-selected strong fixed `q`; the paired one-sided confirmation
interval must exclude zero aggregate gain.  At least 75 percent of task-regime
cells must be within two percent of the envelope, no task may be more than
five percent worse, and every probe/message/environment cost must be charged.

Selection accuracy is secondary: it records whether the observable controller
chooses the empirical envelope action, but a neighboring `q` that attains the
same return is not counted as a scientific failure.

## Staged task suite

The intended four-panel return figure spans more than one benchmark family:

1. PettingZoo MPE `simple_spread_v2` (existing confirmed mechanism; full-q
   baseline audit now required);
2. PettingZoo MPE `simple_speaker_listener_v3` (cooperative communication and
   heterogeneous roles; new development data only);
3. SMACv2 `terran_10_vs_10` only if the complete q-grid restores at least two
   percent regime-oracle headroom and the Lyapunov-predicted actions are within
   two percent of that envelope;
4. MAMuJoCo `HalfCheetah-v2, 2x3` only after a no-outcome compatibility and
   marginal-coupling audit closes the continuous-action interface.

`simple_reference_v2` remains a stopped historical development task and is not
silently promoted.  A different SMACv2 race or unit count is not selected from
return outcomes; any such task needs a new result-blind compatibility record
and development registry.

## Current authorized execution

Two baseline-completeness studies are authorized before any new controller or
multi-task confirmation:

- add fixed `q=2,4` to the eight held-out MPE seeds while preserving every
  original result;
- add fixed `q=2,4` to the two SMACv2 development seeds to test the interior
  optimum predicted by the return-free correlation calibration.

Both write only to isolated scratch roots.  Passing permits, but does not
itself constitute, a fresh multi-task preregistration with new seeds.
