# TSP-V2 learning-progress sensor G0 preregistration

## Material Passport

- Experiment: `TSP-V2-MARL-PROGRESS-G0`
- Role: outcome-isolated interface and sensor qualification
- Status: frozen before HPC4 execution
- Scientific output: no manuscript performance claim

## Purpose

The old dependence-only controller correctly detects rollout correlation but
cannot represent task-dependent interior participation.  This G0 asks whether
a short, fully charged micro-training branch exposes a finite and reproducible
learning-progress signal for all `q in {1,2,4,8}` on both the existing
PettingZoo task and MaMuJoCo HalfCheetah.

Each branch starts from byte-identical actor and critic parameters.  It performs
24 learning updates and records the mean reward observed during those updates.
It does not invoke deterministic evaluation, compute the registered return AUC,
or modify any old experiment artifact.  The diagnostic selection maximizes the
mean reward over the last half of the micro-training branch; this selection is
not yet the final certified V2 controller.

## Frozen cells

- PettingZoo `simple_spread_v2`, independent and shared coupling.
- MaMuJoCo `HalfCheetah-v2/2x3`, independent and shared coupling.
- Two new sensor seeds per task and regime: `140101` and `140102`.
- Candidates `q={1,2,4,8}`; 24 updates per candidate.
- Eight GPU tasks in total; no evaluation-return job.

## Mandatory qualification gates

1. Every candidate branch starts from the same parameter SHA-256 within a cell.
2. Every reward trace contains 24 finite entries and no evaluation return.
3. Exact charged message and environment counts match the public cost formula.
4. Independent MaMuJoCo selects `q=8` in at least one of two seeds and never
   selects `q=1` in both seeds.
5. Shared MaMuJoCo selects an interior action `q in {2,4}` in at least one of
   two seeds.
6. PettingZoo produces finite, nonconstant progress traces in both regimes.
7. A clean replay of the analyzer is byte-identical.

Failure of a gate stops the current sensor design before any full-return
development run.  Passing G0 authorizes design of the coefficient certificate;
it does not authorize manuscript inclusion or confirmation.

