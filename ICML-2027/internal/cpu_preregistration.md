# CPU learning-value qualification preregistration

## Material Passport

- Origin: academic-research-suite / experiment-agent.
- Mode: authorized autonomous design and run.
- Date: 2026-10-13.
- Status at creation: PREREGISTERED-DESIGN; no development calibration observed.
- Experiment: ICML27-LEARNING-VALUE-CPU-001.

## Question and technical change

Does a signal-, noise-, mixing-, delay-, and horizon-aware observable score
recover complete-catalogue participation decisions under paid calibration?
The score replaces correlation-only variance per update by exact finite-horizon
discrete TD risk. The evaluator knows the true context; the controller receives
only calibration observations and public dynamics and resource costs.

The proof uses Gaussian whitening with exact t and chi-square intervals.
These intervals cover every candidate risk simultaneously. The certified
selection regret is a true-risk difference, computed from an upper and a lower
envelope rather than the difference of two upper envelopes.

## Frozen population and methods

The JSON defines 360 contexts, 16 fresh development seeds, q={1,2,4,8}, three
sampling spacings, six public steps, and eight methods. Both white and nonzero
temporal correlation, intermediate and extreme spatial dependence, two signal
levels, two delays, two server overheads, and three actor-transition budgets
are included. The mathematical model is fixed-policy constant-feature TD with
discount zero, not a cooperative policy-improvement benchmark.

Strong fixed-q comparators have full budgets, no calibration tax, and true-
context optimal step and spacing. Their privileged information is stated
explicitly. The strongest single q is the minimizer of aggregate geometric
risk over the fixed public population; this does not use calibration outcomes.
The paid oracle is diagnostic and cannot be executed by the proposed method.

## Estimands and gates

Primary outcome is exact conditional terminal MSE over fresh learning streams
after random calibration. Every registered context contributes. Paired
inference first aggregates all contexts within each calibration seed. The
JSON's thresholds are fixed before outcomes and are never adjusted to rescue
the result. Confidence intervals on aggregate ratios describe seed variation;
the exact t/chi-square parameter certificate has a separate analytical coverage
claim. Interior-q qualification uses an outcome-free exact risk gap of 3%.

Monte Carlo trajectory checks use 4,096 fresh learning replicas on the three
prespecified persistence-.9, delay-2 curve contexts at rho={0,.5,1}; they check
the exact risk evaluator and are not used to fit or choose actions.

Replay requires byte-identical endpoints, certificates, curves, Monte Carlo
tables, and summary. Wall time and host provenance are excluded from that
comparison. Any mandatory gate failure preserves the complete result and stops
confirmation under this identifier.

## Resources and commands

Local CPU only. No remote action or GPU allocation is needed for this stage.

```powershell
D:/anaconda/envs/ust2/python.exe ICML-2027/experiments/run_cpu.py --phase development --out ICML-2027/results/cpu_development_20261013
D:/anaconda/envs/ust2/python.exe ICML-2027/experiments/run_cpu.py --phase development --out ICML-2027/results/cpu_development_20261013_replay
```

The reserved 64 confirmation seeds cannot run until the complete development
decision passes, replay is verified, and a separate confirmation lock is
committed. The TSP manuscripts and their prior claims are outside this patch.
