# Validation: strategic-clock exact problem gate

Date: 2026-09-06

## Decision

**STOP strategic-clock equalization as an ICML mainline.**  The frozen
problem-level gate fails G2 and G4--G7.  No stochastic efficacy run, formal
seed registry, standard MARL benchmark, GPU job or HPC4 operation is
authorized.

This is not an execution failure.  The exact run completed normally with
24,832 method--seed rows over 194 scenarios.  All values are finite, update
mass never exceeds the registered cap, the performance term changes the
count-only action on 69.59% of primary events, and terminal debt is small.
The failure is that the proposed target is not a useful learning objective
against strong baselines.

## Frozen provenance

- problem/threshold preregistration commit: `defee69`;
- implementation freeze commit: `024e732`;
- manifest SHA-256:
  `050396042D89F44C71C7566852CA7C912B3F44066CBC09D0B5E0BECF68E67C8E`;
- runner SHA-256:
  `D9E340A88243A601A50F9350FC59E77E5BD96EEF02E7D750B6465A30F24CB747`;
- result SHA-256:
  `1D7C95F6EE25DD4858448D9F197E05B2C6321B784AD6E22FF4A49928FA6EC2C6`;
- seeds: 91001--91016;
- command:

  ```text
  .\.venv\Scripts\python.exe -m experiments.strategic_clock_equalization.exact_gate --manifest docs\strategic_clock_oracle_gate_manifest_20260906.json --output tmp\strategic_clock_equalization\primary\summary.json
  ```

- local CPU only; no remote storage or GPU;
- targeted tests before the run: `6 passed in 0.64s`.
- targeted tests after the run: `6 passed in 2.29s`;
- repository test root: `1437 passed, 7 skipped in 2330.53s` from
  `python -m pytest -q experiments`.

The 299,028-byte raw summary remains under ignored `tmp/` and was not edited
after the run.

An unrestricted `pytest -q` is not a valid repository regression command in
this checkout: it also discovers retained repository snapshots and third-party
HARL tests under ignored `tmp/`, causing duplicate-module and missing optional
dependency collection errors.  Those provenance directories were not deleted
or altered; restricting collection to the repository's actual `experiments/`
test root gives the clean result above.

## Frozen results

| Quantity | Registered requirement | Observed | Result |
|---|---:|---:|---:|
| finite/accounting validity | all finite, capped | all finite; max mass `0.2` | pass |
| median primary raw-to-sync AUC loss | at least `0.05` | `0.00521352` | **fail** |
| median recovery on active cells | at least `0.50` | `1.90703` | pass |
| additional recovery over strongest causal baseline | at least `0.10` | `-2.81344` | **fail** |
| active cells directionally better than strongest causal baseline | at least `0.60` | `0/12 = 0` | **fail** |
| maximum balanced-control loss | at most `0.01` | `0.317976` | **fail** |
| maximum single-basin/uncoupled-control loss | at most `0.01` | `0.368162` | **fail** |
| primary action differs from count-only debt | at least `0.10` | `0.695858` | pass |
| performance signal active | positive | `0.954582` | pass |
| maximum terminal debt per event | at most `0.01` | `0.00150717` | pass |

G10 byte reproduction was not run because all preceding scientific gates were
mandatory.  The unreproduced output is design-stage evidence only.

## Why the apparent 190.7% recovery is not success

Only 12 primary cells have the registered minimum raw-to-synchronous loss.
They are all stationary 80/20 arrival cells.  In those cells, the Lyapunov
rule often moves beyond the under-served synchronous reference and therefore
reports recovery above one.

However, the population-selected best fixed block scaling is the strongest
causal comparator in every one of the 194 cells.  It beats the Lyapunov rule
in all 12 headroom-active cells, and by much more than the original
raw-to-synchronous loss.  Thus the large recovery statistic measures an easy
reference, not useful adaptation value.

The controls expose the same defect.  A method whose goal is path fidelity
should be nearly free under balanced clocks and within a single basin.  The
observed maximum normalized AUC costs, 31.80% and 36.82%, are incompatible
with that contract.

## Scientific interpretation

The rate-distortion phenomenon is real, but it is not enough to define an
ICML algorithmic problem.  In this cooperative one-state class:

1. raw asynchrony usually has little loss relative to the declared synchronous
   path;
2. when a loss exists, a fixed block scaling selected without seed/path
   foresight dominates the proposed dynamic rule; and
3. preserving one arbitrary synchronous discretization is not equivalent to
   maximizing learning performance.

Changing the synchronous step, tuning `V`, reducing controls, selecting only
the 12 active cells or removing best-fixed scaling after observing this result
would violate the frozen gate.  The strategic-clock candidate is therefore
closed rather than renamed.

## Consequence for the project

The project should not infer that all asynchronous MARL research is
impossible.  It should infer a narrower design rule: **exogenous update-clock
distortion is a mechanism, not yet a learning objective with broad intrinsic
headroom.**  A future mainline must begin with a task whose desired quantity
cannot be matched by static scaling and whose primary metric is meaningful
independently of a chosen synchronous discretization.

No further controller search is authorized from this result.  The next work
should be a paper-level decision between (i) the already proved finite-commit
information--safety limits chain and (ii) a genuinely new application problem
with an exogenous constraint or nonstationary target that is scientifically
motivated before any method is designed.  Neither option is automatically an
ICML paper.
