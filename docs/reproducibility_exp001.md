# Reproducibility report: EXP-001

## Material Passport

- Experiment ID: `EXP-001-dependence-delay-linear`
- Verification status: VERIFIED
- Verification date: 2026-07-29
- Runtime: Python 3.8.12, NumPy 1.20.3, SciPy 1.7.3, pandas 1.4.1,
  Matplotlib 3.5.1
- Canonical result:
  `experiments/dependence_delay_linear/results/baseline/`
- Independent rerun:
  `experiments/dependence_delay_linear/results/reproduction/`

## Analytic implementation checks

The four deterministic unit checks passed:

1. the no-delay white-noise stationary covariance matches its scalar closed
   form;
2. sample-time and server-time alignment are identical when every delay is
   zero;
3. an unstable no-delay step is detected through the spectral radius;
4. an invalid common-factor alignment is rejected.

Command:

```powershell
python -m unittest -v test_linear_model.py
```

## Exact-versus-Monte Carlo check

Three selected operating points used 4,000 independent replications. The
maximum relative difference from the exact finite-horizon MSE was \(2.42\%\);
all exact values were inside the corresponding Monte Carlo 95% intervals.

## Independent rerun

The baseline was rerun with the same command and seed into a distinct output
directory. SHA-256 comparison gave byte-for-byte equality for:

- `sweep.csv`;
- `best_by_setting.csv`;
- `policy_comparison.csv`;
- `monte_carlo_validation.csv`;
- `summary.json`.

The canonical baseline was then refreshed with the verified script. Its stderr
log is empty.

## Verdict

The reported EXP-001 numerical results are reproducible under the recorded
local environment. This verifies implementation repeatability; it does not
establish external validity for temporal-difference learning or nonlinear
reinforcement learning.
