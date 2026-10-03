# TSP-V2 learning-progress sensor G0 operational amendment 1

## Material Passport

- Scientific experiment: `TSP-V2-MARL-PROGRESS-G0`
- Execution attempt: `G0-A1`
- Original frozen scientific source: `2eaa373f3283d70bf4475173ab1365a2e617e2cc`
- Failed pre-payload array: `1914391`
- Authorization: user-authorized autonomous completion after the recorded infrastructure stop
- Scope: operational repair only

## Reason for the amendment

Array `1914391` stopped before its batch payload because the Slurm stdout and
stderr parent directory did not exist at submission time.
It produced no scientific observation, trajectory, reward, gradient, model
update, or evaluation return.
The failure therefore does not consume a scientific seed and supplies no
information about any frozen outcome gate.

## Frozen scientific invariants

This amendment does not change the environments, coupling regimes, seeds,
candidate participation levels, micro-update count, optimizer settings,
diagnostic statistic, analysis code, or mandatory gates.
The four frozen scientific files below have the same Git blob identifiers as
at the original preregistration commit:

| File | Git blob |
|---|---|
| `run_marl_progress_sensor_g0.py` | `aacef348c65062cdee9fa467e146039ad456894c` |
| `analyze_progress_sensor_g0.py` | `9c9e44ba0cbbbe8ba773574042960134d2d9ec71` |
| `progress_sensor_g0.json` | `f741c598bb93a88fc9e48f8203981ca43c90b344` |
| `progress_sensor_g0_a30.sbatch` | `6798718c5df3c2443250584bdc7d0b40840688a3` |

## Operational changes

The amended launcher uses the new, non-overwriting scratch root
`/scratch/jzhuangag/MARL-SDDE-TSP-V2-PROGRESS-G0A1`.
It creates `logs`, `artifacts`, and `tmp` before submission and refuses to
submit if any prior artifact is present.
The batch file adds absolute Slurm output and error paths below the pre-created
log directory.
All scientific command-line arguments are unchanged.

The amended execution remains outcome-isolated.
It will be analyzed twice with the frozen analyzer, and byte identity is still
a mandatory gate.
Any payload failure or scientific gate failure stops the sensor design without
retry.

