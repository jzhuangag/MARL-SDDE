# MPE and SMACv2 q-grid execution freeze

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-14
- Verification Status: PRE-OUTCOME-FROZEN
- Version Label: qgrid_freeze_v1

The two q-grid extensions are frozen before any new `q=2` or `q=4` policy
trajectory is generated.  They preserve all predecessor outcomes and use only
isolated scratch output roots.

## Frozen source hashes

| File | SHA-256 |
|---|---|
| `experiments/marl_mpe_qgrid_audit.json` | `5ac4d1f090422616f7f4c5efbae996945635f9227d80a46092f7dc92bc2a81e6` |
| `experiments/marl_smacv2_qgrid_extension.json` | `abc1d50083a0ad4f122668de3da3d03b285c3ab0172257343180a72121599a08` |
| `experiments/analyze_mappo_mpe_qgrid_audit.py` | `7392a0cdc674f6b2e8aac063fc3472b95b0107d7797da911b866dacc58b9c296` |
| `experiments/analyze_mappo_smacv2_qgrid.py` | `bcf4ad8d3510d9ee0b8bef3ae6261a0e5b17f452e879e630a56d47d0a0bfcad9` |
| `experiments/run_mappo_return_bridge.py` | `85288940c8fea2a948bfc398203367f8af501cab6c67a978599afdd9ceebfeca` |
| `slurm/marl_mpe_qgrid_audit_a30.sbatch` | `055b0e27a59357c1a10285cf32d2110ec1527c50f84bd2098e22985d0cff98ce` |
| `slurm/marl_smacv2_qgrid_extension_a30.sbatch` | `c4d2044c8bcca18eedc729acd562db1d834ed888c32b949ede17485b7a467adb` |

Targeted pre-outcome regression produced `18 passed`.  JSON parsing,
Python compilation, and `git diff --check` passed.  The wider local TSP suite
has two environment-specific pre-existing failures under the machine's
Python-3.8 default (Numba dynamic-module lookup and use of Python-3.9
`str.removeprefix`); all eight manuscript-evidence checks pass under the
bundled PDF runtime.  These environment failures do not alter either frozen
q-grid design and will be rerun in the pinned Python-3.9 HPC environment.

## Scientific separation

The MPE stage is a post-confirmation baseline audit: failure withdraws the
claim that the published controller was compared against a sufficiently broad
fixed-q family.  The SMACv2 stage is an exploratory landscape completion:
passing permits only a new controller preregistration with new seeds.  Neither
stage changes an old gate or promotes old development data to confirmation.
