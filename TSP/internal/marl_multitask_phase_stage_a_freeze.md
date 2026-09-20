# Multi-task participation-phase Stage-A execution freeze

## Authorization

The outcome-free G0 validation passed for both registered tasks.
This freeze authorizes exactly the 32 development runs in
`marl_multitask_phase_headroom.json` and no other outcome-bearing execution.
It does not authorize `q=2,4`, a controller, confirmation seeds, or manuscript
claims.

## Frozen artifact hashes

| Artifact | SHA-256 |
|---|---|
| `experiments/marl_multitask_phase_headroom.json` | `cfee35ecb7c7b158e0ccceaedce6d2211cca99618488ebbe94da887219571855` |
| `experiments/run_mappo_return_bridge.py` | `16289055f3390cf7c51ff954f54a6c47797bb00f8555c9ef7f780d53680787ed` |
| `experiments/analyze_marl_multitask_phase_headroom.py` | `f4469aaa2d5e15540ef6581395a7da95cc3e2f340193da928121ab757be2c856` |
| `slurm/marl_multitask_phase_stage_a_a30.sbatch` | `08a35b7f953762853ca1bea70780a1858b3bf6ee3cabb2d50a12a060b75bf1f4` |

## Runtime boundary

Speaker--Listener uses the pinned MPE runtime and MaMuJoCo uses the pinned
Two-Clocks runtime with its existing MuJoCo 2.1 library.
Every cell writes to a unique directory under
`/scratch/jzhuangag/MARL-SDDE-TSP-MULTITASK-PHASE-A-001/artifacts/stage-a`.
No output is written to `/project` or `/home`.

Any missing, duplicate, nonfinite, over-budget, dirty-upstream, or failed cell
stops analysis and is retained as such.
An operational recovery may repair infrastructure only after a separate
amendment records the failure and proves that tasks, rays, actions, seeds,
hyperparameters, and gates are unchanged.
