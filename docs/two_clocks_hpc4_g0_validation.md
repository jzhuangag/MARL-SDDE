# Two Clocks HPC4 G0 validation

Date: 2026-09-02.

Decision: **pass the outcome-free runtime gate; scientific pilot remains
unrun**. Passing this gate establishes only that the pinned software stack can
execute the registered task shapes and actor interfaces on HPC4. It is not a
learning result and supplies no paper evidence.

## Final passing run

- Slurm job: `1810485`, `gpu-a30/gpu07`, `COMPLETED 0:0`, 14 seconds.
- Code commit: `af83e94f875f971cc5374a1e5f3f19658deec3f5`.
- HARL commit: `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- SMACv2 commit: `577ab5a2cff2391f8df582da5731ea9cd6adf3c6`.
- Runtime: Python 3.9.25, NumPy 1.26.4, PyTorch 2.8.0+cu128,
  CUDA 12.8, NVIDIA A30.
- Result directory:
  `/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902/results/g0-1810485`.
- `summary.json` SHA-256:
  `7f9166628efa33a843b1d17d47e6f5f017af06865b0b57bab45ec6611faff03f`.
- `SHA256SUMS` SHA-256:
  `ebb436cf2c52666824c5b4878fc19ee4d9d08af565ef18801ced98241782fe37`.
- Every entry in `SHA256SUMS` revalidated successfully.
- The committed validator accepted the summary, and no SC2 descendant process
  remained after the job.

The task-shape results were:

| Task | Agents | Observation shape | State shape | Action | Actor transitions |
|---|---:|---:|---:|---|---:|
| MAMuJoCo Ant `4x2` | 4 | 115 per actor | 111 | Box(2) | 4 |
| SMACv2 Terran `5v5` | 5 | 82 per actor | 120 | Discrete(11) | 5 |

Both tasks executed exactly one environment transition. Every actor was a
distinct HARL `StochasticPolicy` object and performed a CUDA forward pass. All
eight frozen G0 invariants passed: CUDA availability, exact task set, distinct
actors, one transition per task, full actor-transition charging, clean pinned
sources, finite timing, and no descendant-process leak. The JSON output
contract rejects reward, return, success-rate and win-rate fields.

## Preserved failed qualification attempts

No failed attempt was deleted or relabeled.

| Job | State | Cause | Corrective commit |
|---|---|---|---|
| `1810470` | `FAILED 1:0`, 54 s | MuJoCo required `/usr/lib/nvidia` in `LD_LIBRARY_PATH`. | `282685d` |
| `1810475` | `FAILED 1:0`, 47 s | HPC4 lacked the GLEW development header. | `d1d2ea8` plus pinned GLEW 2.2.0 source |
| `1810482` | `FAILED 1:0`, 47 s | GLEW attempted to include unavailable GLU; EGL shim does not use GLU. | `60ca945`, `GLEW_NO_GLU` |
| `1810484` | `COMPLETED 0:0`, 70 s | Runtime and task interfaces passed, but `runtime.txt` was changed after its checksum was written. | `af83e94`, manifest written last |
| `1810485` | `COMPLETED 0:0`, 14 s | All runtime and provenance gates passed. | final G0 |

The first four runs are setup/provenance failures, not scientific outcomes.
Their logs and result directories remain under the new project root.

## Source assets and storage

The new independent root occupies 13 GB on `/scratch`; `/scratch` remained at
57% use with 219 GB available after setup. MuJoCo 2.1, SC2 4.10, SMAC maps and
the GLEW 2.2.0 release archive are retained under `external/downloads/` with
SHA-256 provenance. No experiment output was written to `/project` or `/home`,
and the historical `/scratch/jzhuangag/MARL-SDDE` tree was not modified.

## Authorization boundary

This result permits a later, independently committed pilot preregistration.
It does not authorize formal seeds, outcome-selected task changes, or a claim
that the unconstrained neural actor satisfies the finite-policy theorem. The
standard neural layer remains an explicitly empirical extension.
