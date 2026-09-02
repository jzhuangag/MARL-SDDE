# Two Clocks standard pilot: operational amendment 1

## Reason and timing

The preregistered jobs `1810507`, `1810508`, `1810509`, and `1810510` all
stopped during the first CUDA gradient computation.  PyTorch deterministic
mode rejected cuBLAS because `CUBLAS_WORKSPACE_CONFIG` was not set.  Slurm
reported all four jobs as `FAILED (1:0)` after 57--77 seconds.

No job wrote `summary.json`; therefore no scientific row, return, AUC,
contrast, or gate result was observed.  Runtime records, GPU records, stdout,
and stderr remain preserved in the independent scratch root.  This amendment
was made only after verifying the absence of scientific output.

## Sole change

The Slurm environment now exports the CUDA-documented deterministic workspace
setting:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

No runner, analyzer, configuration, task, seed, service profile, method,
metric, threshold, gate, or stopping rule changes.  The original
preregistration and its five artifact hashes remain immutable provenance; the
amended Slurm script SHA-256 is
`6ed6d9a972c93ea8e1e01017a3d83ca504fc445b144c4fbd82fbd6da1e1228b0`.
The next submission must run the amended repository commit exactly, and any
further failure is retained in the same way.

## Failed attempt ledger

| Job | Task | Run | State | Exit | Scientific summary |
|---:|---|---|---|---|---|
| 1810507 | MAMuJoCo Ant 4x2 | primary | FAILED | 1:0 | absent |
| 1810508 | MAMuJoCo Ant 4x2 | reproduction | FAILED | 1:0 | absent |
| 1810509 | SMACv2 Terran 5v5 | primary | FAILED | 1:0 | absent |
| 1810510 | SMACv2 Terran 5v5 | reproduction | FAILED | 1:0 | absent |
