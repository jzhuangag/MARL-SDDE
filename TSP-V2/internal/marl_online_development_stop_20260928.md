# TSP-V2 online MARL development stop decision

## Decision

`TSP-V2-MARL-ONLINE-DEV-001` is stopped before scientific execution.  No
offset beyond zero is authorized, confirmation is not authorized, and no
result from this root may enter the manuscript.

This is an operational payload failure, not a negative learning result.  All
eight offset-zero array elements terminated before their initial reporting
evaluation and before any registered training update.  Consequently there is
no return curve, controller decision, budget comparison, or scientific gate
outcome to interpret.

## Frozen provenance

- Implementation commit: `654b7b6`.
- Preregistration commit and remote source HEAD:
  `357e266da709567fabdf3b07480bcb55e39abf6f`.
- Slurm array: `1916273`, tasks `0--7`.
- Root: `/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-DEV-001`.
- Every task state: `FAILED`, exit code `1:0`.
- Task elapsed times: 5--18 seconds.
- Nodes: `gpu14` for tasks 0--6 and `gpu10` for task 7.
- Root size at audit: 322 MB.
- No `/project` write was made.

The preregistered configuration and executable hashes matched before
submission.  The remote Git worktree remained clean at the frozen HEAD.

## Failure

Every task raised the same exception in the first call to
`deterministic_horizon_return`:

```text
AttributeError: 'OnPolicyMARunner' object has no attribute 'env_name'
```

The runner attempted to construct the disjoint evaluation environment using
`runner.env_name`.  The pinned HARL `OnPolicyMARunner` stores the environment
identifier through its main argument structure, not through that attribute.
This interface was not exercised by the local fake-runner tests.  Controller
tasks created their four candidate runners before failing; fixed-q tasks
created one runner.  None reached `train_one_update`.

## Evidence hashes

The task-paired logs have the following SHA-256 values:

| Tasks | stdout | stderr |
|---|---|---|
| 0--1 | `810d260f4ed9b35a6aa45bfd4889bbf080574585b06b66bf0c0eca677c254319` | `2f2babd95966bf445fa174c8501ac0490b4c3232af16b18fec9fdb56aaae8007` |
| 2--3 | `784767a2ee04bb8677c34dd13df8a6fc87f6d52e83912357eb3468366be2af09` | `93f884e43cefa010c63aa4052eee5730224d72001c5517d24c25f287ad81590a` |
| 4--5 | `784767a2ee04bb8677c34dd13df8a6fc87f6d52e83912357eb3468366be2af09` | `229e83c20c4a613d8146089994f3e4be1e492b3f2d78da5cc01ad7a2f69b2205` |
| 6--7 | `784767a2ee04bb8677c34dd13df8a6fc87f6d52e83912357eb3468366be2af09` | `82c70176d4e11b0c6f25923e9a786830a110c7c3b5f81ccd43912667aef1d37c` |

No `metadata.json`, `progress.jsonl`, or per-artifact `SHA256SUMS` was
produced.  Partially created run directories and logs are preserved without
cleanup or overwrite.

## Stop-rule application

The frozen preregistration states that any payload failure stops development
without retry and that seeds, tasks, methods, constants, budgets, and gates
cannot be changed after submission.  Therefore:

1. offsets `8,16,24,32` are not submitted;
2. array `1916273` is not retried;
3. no analyzer is run on nonexistent scientific output;
4. no confirmation seeds are registered;
5. the frozen `TSP/` manuscript remains unchanged.

A corrected implementation would require a new outcome-free runtime
qualification and a separately preregistered experiment identifier.  That is
future work and is not silently substituted for this stopped experiment.
