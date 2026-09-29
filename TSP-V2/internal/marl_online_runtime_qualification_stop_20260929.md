# TSP-V2 HARL runtime qualification stop decision

## Decision

`TSP-V2-MARL-ONLINE-QUAL-001` is stopped and must not be retried.
The qualification produced no scientific evidence and does not authorize a
development experiment.

The four PettingZoo MPE payloads completed and passed their checksum, finite
value, pinned-HARL, evaluation, training-update, and accounting checks.
All four MaMuJoCo payloads failed during their initial evaluation, before any
registered training update.
The aggregate mandatory requirement of eight successful payloads therefore
failed.

## Frozen provenance

- Implementation commit: `291b402f854568553d747946d584cff315759c34`.
- Qualification preregistration commit:
  `428a8f506960ab9a38a60a8912712368c87513af`.
- Slurm array: `1917818`, tasks `0--7`.
- Scratch root:
  `/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-QUAL-001`.
- Pinned HARL commit:
  `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- Tasks `0--3`: `COMPLETED`, exit code `0:0`.
- Tasks `4--7`: `FAILED`, exit code `1:0`.
- Root size at audit: 324 MB.
- No `/project` write or cleanup was performed.

## Failure

Pinned HARL uses two legal unavailable-action representations.
PettingZoo MPE supplies a two-dimensional action mask, whereas MaMuJoCo
supplies a one-dimensional container whose first entry is `None`.
The corrected bridge tested only whether the container itself was `None` and
then indexed it as two-dimensional.
Every MaMuJoCo task consequently raised

```text
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

in `deterministic_horizon_return` before the first training update.
The pinned HARL `eval()` implementation instead passes an action mask only when
`available_actions[0] is not None`; the next implementation must reproduce that
contract and test both representations explicitly.

## Preserved evidence

The MPE artifacts retain valid `metadata.json`, `progress.jsonl`, and
`SHA256SUMS` files.
Their values are qualification diagnostics only and cannot be used for tuning
or manuscript claims.
All MaMuJoCo partial directories and all stdout/stderr logs are preserved.

The repeated MaMuJoCo stderr hashes are:

- controller tasks `4` and `6`:
  `a99e209712cb47a03f9fc53c83d43d1c2c076e4ea15fe412b78c03bd3eb23f49`;
- fixed-q tasks `5` and `7`:
  `03cd5674dd077c41176e544ebc241bd205d154523cb9d6e545b76e6c09fa6f91`.

## Next admissible action

A new implementation may align the unavailable-action check with pinned HARL,
add a unit test for the one-dimensional all-`None` representation, and undergo
a separately frozen `QUAL-002` runtime qualification.
The stopped `QUAL-001` root, seeds, artifacts, and job are not reused.
