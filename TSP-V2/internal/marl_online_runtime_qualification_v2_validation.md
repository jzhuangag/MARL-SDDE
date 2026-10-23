# TSP-V2 second HARL runtime qualification validation

## Decision

`TSP-V2-MARL-ONLINE-QUAL-002` passes every frozen runtime gate.
This authorizes a separately preregistered scientific development experiment;
it is not performance evidence and none of its returns may enter the paper.

## Provenance

- Implementation commit:
  `ef08d1f56e62f3094dded54299a5acfa361344db`.
- Qualification preregistration and remote source commit:
  `52a227b757cae07e9075ed1786fe52ebf778bff7`.
- Slurm array: `1918092`, tasks `0--7`.
- Scratch root:
  `/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-QUAL-002`.
- Pinned HARL commit:
  `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- All eight tasks: `COMPLETED`, exit code `0:0`.
- No `/project` write or cleanup was performed.

## Gate results

| Gate | Result |
|---|---:|
| Eight unique task/coupling/method payloads | pass |
| Every exit code zero | pass |
| Initial and post-update evaluation present | pass |
| At least one training update per payload | pass |
| Controller before/after validation observation | pass |
| All numeric outputs finite | pass |
| Exact message and environment accounting | pass |
| Pinned HARL commit unchanged | pass |
| Every per-artifact `SHA256SUMS` | pass |
| Scratch-only output | pass |

For controller tasks, the selected qualification action was `q=1`.
MPE therefore charged 375 messages and 75 environment ticks, equal to
`(100+25)+2(100+25)` and `25+2(25)`.
MaMuJoCo charged 2700 messages and 300 environment ticks, equal to
`(800+200)+2(800+50)` and `200+2(50)`.

The fixed-q qualification branch completed three MPE updates and one MaMuJoCo
update, as implied by the frozen dual budgets.
It charged 375 messages/75 ticks for MPE and 1000 messages/200 ticks for
MaMuJoCo.

## Artifact integrity

The eight `SHA256SUMS` manifest hashes are:

- `b8aad930dd313c328e451f8ba4fcddb4762204e8a7406512e1a1ab848750ebd1`;
- `5db1070957591b3a670debef7743b8c105f084230368fa9a3d3e5bcabd26ec3f`;
- `583a7536b480ab471d00a8aad3462acf176370f98d52a7f0ba6148ef59df9710`;
- `6f4137789eed325541798b310b1ece95c559226de906e4d8c8ce61e3dade08bb`;
- `6a91f3c5d7b73286185b7761c6f23b98d3c5660f82accc61226d13bae58b8eb8`;
- `e240ec2acd4a1b638f1ddeb35000ef2900b80610657316389f72a6e6a4761245`;
- `5492f665d127cde408a478015e25fd0c5f5b9fa49d271bc8dbda7e51004b99d8`;
- `53a3857708b567c359bbb7c98a4abe97f9f6824553ef9c4362830918940ecf69`.

MPE stderr contains only the pinned PettingZoo deprecation warnings.
All four MaMuJoCo stderr files are empty.
No traceback, exception, or runtime error is present.

## Authorization boundary

The pass validates execution and accounting, not scientific performance.
A development run must use a new identifier, a new scratch root, fresh training,
validation, and evaluation seeds, and a preregistration commit that leaves the
scientific task, controller, budgets, methods, and mandatory gates unchanged
from the stopped development design.
