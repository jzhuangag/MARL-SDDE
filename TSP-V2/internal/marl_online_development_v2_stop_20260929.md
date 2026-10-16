# TSP-V2 online MARL development 002 stop decision

## Decision

`TSP-V2-MARL-ONLINE-DEV-002` is stopped after the first eight-cell wave.
No further offset, analyzer decision, confirmation experiment, or manuscript
migration is authorized under this identifier.

All eight offset-zero payloads completed with exit code zero and produced
finite artifacts, but the mandatory exact-accounting audit cannot be completed
from the registered output.
This is an experiment-payload observability failure, not a scientific
performance result, so none of the observed returns may be interpreted or
entered into the manuscript.

## Frozen provenance

- Implementation commit:
  `ef08d1f56e62f3094dded54299a5acfa361344db`.
- Development preregistration and remote source commit:
  `8f5a63c746f1d10f585d03f53364ce139369a9a8`.
- Slurm array: `1918107`, tasks `0--7`, offset `0`.
- Scratch root:
  `/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-DEV-002`.
- Every array task: `COMPLETED`, exit code `0:0`.
- Eight expected MPE independent-coupling artifacts are present.
- No `/project` write or cleanup was performed.

## Mandatory-gate failure

The runner charges the selected action online and reports cumulative messages,
cumulative environment ticks, and remaining budgets.
However, its terminal `selected_counts` field is computed from
`len(controller.histories[q])`.
Each history is truncated to the sliding-window length eight, so this field is
not a cumulative action count.

The two completed controller cells executed 100 registered decisions, while
their reported count totals were only 32 and 29:

- seed `153001`: `{1:8, 2:8, 4:8, 8:8}`, sum `32`;
- seed `153002`: `{1:8, 2:8, 4:8, 8:5}`, sum `29`.

Consequently the reported cumulative charges cannot be independently
reconstructed from a complete action ledger.
The online arithmetic may be internally consistent, but the preregistered
requirement was an auditable exact dual-budget accounting result.
That evidence is absent, so the gate cannot be marked passed.

This also makes `controller_q_counts` in the frozen analyzer window-capped
rather than exact, although its nondegeneracy predicate still detects whether
an action has ever received an observation.
The mismatch must be fixed in a future implementation by recording every
decision or maintaining non-windowed cumulative counts separately from the
statistical sliding window.

## Stop-rule application

The preregistration requires any mandatory failure to stop confirmation and
forbids changing the executable or analysis after observing development
outcomes.
Therefore:

1. offsets `8`, `16`, `24`, and `32` are not submitted;
2. array `1918107` is not retried;
3. the frozen analyzer is not used to issue a development pass;
4. confirmation seeds are not registered;
5. the TSP manuscript remains unchanged;
6. all offset-zero artifacts and logs are preserved in scratch.

A future attempt requires a new implementation, outcome-free tests of the
complete action ledger and charge reconstruction, a new experiment identifier,
and fresh seeds.
