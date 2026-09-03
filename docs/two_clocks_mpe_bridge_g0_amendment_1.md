# Two Clocks public-MPE bridge G0 Amendment 1

Date: 2026-09-03.

## Reason

The frozen G0 defines one complete packet as two independent trajectories and
correctly charges `2 * episode_length` completed environment steps.  Its
frozen-barrier tail calculation, however, used only `episode_length` when
converting an unfinished service fraction to charged work.  The asynchronous
tail calculation already used the full two-trajectory packet charge.

This is an outcome-free accounting defect.  No reward or learning comparison
was generated, but exact charging is mandatory, so the validation attached to
`b406e81` is superseded rather than silently repaired.

## Frozen correction

For remainder `r` of a packet with declared service duration `s`, the corrected
barrier charge is

`floor(2 * episode_length * r / s)`,

capped at `2 * episode_length`.  No task, method, seed, learning rate,
interaction constant, event-delay bound, or other G0 condition changes.

The corrected implementation must pass the same 16-case structural gate and
produce byte-identical primary/reproduction output before pilot
preregistration is restored.

Frozen corrected provenance:

- contract SHA-256:
  `be282e6f7e8d1c5911ffada71c6b4c0a6407ae56422048442b270ba397d2a634`;
- G0 runner SHA-256 (unchanged):
  `d1764fa7b3e81e7d3a112e07f3839900b6f07361e5a9bf84c89036b37c136ae8`;
- G0 configuration SHA-256 (unchanged):
  `75aaeeb197abbb1805e92d24efc16100002479d5d313d56f94a0b20081e6e562`.
