# T-063A formal validation

## Decision

T-063A is a **formal failure under the frozen analysis plan**.  The primary
run is strongly positive on efficacy, and the independent chunked reproduction
is byte-identical, but three mandatory analyzer gates do not pass:

1. breadth lower confidence bound `0.583333 < 0.60`;
2. the frozen maximum-per-seed rho-zero collision gate (`2/96 > 0.02`);
3. exact numeric summary replay, because CSV round-tripping changes several
   floating-point summary values at approximately `1e-14`.

No threshold, seed, cell, comparator, or analysis rule was changed to rescue
the result.  The exact endpoint/cell/summary artifacts match byte-for-byte,
so the replay failure is a serialization-level audit defect rather than a
scientific disagreement; it nevertheless remains a failed preregistered
gate.

## Formal efficacy result

| metric | point | one-sided bound | threshold | result |
|---|---:|---:|---:|---|
| aggregate controller/strong | 0.829858 | 0.850257 upper | 0.95 | pass |
| Asterix | 0.811510 | 0.851898 upper | 0.98 | pass |
| Breakout | 0.826209 | 0.873443 upper | 0.98 | pass |
| Seaquest | 0.852368 | 0.896957 upper | 0.98 | pass |
| delay 0 | 0.830265 | 0.854907 upper | 0.97 | pass |
| delay 8 | 0.829451 | 0.853900 upper | 0.97 | pass |
| true-rho oracle proximity | 1.010319 | 1.016015 upper | 1.15 | pass |
| strict-cell breadth | 0.678571 | 0.583333 lower | 0.60 | **fail** |

The aggregate geometric error reduction is `17.0142%`.  The point strict
fraction is `57/84 = 0.678571`; its complete-seed cluster bootstrap lower
bound is the value used by the frozen gate.

## Provenance

- configuration SHA-256: `8e90b08f18a14b777356ab3c575c738d9c8b62c5a9a0d7b2ff06ae78605d457d`;
- primary endpoints SHA-256: `271969a7ad0e7f80ccef484c187109b3733000307f1408a3d171b8d4aa6ffab9`;
- reproduction endpoints SHA-256: identical;
- primary cells SHA-256: `879e59d94e1189ab0e2a83b7fabbf04bb43ea87b7e98fb83a84a11c5f0a2cc80`;
- reproduction cells SHA-256: identical;
- primary summary SHA-256: `32fcba18d68b292d72a9c43e1f93e4be2916f0059720edd442a4aa2d514c0c8a`;
- reproduction summary SHA-256: identical;
- full nonlinear tests: `154 passed, 7 skipped`;
- GPU/HPC4: not used.

The machine-readable analyzer output is
`docs/validation_t063a_reward_free_controller_formal.json`.

## Consequence

T-063A supports a strong but qualified nonlinear efficacy result; it does not
provide an all-gates-passed formal claim.  T-063B must not silently inherit
the exact-replay comparison bug.  Before any T-063B execution, its analyzer
and replay rule require a separate outcome-free amendment and audit.  The
T-063A failure remains permanently preserved.
