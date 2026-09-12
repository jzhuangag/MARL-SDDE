# Validation of Two Clocks public-MPE bridge G0 Amendment 1

Date: 2026-09-03.

## Decision

**Corrected G0 passes.**  CPU pilot preregistration is restored.  No
learning-performance outcome was generated and no GPU run is authorized.

The corrected frozen source commit was `bed08d1`.  All 12 targeted tests and
all 16 task/profile/method cases passed.  Primary and isolated reproduction
outputs were byte-identical with SHA-256
`26b850295f3c894e44f1d02f4fd104e61a5e0d19aa9009029349e5045f7862ab`.

The corrected heterogeneous `simple_spread_v2` barrier now charges 9 partial
environment steps at the four-unit horizon, matching the full
two-trajectory packet definition.  Its 7 complete packets and 3 batched owner
updates are unchanged.  The asynchronous methods also charge 9 partial steps.
All other registered shapes, seeds, methods, constants, and invariants are
unchanged from the original G0.

This validation authorizes only a separately frozen CPU pilot with fresh
seeds.  The original `b406e81` output remains preserved as superseded
provenance.
