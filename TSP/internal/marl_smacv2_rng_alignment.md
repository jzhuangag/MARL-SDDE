# SMACv2 procedural-RNG alignment audit

## Finding

The fixed shared-`q=8` integration run completed, but the first controller
smoke (`1851244`) showed that equal environment seeds and public action
uniforms were not enough to synchronize SMACv2 rollouts.  The four residual
fingerprint blocks had a maximum within-block range of `1.0064073982`, and the
uncorrected probe selected `q=8`.

Source inspection found the cause.  SMACv2 passes its public `seed` to the
StarCraft II game request, but each procedural capability distribution creates
its own `numpy.random.default_rng()` without that seed.  Team generation also
uses Python's global `shuffle`.  Consequently, procedural teams and start
positions were entropy-seeded even when the environment seed matched.  The
same issue made nominally deterministic evaluation irreproducible across
separate processes.

## Outcome-free correction

Before any development run, the local bridge now assigns each procedural
distribution an independent child of `SeedSequence(environment_seed)` and
assigns a separate child to a per-environment Python-random state.  The latter
is swapped in only during environment reset and the caller's global state is
restored afterward, so deterministic team shuffling cannot perturb MAPPO's
learner RNG in a single-process `q=1` run.  Nested team/position generators
receive distinct child streams, so the single-rollout joint law does not gain
artificial dependence among capability components.  SMACv2 continues to draw
from the same categorical and continuous distributions, while its existing
StarCraft seed remains unchanged.

The bridge applies the same deterministic seeding rule to evaluation workers.
No HARL or SMACv2 upstream file is edited.  Unit tests check reproducibility
for equal seeds, separation for unequal seeds, and independence of child
streams.  A repeated shared-`q=8` controller smoke must show identical worker
fingerprints and select `q=1` before the 24-cell development array is
authorized.

Runs `1851226`, `1851231`, `1851244`, and `1851257` remain integration
diagnostics only; none enters the development analysis or the paper.  In the
post-fix run `1851257`, three of four blocks were bitwise equal, the fourth had
a maximum range of `0.0196739`, the estimated correlation was `0.999504`, and
the controller selected `q=1`.  The exact-equality smoke assertion therefore
remained failed even though the intended decision was recovered.

The first checksum command also included `SHA256SUMS` in its own manifest.
Every scientific payload entry verified, but the self-entry necessarily did
not.  The scripts now exclude the manifest itself; historical manifests are
not rewritten.
