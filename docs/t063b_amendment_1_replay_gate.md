# T-063B Amendment 1: numeric replay gate

## Status

This amendment is frozen before any T-063B scientific run.  It does not alter
the controller, tasks, seeds, budgets, efficacy thresholds, collision gate,
bootstrap, or comparator.  It corrects only the replay audit's treatment of
decimal CSV serialization.

## Correction

T-063A showed that recomputing the point summary from its CSV endpoints changes
several floating-point values by about `1e-14`, while primary and reproduction
artifact bytes are identical.  T-063B therefore requires both:

1. byte-exact equality of `endpoints.csv`, `cells.csv`, and `summary.json`; and
2. recursive replay equality with absolute and relative tolerances `1e-12`.

Differences larger than this tolerance fail the replay gate.  The exact
artifact requirement remains mandatory.  No T-063A gate or result is changed.

## Authorization

Because this is a preregistration amendment, T-063B authorization is reset to
false until the updated configuration hash, analyzer, tests, and this note are
committed and statically audited in a separate authorization commit.
