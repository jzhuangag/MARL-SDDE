# Amendment 1: lossless full-cadence credit bank

Date: 2026-10-15.

The preregistration commit is `e9d7716`. Before any result analysis, a static
audit of the running ledger found that the full-cadence HAPPO comparator used
a credit bank capped at the team size `N`. For `K` credits per block, this
drops credits whenever `K` does not divide `N`; on three-agent spread with
`K=2`, it would spend 36 rather than the registered 48 packets. The frozen
analyzer correctly required a fixed schedule to exhaust its packet budget, so
the original implementation could not pass integrity and did not provide an
equal-resource full-cadence comparator.

The initial scientific execution was interrupted after two spread payloads
completed. Their return files were not opened or analyzed. The directory is
preserved and is ineligible for any scientific claim.

The correction is outcome-free:

- full-cadence HAPPO uses the minimal lossless bank cap `N+K-1` and launches
  all `N` actors whenever the bank contains at least `N` credits;
- all other controller behavior, tasks, horizons, costs, methods, actions,
  gates, and analyzer rules remain fixed;
- all development, holdout, replay, and qualification seeds are replaced;
  and
- a new source lock and disjoint operational qualification are required before
  restarting scientific execution.

This amendment changes resource feasibility, not a performance threshold.

The amended 12-block operational qualification used the disjoint seed
`72299992` and completed all 14 task--method payloads. The integrity audit
passed complete mapping, identical task initialization, finite outputs,
unique trajectory identities, source hashes, controller separation, and all
transition, optimizer, message, counterfactual, and diagnostic charges. In
particular, full-cadence HAPPO spent exactly its registered packet budget with
zero overflow.
