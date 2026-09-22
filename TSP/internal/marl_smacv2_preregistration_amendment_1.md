# TSP-MARL-SMACV2-DEV-001 Amendment 1: valid StarCraft seeds

## Reason and timing

The first two integration-only smoke arrays (`1851226` and `1851231`) failed
before producing a completed evaluation or any development outcome.  The
second array reached HARL evaluation setup and exposed an implementation
constraint that the static plan had missed: HARL constructs each evaluation
seed as

```text
training_seed * 50000 + evaluation_rank * 10000,
```

whereas StarCraft II's protobuf field is an unsigned 32-bit integer.  The
original development seeds near 110,000 therefore overflow before evaluation.
This is a feasibility defect, not a scientific result, and it cannot be fixed
by changing StarCraft II or by clipping seeds because either would make the
run law implicit.

## Frozen amendment

Before any scientific SMACv2 trajectory is run, replace only the unused seed
registries:

- development training: `80101, 80102, 80103, 80104`;
- separated probe: `80201, 80202, 80203, 80204`.

These integers were checked against the repository's non-result registries and
were unused when selected.  With eight evaluation threads, the largest
constructed evaluation seed is

```text
80104 * 50000 + 7 * 10000 = 4005270000 < 2^32.
```

The new smoke-only seeds are `80301` and `80302`; they remain disjoint from the
development registries and are never included in analysis.

No task, coupling, method, model, optimizer, controller, budget, metric,
threshold, gate, or stopping rule changes.  The original preregistration and
both failed integration logs remain intact.  A static unit test now enforces
the SMACv2 uint32 condition so that the error cannot recur silently.
