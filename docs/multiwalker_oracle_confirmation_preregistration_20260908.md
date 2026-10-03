## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: independent confirmation preregistration
- Origin Date: 2026-09-08
- Verification Status: FROZEN BEFORE CONFIRMATION OUTCOME GENERATION
- Version Label: multiwalker_oracle_confirmation_v1

# Multiwalker exact dynamic-value independent confirmation

## Question and evidence boundary

The development Amendment found positive equal-resource headroom for an exact
prefix-budget cache-refresh oracle. This confirmation asks only whether that
headroom repeats on untouched environment seeds. It does not fit or evaluate a
critic, train a policy, or claim that an observable scheduler reaches the
oracle.

Seeds `96100--96107` were checked against tracked source and documentation and
were not previously used. Development seeds `95700--95707`, Pursuit data, and
all earlier controller outcomes are excluded. The frozen machine-readable
configuration is `multiwalker_oracle_confirmation_config_20260908.json`.
Its SHA-256 is
`AD5284984C99CB08EAD4014D64E5300CE41E867340164AD386B7F5D899D38BCB`.

## Frozen computation

For every seed, motion scale `0.01/0.04`, and prefix edge-budget rate
`0.25/0.50`, the runner first regenerates all eleven frozen v1 policies from
the same public all-current Multiwalker prefix. The strong online envelope
excludes only the invalid receding diagnostic `oracle_h8` and `no_refresh`; it
includes the privileged one-step oracle, action/parameter mismatch, age,
round-robin, random, fixed edges, and complete-neighbor refresh.

The exact diagnostic then enumerates all null-or-one-edge cache sequences per
owner and solves the same multiple-choice binary program under every one of
the 40 prefix constraints. It must have certified optimal status, MIP gap at
most `1e-9`, five selected owner sequences, exact no-refresh replay within
`1e-10`, and no positive prefix-budget excess.

The 352 baseline rows and 32 exact rows may be generated as four disjoint
two-seed CPU chunks. Only a deterministic merge over the complete frozen seed
registry is analyzed. A clean reproduction repeats all chunks from source;
baseline rows, exact rows, and the scientific summary must be byte-identical.

## Mandatory gates

| Gate | Frozen rule |
|---|---|
| C1 | merged seeds equal exactly `96100--96107` |
| C2 | exactly 352 baseline and 32 exact rows |
| C3 | all baseline values finite, exact replay, and prefix-budget valid |
| C4 | all exact optimizers and replay/prefix checks valid |
| C5 | aggregate active exact-oracle gain over no refresh is positive |
| C6 | aggregate active oracle recovery above strong envelope is at least 10% |
| C7 | exact oracle strictly beats strong envelope in at least 75% of 16 active cells |
| C8 | median active normalized absolute headroom is at least 0.1% |
| C9 | clean reproduction is byte-identical for baseline rows, exact rows, and summary |

The scientific C5--C8 thresholds are identical to the development Amendment.
No threshold, seed, regime, policy, or action class may change after outcomes
are observed. Any C1--C9 failure stops critic design on Multiwalker. A complete
pass authorizes only an outcome-free statistical critic/certificate design;
it does not authorize a controller success claim, formal seeds, GPU, or HPC4.

## Planned commands

Each primary and reproduction run uses four commands of the form

```text
D:/anaconda/envs/ust2/python.exe -m experiments.policy_dependency_sync.multiwalker_oracle_confirmation --output-dir <chunk-dir> --seeds <two frozen seeds>
```

followed by

```text
D:/anaconda/envs/ust2/python.exe -m experiments.policy_dependency_sync.multiwalker_oracle_confirmation --output-dir <merged-dir> --merge-inputs <four chunk directories>
```

No GPU or remote resource is required.
