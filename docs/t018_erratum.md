# T-018 bookkeeping/statistical erratum

## Erratum

The original T-018 static scan used `2000001` as a sentinel when the search
for `B_value` reached 2,000,000 without finding a qualifying learning-aware
threshold. That sentinel was then summarized like a finite threshold. This was
a bookkeeping error, not a scientific outcome.

The corrected representation uses explicit states:

- `finite`
- `search_censored`
- `never_qualifies_if_analytically_proved` reserved for a future proof

For `search_censored` scenarios, `B_value` is `null`, the value probe has no
`q/b/n`, and no `B_value`-derived budget cells are generated.

## Corrected counts

- All scenarios: 3456
- Finite `B_value`: 3448
- Search-censored: 8
- Censored fraction: `0.0023148148148148147`
- Finite active-zone coverage:
  `1`
- Descriptive effect coverage among finite `Z` cells:
  `0.70102233178654294`
- Message-binding finite scenarios: 1766
- Environment-binding finite scenarios: 1402

## N3 correction

Commit 1 froze the cell-level practical-effect threshold of 3%. It did not
freeze an effect-coverage proportion gate. Therefore the corrected coverage is
a descriptive outcome-free scan result, not a retrospective preregistered pass
gate.

## Safety recheck

The future theorem-facing metric remains

```text
S_mean = [E L_policy - E L_all]_+ / E L_all.
```

`S_path` remains a descriptive tail metric only. Amendment 1's `-0.015104`
margin came from its static prospective path model and does not license a
claim that `epsilon_safe` controls `S_path`.
