# TSP-V3-CERT-SHIELD-002 confirmation validation

## Decision

Pass.  The robustness corollary and compact sensitivity/ablation result are
authorized for manuscript migration.

Sixty-four disjoint confirmation seeds produced 157,440 finite checkpoint rows
with exact resource accounting.  Every frozen gate passed:

- all 24 conservative level-by-cell one-sided 99 percent upper means were
  covered by their robust certificates;
- moderate and strong shields both retained the nominal action in all 12 cells
  and had terminal parameter-risk ratio 1.000000;
- finite-horizon / one-step-myopic terminal parameter-risk ratio was 0.878042,
  with paired-bootstrap 95 percent interval [0.835939, 0.923051];
- finite-horizon / correlation-only terminal parameter-risk ratio was
  0.357783, with interval [0.307446, 0.412218].

The last two comparisons correspond to terminal-risk reductions of 12.20 and
64.22 percent.  The result supports the prespecified finite-horizon terminal
risk claim; it is not presented as AUC dominance over the one-step rule.

## Reproduction

A clean confirmation replay produced byte-identical scientific tables:

| File | SHA-256 |
|---|---|
| `metrics.csv` | `90AA41E163588617571FB44041D9B2B24A8CF88F41FB4154D140C1617D53B2E2` |
| `action_table.csv` | `CC8DBE61059D7A1A432729BD7D675E52E08B2CC762C499A6DF842981717D80B7` |
| `seed_cell_metrics.csv` | `411C15E7DCE717587BCFA387653235880776A73DCFF409207A662A13D90B67C0` |
| `paired_comparisons.csv` | `C1FBF3AEC76585D157AE584A22C862B354FC80349BC268458E63876D86BB814F` |
| `coverage.csv` | `DD62196BAF0F6C2ADE739C56CECDD836C78F0A93D17612C0981AF14EC4825994` |

Only elapsed wall time differs between `summary.json` files.
