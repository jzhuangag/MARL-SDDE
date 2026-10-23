# TSP-V3 robust-certificate manuscript audit — 2026-10-03

## Scope and publication decision

The manuscript revision integrates only the independently confirmed positive result from `TSP-V3-CERT-SHIELD-002`.
The earlier direct-pessimistic-ranking design remains in the internal experiment ledger and is not used as publication evidence.
No citation or bibliography entry was added, removed, or materially edited in this revision.

## Theoretical addition

Corollary 2 separates two roles of a conservative certificate:

- upper moment and mixing quantities define a robust feasible set and an upper risk recursion;
- a predictable central finite-horizon score ranks actions only within that certified set.

On the joint coverage event, substituting `K_q^+ >= K_q`, `Omega_q^+ >= Omega_q`, and `r_q^+ >= r_q` preserves the true recursion.
The proof gives explicit perturbations of the contraction and forcing coefficients and the finite-horizon sensitivity bound

`|U_a^+ - U_a| <= (n R_0 + d_bar/(1-c_bar)^2) Delta c_a + Delta d_a/(1-c_bar)`.

Minimizing an upper score therefore incurs at most twice the maximum score perturbation relative to the exact-certificate oracle.
This statement quantifies the price of conservative upper bounds without using them as plug-in truth for nominal ranking.

## Confirmed controlled study

- Experiment: `TSP-V3-CERT-SHIELD-002`.
- Confirmation seeds: 64 disjoint seeds (`36000001` through `36000064`).
- Scientific checkpoint rows: 157,440.
- Conservative stresses: 10% moment inflation plus 0.05 mixing-log-rate slack, and 25% plus 0.10.
- Coverage: 24/24 one-sided 99% upper means were below the robust certificate.
- Selection: both conservative shields retained the nominal action in all 12 cells.
- Empirical terminal-risk ratio under each shield: 1.000000.
- Finite-horizon versus one-step-myopic terminal-risk ratio: 0.878042, paired-bootstrap 95% interval `[0.835939, 0.923051]`.
- Finite-horizon versus correlation-only terminal-risk ratio: 0.357783, interval `[0.307446, 0.412218]`.

The last two comparisons are equivalent to 12.20% and 64.22% reductions in terminal parameter risk.
The manuscript does not claim area-under-curve dominance over the one-step rule because that comparison was not confirmed.

## Reproduction and source checks

The independent confirmation replay produced byte-identical scientific tables:

- `metrics.csv`: `90AA41E163588617571FB44041D9B2B24A8CF88F41FB4154D140C1617D53B2E2`
- `action_table.csv`: `CC8DBE61059D7A1A432729BD7D675E52E08B2CC762C499A6DF842981717D80B7`
- `seed_cell_metrics.csv`: `411C15E7DCE717587BCFA387653235880776A73DCFF409207A662A13D90B67C0`
- `paired_comparisons.csv`: `C1FBF3AEC76585D157AE584A22C862B354FC80349BC268458E63876D86BB814F`
- `coverage.csv`: `DD62196BAF0F6C2ADE739C56CECDD836C78F0A93D17612C0981AF14EC4825994`

The combined `TSP/tests`, `TSP-V2/tests`, and `TSP-V3/tests` regression suite passed 155 tests in the `ust2` environment.
The LaTeX source has no leading whitespace, forbidden `\qquad`, long-form `Equation` or `Figure` cross-reference, missing citation key, uncited bibliography entry, or duplicate key.

## Build and visual audit

The canonical repository build succeeds with no undefined citation, undefined reference, overfull box, or LaTeX error.
The built-in editor compiler is unavailable on this Windows host because it cannot resolve its platform standard directories; this is an editor-runtime limitation rather than a source diagnostic.
The canonical PDF is 13 IEEE two-column letter pages and has SHA-256 `CC4F4D6AFE0D17F537BDCBE0C5A8BC160C1FC41EE73D6B9B4E339D7CC13C37CB`.
Rendered pages containing the new corollary, simulation result, appendix proof, figures, and final references were visually inspected without clipping, overlap, unreadable labels, or blank spill pages.
