# IEEE bibliography style audit

Audit date: 2026-09-09.

## Baseline

The manuscript uses `IEEEtran.bst` under the `IEEEtran` journal class.
The repository had no prior TSP bibliography, so the canonical IEEE skill template was used as the fallback normalization baseline.

## Definite-error checks

- 32 unique entries and 32 cited keys; no duplicate or orphan entry.
- No empty BibTeX field and no BibTeX warning.
- Journal abbreviations, `Proc.` conference style, page ranges, and brace-protected acronyms were checked.
- DOI fields are retained when an authoritative DOI was verified.
- Standard published proceedings entries omit long access URLs from the typeset bibliography; their authoritative URLs remain in the machine-readable citation report.
- IEEEtran compilation produced no bibliography overflow, undefined citation, or line-box warning.

Definite errors: none.

## Project-specific judgment calls

- Official NeurIPS metadata for the 2011, 2015, and two 2019 records does not supply page ranges; those entries retain the official volume and year without inventing pages.
- The PMLR page ranges follow the official proceedings records, including the short official record for Bhandari--Russo--Singal.
- The Crossref given-name typo for Veeravalli is not propagated. The spelling in the publisher article and arXiv record is used, and the discrepancy is recorded in the citation-verification report.

Unresolved judgment calls: none that block the working manuscript.
