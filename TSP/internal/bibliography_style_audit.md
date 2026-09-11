# IEEE bibliography style audit

Audit date: 2026-09-11.

## Baseline

The manuscript uses `IEEEtran.bst` under the `IEEEtran` journal class.
The repository had no prior TSP bibliography, so the canonical IEEE skill template was used as the fallback normalization baseline.

## Definite-error checks

- 36 unique entries and 36 cited keys; no duplicate or orphan entry.
- No empty BibTeX field and no BibTeX warning.
- Journal abbreviations, `Proc.` conference style, page ranges, and brace-protected acronyms were checked.
- DOI fields are retained when an authoritative DOI was verified.
- Standard published proceedings entries omit long access URLs from the typeset bibliography; their authoritative URLs remain in the machine-readable citation report.
- The PettingZoo and MAPPO records match the manuscript's existing NeurIPS proceedings shape, including natural-order author names, abbreviated venue, volume, pages, and protected acronyms.
- The Yu--Chen--Poor record matches the authoritative IEEE and Crossref metadata for DOI `10.1109/TSP.2025.3546574`; its abbreviated journal, volume, pages, month, and year follow the manuscript's IEEE journal-entry style.
- The Perazzone--Wang--Ji--Chan record matches IEEE, arXiv, DBLP, and Crossref metadata for DOI `10.1109/TON.2025.3539857`; its journal, issue, pages, month, and year follow the same style.
- IEEEtran compilation produced no bibliography overflow, undefined citation, or overfull line-box warning. The remaining underfull boxes are visually benign in the rendered two-column layout.

Definite errors: none.

## Project-specific judgment calls

- Official NeurIPS metadata for the 2011, 2015, and two 2019 records does not supply page ranges; those entries retain the official volume and year without inventing pages.
- NeurIPS entries omit conference month and address consistently with the project bibliography; this is a project-specific proceedings convention rather than missing invented metadata.
- The PMLR page ranges follow the official proceedings records, including the short official record for Bhandari--Russo--Singal.
- The Crossref given-name typo for Veeravalli is not propagated. The spelling in the publisher article and arXiv record is used, and the discrepancy is recorded in the citation-verification report.

Unresolved judgment calls: none that block the working manuscript.
