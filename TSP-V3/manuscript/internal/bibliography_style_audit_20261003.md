# IEEE bibliography-style audit — 2026-10-03

## Scope and local standard

All 37 entries in `references.bib` were re-audited against the manuscript's established `IEEEtran.bst` output and compact IEEE venue style.
The bibliography file is unchanged by the robust-certificate revision.
Citation existence and metadata are verified separately by `citation_verification_20261003.json`.

## Definite errors

None found.

- All 37 cited keys are present, and every entry is cited.
- There are no duplicate keys, missing keys, empty rendered entries, duplicate rendered references, or BibTeX warnings.
- Titles preserve braces around acronyms and proper names where IEEEtran case conversion requires them.
- Journal and proceedings names use the manuscript's compact IEEE convention.
- Page ranges, DOI fields, and author-name rendering remain consistent with the authoritative metadata records.

## Venue-specific style choices

- NeurIPS records use `Adv. Neural Inf. Process. Syst.`.
- PMLR records retain compact `Proc.` conference names instead of adding location fields not used by the local bibliography.
- Some proceedings entries retain `eprint` and `archivePrefix` provenance, which IEEEtran does not duplicate in the rendered list.
- Full natural-name author fields are retained because `IEEEtran.bst` initializes and orders them correctly.

## Render check

The canonical IEEEtran/BibTeX build contains 37 numbered references and reports no undefined citation, bibliography warning, or malformed reference line.
