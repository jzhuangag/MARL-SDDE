# IEEE bibliography-style audit — 2026-10-01

## Scope and local standard

- Audited every entry in `references.bib` against the manuscript's `IEEEtran.bst` rendering and the established bibliography style of this manuscript.
- Scope: all 37 entries cited by `main.tex` and `appendices.tex` (10 journal articles and 27 conference papers).
- Citation existence and metadata are checked separately in `citation_verification_20261001.json`.

## Definite errors

None found.

- All 37 cited keys are present, and every BibTeX entry is cited.
- No duplicate key, empty field, unresolved citation, duplicate rendered reference, or BibTeX warning is present.
- Titles retain braces around acronyms and proper names where capitalization must survive IEEEtran processing.
- Journal and proceedings names follow the manuscript's compact IEEE convention.
- Page ranges use en-dash BibTeX syntax, month fields use standard macros where present, and DOI fields agree with the verified authoritative records.
- Personal names with diacritics remain BibTeX escaped and render correctly.

## Venue-specific judgment calls

- NeurIPS records use `Adv. Neural Inf. Process. Syst.` to match the manuscript's IEEE bibliography convention.
- PMLR records retain compact `Proc.` conference names rather than adding publisher-location fields that the local bibliography does not otherwise use.
- Some proceedings entries retain `eprint` and `archivePrefix` provenance. IEEEtran does not duplicate these fields in the rendered bibliography.
- Author fields remain in full natural-name BibTeX form because `IEEEtran.bst` correctly initializes and orders them in the compiled reference list.

## Render check

The final IEEEtran/BibTeX build contains 37 numbered references and reports no undefined citations, bibliography warnings, or malformed reference lines.
