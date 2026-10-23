# IEEE bibliography-style audit — 2026-09-30

## Scope and standard

- Audited `references.bib` against the manuscript's own IEEEtran bibliography rendering and the established style of the preserved TSP manuscript.
- Scope: all 37 entries cited by `main.tex` and `appendices.tex` (10 journal articles and 27 conference papers).
- Citation existence and metadata are checked separately in `citation_verification_20260930.json`.

## Definite errors

None found.

- 37 citation keys are used and 37 BibTeX entries are present.
- No cited key is missing; no entry is uncited.
- No duplicate key, empty field, or BibTeX warning is present.
- Titles that require capitalization protection retain braces around acronyms and proper names.
- Journal and conference fields follow the manuscript's compact IEEE abbreviations.
- DOI fields are used when an authoritative DOI exists; official proceedings and arXiv identifiers are retained where they are the canonical public record.

## Venue-specific judgment calls

- NeurIPS records use `Adv. Neural Inf. Process. Syst.` to match the local IEEE bibliography convention rather than spelling out the proceedings title.
- PMLR records retain the established `Proc. Mach. Learn. Res.` form.
- Some proceedings entries retain `eprint` and `archivePrefix` in addition to volume and pages. IEEEtran does not print these redundantly in the compiled bibliography, and they provide useful provenance, so they are retained.
- Personal names with diacritics remain BibTeX escaped (for example, `B{\"o}hmer`) so IEEEtran renders them correctly.

## Render check

The final IEEEtran/BibTeX build contains 37 numbered references and reports no undefined citations or bibliography warnings.
