# Authentic author metadata — post-build review,2026-10-18

Decision: AUTHOR_METADATA_VERIFIED_SUBMISSION_DECLARATIONS_PENDING.

The direct human-provided block is inserted into the independent revised copy.
Ordered authors are Jialin Zhuang (Member, IEEE) and Vincent K. N. Lau (Fellow,
IEEE); HKUST ECE affiliation, both supplied emails, Vincent K. N. Lau as
corresponding author and the two supplied RGC AoE grants are retained. Only two
nonbreaking email boxes are added for display. ORCIDs, submission disclosures
and author consent are not inferred from the author block.

The reproducible author_metadata_gate.py compares current main.tex with technical
audit commit3407febe1b5018bb2c6e2cda590a1018b652c00b: only the author block changed.
Appendix, bibliography and tracked figures match that commit byte-for-byte;
the original V3 manuscript/source/Bib/PDF hashes still match its frozen audit.
All previous scientific claims, proofs and evidence retain their prior scope.
No native scientific data are generated/read for this update.

The single final build succeeds:13 pages, default10pt IEEEtran journal/two
columns/one-inch margins,218-word abbreviation-free abstract/five keywords,
zero undefined items/overfull boxes and26 underfull warnings. All fonts embed.
All13 pages of the author draft were inspected. Nonbreaking email corrections
changed only page1; final render pages2-13 are byte-identical to the inspected
images, and final page1 was inspected again. No clipping, overlap or displaced
figure/proof content is visible. Both emails extract correctly as complete
addresses in the final PDF. Current PDF SHA256:
7a452b042f6ce8270b33e02648f0c5ded36aa4f457eb98ddc09dbfe19d937ed3.

Fresh lookup results for all37 references are saved in a new directory, without
overwriting earlier evidence. Raw36 PASS and the Paulin number/issue conflict are
kept as reported. A fresh institutional page confirms volume20,2015, article79,
32 pages and DOI10.1214/EJP.v20-4039; the conflict is representation rather than
identity. Existing no.79 rendering is a documented style judgment; BibTeX was
not edited under this metadata-only request. Resolver outages remain explicit.
The separate IEEE style report finds no definite error. With the now-known author
names, the retained37-reference author lists contain zero Zhuang/Lau self-cites.
The citation request's source fingerprint is before only the two nonbreaking
email wrappers; the gate reconstructs it exactly and checks Bib/other source
bytes unchanged. This does not reuse cached PASS records.

Artifacts: author_metadata_release_check_20261018.json,
author_metadata_unbroken_preflight_20261018.json,
author_metadata_unbroken_render_20261018/page-01.png through page-13.png,
author_metadata_citation_20261018/ and
author_metadata_citation_resolution_20261018.json. Prior technical proof/context
audit remains authoritative and preserved, not rerun as new research evidence.

Outstanding human checks are prior/concurrent/related-manuscript disclosures,
all-author consent/final release, and authentic ORCIDs where the system requires
them. No actual submission, independent external review, full-text plagiarism
clearance or acceptance guarantee is claimed. The independent140h science owner
continues under its unchanged cap/gates; its outcomes are not used in V3.

Skills used: academic-research-suite (format/citation integrity),
ieee-tsp-paper-writing, publishable-academic-writing, ieee-bib-style-audit and
pdf:pdf (render/visual verification).
