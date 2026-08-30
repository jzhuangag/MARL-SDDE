# Reproducible scientific artifact protocol

Effective after T-081, every experiment that freezes byte-identical
reproduction must separate deterministic scientific content from execution
metadata before its first scientific run.

`summary.json`, endpoint tables, cell tables, seed registries, and gate ledgers
may contain only deterministic scientific values.  Wall-clock duration,
timestamps, process identifiers, host names, and scheduler metadata belong in
`execution.json` or a run ledger.  A byte-identity gate applies to the declared
scientific artifacts.  Runtime gates are evaluated from `execution.json` but
do not make `summary.json` nondeterministic.

The shared writer rejects top-level scientific-summary keys containing
`runtime` or `wall`.  Each preregistration must enumerate the files covered by
its byte-identity gate.  This protocol does not amend or reinterpret T-081;
its C13 remains failed.
