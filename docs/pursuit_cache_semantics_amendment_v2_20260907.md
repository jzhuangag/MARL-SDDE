## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: pre-execution numerical evaluator amendment
- Origin Date: 2026-09-07
- Verification Status: FROZEN V2 DESIGN; V1 FAILURE PRESERVED
- Version Label: pursuit_cache_semantics_qualification_v2

# Pursuit cache semantics v2 amendment

The v1 audit remains a byte-exact `6/7` failure. Its C3 evaluator subtracted
two full fixed-universe energies to recover one edge reset, creating an absolute
`3.49e-7` cancellation error against a `1e-10` gate. The topology theorem,
cache state machine, and gate threshold are unchanged.

V2 changes only the numerical identity evaluator. It computes the selected
edge's energy before and after the exact cache copy, then compares that local
difference with the declared reset. The full-universe subtraction is retained
as a descriptive cancellation diagnostic and does not determine the gate. This
is the stable algebraic quantity used by the controller itself.

The seven frozen checks and all environment parameters remain unchanged. V2
uses untouched seeds `92100--92107` and the same action seed `9981`. Primary and
isolated reproduction JSON must be byte-identical.

Frozen v2 hashes:

- audit runner: `9b3dcfcde0b5296ab0917489462f7caf97a94dbde113bc869a9d325a2fc2b879`;
- audit tests: `93f65b5ed876bf5a147fa24c0ff00828702fe2dc410086b17ff03556c6fa0cfc`;
- packet/cache state machine: `fc5f5250fa30cc8345df02dc43df11816042e9d5808cebbf0bc3f5cc6c98f899`;
- cache/topology algebra: `770d412940584a2383b2c8e6292b52d5f1a6b2699cbce63848628c0cb56a700b`.

This remains an outcome-free CPU interface audit. It does not authorize a GPU
efficacy experiment.
