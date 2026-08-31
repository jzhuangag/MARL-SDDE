# Autonomous CPU research plan — 2026-08-31

Hourly in-thread heartbeat `sdde-cpu` is ACTIVE. It checks current processes before starting work; this is periodic continuation, not an always-running scientific process. Local machine and app must remain available.

1. Completed: T-083A result/reproduction audit and full regression (663 passed, 7 skipped). Frozen decision FAIL: reproduction F12 and strict F13. See validation_t083a_formal_end_block_confirmation.md.
2. In progress: certificate double counting confirmed by source inspection and a deterministic four-row counterexample; see end_block_certificate_audit_20260831.md. Next audit coverage, empirical debt versus true risk, Markov block law and scaling. Do not modify frozen controller/results.
3. In progress: end_block_debt_theory_audit_20260831.md derives the exact pre-mix paired-loss identity, nonzero AR(1) conditional bias, queue telescoping/drift, and a conditional convex target-interval shield. Five deterministic algebra tests pass. Cross-block noise is continuous, not reset. Post-mix risk and queue stability are not yet covered. Next inspect exact-moment baseline law alignment, then target coverage under explicit noise/mixing assumptions. Formal outcomes must not tune parameters.
4. Conditional: only after theory/interface and feasibility checks, preregister a separate CPU validation with fresh seeds; execute once, reproduce, retain every gate failure.
5. Update this plan and ledger with evidence; commit/push reviewed changes. Do not start GPU/HPC4 or perform remote storage cleanup. Report a concrete GPU handoff only when needed. Pause heartbeat when authorized work is complete or user input is necessary.

Unifying question: when can causal time-varying collaboration improve personalized finite-time learning under resource and delay constraints? The current small affine simulator is not an ICML-ready full experimental package. No acceptance or positive-outcome guarantee is made.
