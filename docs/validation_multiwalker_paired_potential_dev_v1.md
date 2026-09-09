## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: validate
- Origin Date: 2026-09-09
- Verification Status: ANALYZED; INDEPENDENT REPRODUCTION INCOMPLETE
- Version Label: validation_multiwalker_paired_potential_dev_v1

# MW-PF-DEV-001 validation

## Decision

The frozen potential-only bridge fails and is permanently stopped.
The primary run is complete and internally consistent, but it supplies no
evidence that the exact cache-reset energy is a useful continuation-value
proxy.  No catalogue value, seed, comparator, threshold, or gate was changed
after outcomes were generated.  This result does not authorize a learned
critic, a fresh-seed confirmation, a standard MARL benchmark, GPU work, or
HPC4 work.

The failure is specific and informative.  The strengthened exact dynamic
oracle retains positive headroom, whereas every leave-one-seed-out fold selects
zero reset weight.  Thus the benchmark still contains intertemporal graph
value, but the cache potential tested here does not recover it.

## Frozen provenance

- Preregistration commit: `ba7fb7ed949a5e81eaa353285a28ec49926c598e`.
- Execution source state: descendants of the preregistration through
  `aa4cb3a5a6e78cf91ff4dad5d76507949a9a6a16`; the frozen runner and config
  identities below did not change.
- Config SHA-256:
  `8EC001E4F394918673CA09BC3F6749869592EAEFEF5B474039D149E8B86BAC8C`.
- Runner/analyzer SHA-256:
  `9E50453A798F1B364A4E7BCAAE29FE5BC7D9A0536F9FDB87F555F8810A2D7227`.
- Authenticated baseline rows SHA-256:
  `0F523DC6BA9474BE008533E1FAA90E6A8700CEF246970C6FAD9823080F313ECC`.
- Authenticated exact rows SHA-256:
  `4CD2556B605FBD63659100165FD2A95C39FAE87BE551999B697A2A93046F0FF5`.
- Primary output: 320 rows over all eight registered design seeds.
- Primary rows SHA-256:
  `7CFACB7D00613738A93CF328BB813E1CF978968851C2ABA278FA525367A74FE1`.
- Primary summary SHA-256:
  `25ABDD2503D1F7C506A1037B3F13CACCE2C15E5790CC4D29A02465EE545FCE8A`.

Four isolated two-seed chunks completed, each with 80 rows and a manifest.
A deterministic audit re-merge on 2026-09-09 reproduced both primary hashes
byte for byte.  Four clean reproduction processes were launched separately,
but they later disappeared without producing a reproduction directory or
completion artifacts.  Their terminal exit states cannot be recovered from
the current process table.  They are therefore recorded as an incomplete
reproduction attempt, not silently retried and not counted as verification.

## Frozen gate results

| Gate | Result | Observed value |
|---|---:|---:|
| P1 complete grid | pass | 320 unique rows |
| P2 finite, replay, prefix, and radius validity | pass | all valid |
| P3 cross-fitted frozen catalogue | pass | all choices compliant |
| P4 strengthened exact-oracle headroom positive | pass | 2.4527007042 |
| P5 active recovery at least 10% | **fail** | -0.0158560084 |
| P6 active directional rate at least 75% | **fail** | 0.4375 |
| P7 active median normalized headroom at least 0.05% | **fail** | -0.0000044665 |
| P8 nonzero reset cap selected in every fold | **fail** | 0/8 folds |
| P9 queue changes at least one active choice | pass | 37 choices |
| P10 aggregate gain over no refresh positive | pass | 6.6712690159 |

The cross-fitted candidate has aggregate active headroom
`-0.0388900430` relative to the strengthened online envelope.  Every held-out
seed selects `(reset_bonus_cap, queue_step)=(0,0.0025)`.  The tested cache
energy therefore contributes no selected control signal.  Passing P9 does not
rescue the method: the queue changes actions, but those actions do not meet the
registered learning-value gates.

## Statistical interpretation

These eight seeds were explicitly reused as post-outcome design data.  The
reported quantities are complete-population descriptive gate values, not
p-values, confidence intervals, or confirmatory effect estimates.  The exact
dynamic oracle is outcome-aware and is used only to diagnose remaining
headroom.  Its active normalized headroom is positive in 16/16 seed-budget
cells, with median `0.00713693` and mean `0.00813574`, but this is a mechanism
ceiling rather than deployable-controller evidence.

## Fallacy scan

Coverage: 11/11 statistical fallacy types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Aggregate failure agrees with the subpopulation gate failures; cell heterogeneity is retained in the row artifact. |
| Ecological fallacy | NOTE | No claim about individual agents is inferred from seed-level aggregates. |
| Berkson's paradox | CAUTION | The eight seeds are a previously inspected design population and cannot support confirmation. |
| Collider bias | NOTE | No post-outcome conditioning variable is used in the frozen cross-fit rule. |
| Base-rate neglect | NOTE | No diagnostic classifier or prevalence claim is made. |
| Regression to the mean | NOTE | No extreme-seed selection or pre/post claim is used. |
| Survivorship bias | NOTE | All registered primary seeds and rows are present; the failed reproduction attempt is reported separately. |
| Look-elsewhere effect | CAUTION | Ten catalogue pairs are screened, but selection is leave-one-seed-out and only the preregistered candidate is judged. |
| Garden of forking paths | CAUTION | This is development evidence; the frozen catalogue and gates prevent retrospective rescue, but it is not confirmation. |
| Correlation versus causation | NOTE | The candidate is executed in the simulator; general MARL efficacy is not inferred. |
| Reverse causality | NOTE | Launch actions precede simulated returns; no observational directional claim is made. |

## Reproducibility verdict

- Deterministic chunk merge: **REPRODUCIBLE**, byte-exact for rows and summary.
- Independent end-to-end rerun: **CANNOT VERIFY**, because the clean processes
  left no terminal record or artifact.
- Overall validation confidence: **CAUTION** for reproducibility and
  **definitive failure** with respect to the frozen development gates.

## Regression verification

- MW-PF and controlled-cache targeted tests: 29 passed in 29.69 seconds.
- Complete `experiments/policy_dependency_sync` regression: 694 passed with
  three pre-existing Box2D SWIG deprecation warnings in 350.10 seconds.
- JSON parse and repository whitespace checks passed.

## Consequence for the research program

The potential-only continuation model must not be tuned or revived under a new
identifier.  A possible successor must model the missing conditional future
value explicitly and must first close its causal launch/receipt interface,
Markov-bias control, confidence bound, and actual cache-dependent rollout
feasibility.  The drift-structured residual-value note is such a theory
candidate, not yet an authorized experiment or an ICML-ready algorithm.
