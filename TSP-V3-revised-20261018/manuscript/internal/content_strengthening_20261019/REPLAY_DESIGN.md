# Existing-evidence gate reconstruction (frozen before new outputs)

## Material Passport

- Origin: academic-research-suite, user-authorized substantive V3 revision, 2026-10-19.
- Mode: deterministic analysis of existing evidence, not a new simulation.
- Prior outcomes are known. This is NOT a prospective registration or independent validation.
- Original V3, EXP-016B sources/data/gates and the native 140h study are immutable.

## Claim intent

1. Give the same-stream risk-corrected gate an explicit implementable rule and proof.
2. Construct its empirical cell means from already saved potential-outcome branches.
3. Report admission, rejection, conservatism, adverse cells and all old comparators.
4. Do not claim unknown-model calibration, neural-MARL guarantees, new-seed confirmation,
   empirical confidence intervals, or global priority.

## Frozen reconstruction

Use EVERY cell of the corrected `same_stream_directional_evidence_20261018/risk_bounds.csv`.
The sole probe candidate is the frozen EXP-016B `information_only` program, when
it probes; do not search a new catalogue or change any probe/action/seed/budget.
Its two post-identification programs are frozen before probing. For each public
scenario/budget/layer, compute BOTH hypothetical-instance upper risks U_j from
Theorem 5. Let F_j be the full-budget `always_all` commitment risk. Admit iff
U_1 < F_1 AND U_0 <= F_0 (epsilon=0); otherwise choose `always_all` without probe.
This rule uses no observed terminal risk or actual unknown regime. As an ablation,
apply the identical rule after subtracting only the sqrt(delta_j)*Gamma_j term.
Directional errors are the saved exact per-probe affinity bounds, NOT nominal .025.

On admission, use the saved `information_only` empirical mean and cost; on rejection,
use the saved `always_all` mean and cost. This is exact at the empirical-mean level:
the decision is pre-observation deterministic, and `run_exp016b_pilot.py` generates
policy-independent potential observations once per seed/scenario/regime/layer.
It is not a simulator execution of a new policy, and no new independent sample is obtained.
All 192 saved seeds/cell contribute through the unchanged cell means. No new CI.

Report all 8 original policies, corrected gate and no-carryover ablation, all layers,
all budget points, both regimes, including search-censored and adverse families.
Use unweighted cell means descriptively; preserve every cell in output.
No success-dependent filtering or adaptive repetition. Any mismatch stops analysis.

## Compute authority

This read-only algebra/CSV analysis is authorized existing-evidence work. Record
its measured process CPU separately; it is not charged to or run by the 140h driver.
No trajectory generation, training, random draws or experiment entry points.
Fresh scientific validation remains subject to the separate proposed CPU budget.
