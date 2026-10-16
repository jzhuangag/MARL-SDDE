## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: preregistered CPU development-gate validation
- Origin Date: 2026-09-09
- Verification Status: FROZEN GATE FAILURE; ALIGNMENT CONTROLLER STOPPED
- Experiment ID: MW-AH-DEV-001
- Preregistration commit: `795ebbadda561a105185fcf15e4442005a12fd69`
- Frozen config SHA-256: `6AEF8CFB249A4D5A259A2629A6F538873D883BFE826CFAE8B8310DD369813063`

# Multiwalker privileged Lyapunov-alignment headroom validation

## Decision

MW-AH-DEV-001 fails one of twelve mandatory gates and therefore permanently
stops the Multiwalker **edge-dependent one-update alignment controller**. It
does not authorize selected-feedback alignment-critic fitting, confirmation
seeds, formal evidence, GPU training, or HPC4 work.

This result does not invalidate the separately confirmed cached-profile
`H`-return headroom. It shows that the return effect cannot be identified with
material rotation of the owner's one-update deterministic policy gradient.

## Frozen provenance and execution

The runner, configuration, and preregistration are unchanged from the frozen
commit. Their SHA-256 values are:

- runner: `31A018185B6FE71FD9C029012068E33DDEA93BD31EF881C3DCB5FA982133F28F`;
- config: `6AEF8CFB249A4D5A259A2629A6F538873D883BFE826CFAE8B8310DD369813063`;
- preregistration: `D70A713F150064E1F11A78322F0300195135A03D691403938FA4612CF59DFB5F`.

The primary computation used local `ust2` Python 3.11 CPU processes and the
four frozen seeds `96300--96303`. Eight isolated seed-by-drift processes ran
from 2026-09-08 22:44:09 to 23:18:50 Asia/Shanghai. Before this primary run, a
discarded detached launch was terminated before it wrote any row, summary, or
scientific outcome; it is not part of the result.

All eight primary row artifacts completed. The four low-drift processes then
returned exit code one while attempting to serialize their per-chunk summary:
the analyzer formed a median over an empty active-drift subset and strict JSON
correctly rejected the resulting `NaN`. Every underlying row is finite, and
the frozen merge path analyzes all sixteen rows together with a nonempty active
population. Thus this is a mechanical partial-chunk reporting defect, not a
missing cell or an altered scientific value. It is retained in provenance.
After the frozen validation commit, the partial-chunk analyzer was patched to
emit finite zero-valued active metrics and explicit failed active-only gates
when no active row is present. A full-population post-fix merge preserves both
frozen artifact hashes exactly; the scientific result was not recomputed or
changed.

A clean reproduction used four isolated seed processes, each containing both
drifts, from 2026-09-08 23:22:04 to 2026-09-09 00:06:10. All four returned
exit code zero with empty stderr. The primary and reproduction merges are
byte-identical:

- `alignment_rows.json`:
  `CF77C302BA623ACABBDB9A0BCDD11169A2FDEDE7291D2036070D746BD6110E6F`;
- `summary.json`:
  `BB9576902164A2B9D40DF497D25C4552C6A4A7852F07980B1242A781DD236502`.

Ignored artifacts are stored below
`tmp/policy_dependency_sync/multiwalker_alignment_headroom_dev_v1/`. No GPU,
HPC4, `/project`, or remote artifact store was used.

## Frozen gate result

| Gate | Result |
|---|---:|
| H1 complete and finite | pass; 16/16 rows finite |
| H2 exact MILP optimum | pass; status 0 and reported gap 0 |
| H3 prefix communication budget | pass; maximum excess 0 |
| H4 deterministic replay | pass; maximum error 0 |
| H5 owner coverage | pass; every owner has four launches |
| H6 fully charged diagnostic | pass; 306,144 transitions |
| H7 reference signal | pass in every cell |
| H8 active oracle gain | pass; `1.0464866243e-05` |
| H9 active recovery | pass; `0.9460269595` |
| H10 active direction | pass at boundary; `6/8=0.75` |
| H11 active normalized headroom | **fail**; `1.5224130657e-06 < 0.005` |
| H12 nontrivial packet weights | pass; 18--20 positive weights per active cell |

The observed H11 effect is `0.00015224%` of ideal reference descent and is
about 3,284 times smaller than the frozen `0.5%` threshold. This is not a
borderline power failure. In two active cells the reported oracle is below a
strong feasible schedule by only `1.17e-08` and `1.94e-08`; this is at the
floating MILP resolution of an objective near one. Treating both as ties
would make direction `8/8`, but it would not materially change H11 or the stop
decision. The frozen rows are not edited.

## Scientific interpretation

The high recovery ratio is misleading in isolation: the exact dynamic
schedule recovers most of an extremely small cache-sensitive gradient effect.
Across the active population, its total gain over no refresh is only
`1.0465e-05`, compared with total ideal reference descent `3.03428`.

The causal mechanism is therefore narrower than the confirmed return result.
Refreshing a teammate cache can change the joint trajectory and its short
horizon reward, while the derivative of that return with respect to the
owner's first action remains almost unchanged. No theorem error or simulator
failure is needed to explain the two observations; they estimate different
objects.

The prohibited response is to lower H11, enlarge drift, change the finite-
difference horizon, train on the privileged branches, or assign confirmation
seeds under the same alignment claim. A continuation must change the stated
objective and receive a new outcome-free feasibility audit. The post-outcome
candidate `paired_freshness_reformulation_20260908.md` does this explicitly by
using cache-state return for the launch decision and the realized selected
packet for the receipt weight. It is a theory candidate, not evidence and not
an authorization to run another experiment.
