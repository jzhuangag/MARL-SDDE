# TSP-V3-CERT-SENS-001 development validation

## Decision

Stop after development.  Confirmation is not authorized and none of these
results may enter the manuscript as confirmation evidence.

The frozen configuration hash was
`2933ab41354b24b4998a9f1305c823c4210988ecfd9196b34e7a7a265348b8dc`.
Eight fresh development seeds produced 19,680 finite checkpoint rows with no
resource-budget violation.  All selected coefficient recursions were finite
and stable.

## Frozen gates

| Gate | Frozen threshold | Observed | Result |
|---|---:|---:|---:|
| Moderate conservative / nominal terminal parameter risk | at most 1.10 | 1.328771 | fail |
| Strong conservative / nominal terminal parameter risk | at most 1.25 | 2.016694 | fail |
| Finite horizon / one-step myopic terminal parameter risk | at most 0.95 | 0.854392 | pass |
| Finite horizon / correlation-only terminal parameter risk | at most 0.98 | 0.343449 | pass |

The two conservative certificate levels covered all 24 level-by-cell
one-sided 99 percent bootstrap upper means.  Thus the safety side behaved as
predicted, but using the most conservative score for both feasibility and
ranking paid more learning cost than the frozen tolerances allowed.

The finite-horizon design nevertheless showed the intended ablation mechanism:
terminal parameter risk was 14.56 percent below one-step myopic drift and
65.66 percent below the correlation-only controller.  Because the joint
development decision failed, these values remain development diagnostics and
do not authorize a cherry-picked confirmation.

## Reproduction

The complete run and a clean replay produced byte-identical hashes for every
scientific table:

| File | SHA-256 |
|---|---|
| `metrics.csv` | `5166D09E71312DD133B6A660F9E6585BDF55C2CA9690B7E24197A653DA3EFDF1` |
| `action_table.csv` | `F7EA653CEED23C82EB54C7C5092AF9FAE8F15242CAE6462DF1CFDFBF03BF0563` |
| `seed_cell_metrics.csv` | `6CC21D10BB0BEDA859264324CC76A808887EDBE98DEA262B36AC6F0766F929B7` |
| `paired_comparisons.csv` | `854C1B23C59F153FC210D61F95DF333073851ADF16BCD0B81AB163221DBD72A2` |
| `coverage.csv` | `E51F7399F827D74336BBF2E11AB9B9F368310E287DE2CC13C477628ED1C21FBD` |

The differing wall-clock field in `summary.json` is operational metadata and
is excluded from the byte-replay requirement.

## Mechanism and next design constraint

The failure isolates a design issue rather than invalidating the robustness
corollary.  Conservative coefficients are appropriate for defining the safe
action set and for reporting a valid risk envelope.  Re-optimizing the entire
performance score under simultaneous worst-case curvature, noise, and mixing
bounds can be unnecessarily pessimistic.  Any successor must therefore keep
the upper certificate as a hard feasibility shield while separating it from
the nominal finite-horizon ranking objective; it must use new seeds and an
independent preregistration.
