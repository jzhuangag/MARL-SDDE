# T-032 exact full-risk validation

## Decision

T-032 fails four mandatory gates and permanently stops the proposed
fresh-diversity subset selector under the registered equal-variance block
AR(1) model.  No selector theorem, sampled learning pilot, formal seeds, GPU,
or HPC4 run is authorized.

This is a mechanism failure rather than a power failure.  The evaluated
quantity is an exact finite-time oracle ceiling with no estimation, probing,
or controller overhead.  A deployable selector cannot exceed it.

## Provenance

- preregistration commit:
  `8847e5f934d0ab7b9bcf4d682aa39027049105e0`;
- frozen configuration SHA-256:
  `826bc4fde0231a017a719ddd76e937916400a1018e5815a69b97769b37511c25`;
- primary result SHA-256:
  `5A8666B49AC59670782CC3F18576618636715777195551C64DF97C117A36E9C6`;
- isolated CPU reproduction: byte-identical;
- analytic cells: 576, including 144 active and 144 homogeneous controls;
- scientific trajectories and seeds: zero.

## Frozen gate ledger

| Gate | Frozen requirement | Result | Status |
|---|---:|---:|---:|
| F1 | all 576 cells finite | 576/576 | pass |
| F2 | zero budget violations | 0 | pass |
| F3 | active aggregate improvement at least 15% | 0.628982% | **fail** |
| F4 | at least 70% active cells improve by 5% | 6/144 = 4.1667% | **fail** |
| F5 | homogeneous ratio in [0.98, 1.02] | 1.0000 | pass |
| F6 | at least three oracle structures | 7 | pass |
| F7 | every budget ray contains a 5% path | message 0%; environment 0%; wall 18.203% | **fail** |
| F8 | median count-only separation at least 1.10 | 1.0000 | **fail** |
| F9 | exact full-risk recursion | yes | pass |
| F10 | no prior-outcome input | yes | pass |
| F11 | CPU only | yes | pass |

Overall: **7/11 passed**.  The registered rule requires every mandatory gate.

## Read-only diagnosis

All subgroup analyses below preserve the frozen population and gates.

| Active subgroup | Mean oracle improvement | Maximum |
|---|---:|---:|
| message binding | 0% | 0% |
| environment binding | 0% | 0% |
| wall binding | 1.7643% | 18.2030% |
| participation m=4 | 1.1762% | 18.2030% |
| participation m=8 | 0% | 0% |
| balanced layout | 0% | 0% |
| clustered layout | 1.1884% | 18.2030% |
| permuted layout | 0.5759% | 16.6229% |

The best single-factor baseline already absorbs essentially all diversity and
freshness value.  The remaining benefit is sparse and confined to wall-clock
cells.  Restricting the paper to those cells would be a post-outcome endpoint
change and would reduce the contribution to a narrow systems-scheduling
effect.

## Scientific interpretation

T-032 does not refute the dependence-adjusted speedup law or the possibility
of useful correlation structure in another model.  It does refute the claim
that equal-count selection over equal-variance correlated agents supplies a
broad, meaningful learning gain in the preregistered family.  Changing to
heteroscedastic agents would define a new problem and would risk collapsing
to familiar quality-aware client selection.

The only algorithmic direction retained for a fresh audit is low-rank shared
factor correction: aggregate all useful idiosyncratic information while
cancelling an identifiable common Markov factor.  That direction is not yet a
claim, experiment, or authorization.
