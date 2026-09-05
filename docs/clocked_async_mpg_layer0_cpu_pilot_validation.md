# Clocked MPG Layer-0 fresh-seed CPU pilot validation

## Decision

The pilot is a reproducible failure.  Four of nine mandatory gates passed, so
no GPU pilot, HPC4 job, formal seed, or standard benchmark is authorized.
Neither thresholds nor seeds were changed after outcome access.

- Preregistration commit: `40d8ee4f498fa04e98347aa55911761377d32618`.
- Configuration SHA-256:
  `98d0afe3a461d100aff14863ecae400d3fbd2175ec0fbcc588309a87fed56e35`.
- Primary and isolated reproduction SHA-256:
  `a09494d2c6b4b6b3336b3584c7c9c6fb9c40cc3260bc59ffb8e34861c19321b7`.
- Seeds: 3701--3708, unseen before the frozen commit.
- Rows: 48; all finite and exactly charged.

## Gate ledger

| gate | result | observed value |
|---|---:|---|
| L1 validity/accounting | pass | 48 finite rows; 61,200 charged/completed actor transitions each; zero self-fresh error |
| L2 positive learning | pass | strategic mean return change `+1.5093` balanced, `+1.9245` heterogeneous |
| L3 heterogeneous lower tail | **fail** | required `+2.5386`; observed `-9.1116` versus full-data raw |
| L4 heterogeneous mean safety | **fail** | relative shortfall `8.5503%` versus maximum `1%` |
| L5 heterogeneous directionality | **fail** | `1/8 = 12.5%` versus required `5/8` |
| L6 nontrivial debt control | pass | intermediate scales `31.2813%` balanced, `34.4375%` heterogeneous |
| L7 balanced mean safety | **fail** | relative shortfall `9.1234%` versus maximum `1%` |
| L8 split-cost value | **fail** | does not beat raw-half in both mean and lower tail across both profiles |
| L9 reproducibility | pass | primary and reproduction byte-identical |

## Aggregate outcomes

Higher return is better.

| service | method | mean final return | lower-quartile mean | mean change | positive-change seeds |
|---|---|---:|---:|---:|---:|
| balanced | strategic split | -122.2915 | -132.7163 | +1.5093 | 6/8 |
| balanced | raw full data | -112.0672 | -126.1822 | +11.7336 | 7/8 |
| balanced | raw half data | -115.7865 | -128.4996 | +8.0144 | 6/8 |
| heterogeneous | strategic split | -121.8763 | -136.0436 | +1.9245 | 6/8 |
| heterogeneous | raw full data | -112.2763 | -126.9320 | +11.5245 | 7/8 |
| heterogeneous | raw half data | -122.9036 | -134.4655 | +0.8972 | 5/8 |

The controller is not a trivial fallback: its mean scale is `0.1552` under
balanced service and `0.1484` under heterogeneous service, and roughly one
third of its scales are strictly between zero and one.  It also learns in both
profiles.  The failure is therefore scientific rather than mechanical: the
accepted updates are too small and too information-inefficient to compete with
the strong full-data raw baseline.

## Failure localization

The pilot rejects the current implementation, not the entire clocked strategic
learning question.

1. **The validation trajectory is independent but not arrival-fresh.**  Both
   proposal and validation trajectories are sampled under the packet's birth
   joint policy.  The validation gradient can detect sampling disagreement but
   cannot directly identify whether teammate changes have rotated the current
   strategic gradient.
2. **The scalar action cannot repair a stale direction.**  It can only multiply
   the birth proposal by a number in `[0,1]`.  When the current ascent direction
   rotates, every positive scalar remains misaligned.
3. **The strong baseline uses all charged data for learning.**  Full-data raw
   averages both gradients; strategic split uses the second half only as a
   sensor.  This price is visible even under balanced service.
4. **The debt is active but over-attenuates.**  Setting `V=epsilon` makes the
   Lyapunov queue dimensionally active, yet the resulting mean scale near
   `0.15` gives up most of the available improvement.  Merely retuning `V`
   would not solve the first three structural defects.

## Next admissible algorithmic step

The old scalar controller is stopped for standard MARL.  The next version must
be a new, separately audited mechanism with:

- a fully charged short validation rollout launched at packet arrival under
  the current joint policy;
- a vector proximal/trust-region correction that can rotate the stale proposal
  toward the arrival-fresh gradient, rather than only shrinking it;
- a closed-form O(d) action and a Lyapunov queue expressed in consistent return
  units;
- a strong full-data raw baseline with the same total trajectory charge;
- an outcome-free CPU oracle/headroom gate before any new fresh-seed pilot.

This change preserves the unified research story: asynchronous strategic
updates create endogenous cross-policy drift; fresh sensing identifies the
current value of a completed packet; a Lyapunov controller spends a declared
risk budget while correcting, accepting, or suppressing it.  It changes the
algorithmic response from scalar filtering to freshness correction.
