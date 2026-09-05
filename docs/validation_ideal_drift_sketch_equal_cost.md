# Validation: equal-cost ideal Lyapunov drift sketch

Date: 2026-09-05.

## Decision

**STOP the coupled-timescale drift-sketch candidate as the ICML mainline.**
The preregistered stopping gates I4, I5 and I6 failed by non-marginal amounts.
No observable coefficient estimator, sampled efficacy experiment, formal run,
GPU benchmark or HPC4 job is authorized.

This conclusion is stronger than an ordinary negative pilot.  The controller
was granted independent samples of the latent Gaussian error state and hence
unbiased access to the five exact Lyapunov-drift coefficients.  Its `S1/S2/U`
split was nevertheless charged against the same 256-trajectory event budget
as the baselines: 13 trajectories estimated the two linear terms, 13
estimated the three quadratic terms, and only 230 remained for the update.

## Frozen provenance

- preregistration commit: `09ed268`;
- source scenario hash:
  `528fbc597015981f8868e2d7aab567003d5b0574fadecff6b462be2c8b746ae0`;
- configuration SHA-256:
  `1ad99e09685d62008de5dd075e0b9379ddd56ad802b6f4cfb70b6915e032c939`;
- runner SHA-256:
  `b32318ebfd5e917c2eb0526ebe5585cc8c0d96569be5e00962a98396cefa0a6f`;
- frozen batch split: `S1=13`, `S2=13`, `U=230`, total `256`;
- frozen sensor seeds: 905001--905016.

The primary and clean reproduction each contain 2,560 rows, including 2,048
primary seed-cells and 256 zero-target seed-cells.  Both 6,347,376-byte JSON
files have SHA-256
`BDA4AF8351F901B7B0AEE0F94AEC19BCC29121632A867F33BA48F9F20545FD96`
and are byte identical.  Primary and reproduction wall times were about 59:53
and 60:02, respectively, on two independent local CPU cores.

## Registered results

| Gate quantity | Threshold | Result | Gate |
|---|---:|---:|---:|
| ideal sketch / best fixed geometric AUC | at most 0.90 | **0.785693** | I3 pass |
| ideal sketch / exact online diagonal AUC | at most 0.97 | **1.016520** | I4 fail |
| seed-cells better than exact diagonal | at least 60% | **32.9102%** | I5 fail |
| exact coupling headroom recovered | at least 60% | **-31.8416%** | I6 fail |
| cells with regret fraction at most 25% | at least 70% | **100%** | I7 pass |
| median coefficient decision regret fraction | descriptive | **3.4281%** | -- |
| zero-target full/diagonal sketch difference | at most `1e-10` | **0** | I8 pass |
| minimum sampled-Q eigenvalue | at least `-1e-9` | **0.0178642** | I1 pass |

I10 passes after the byte-identical reproduction.  Thus I1--I3 and I7--I10
pass, while all three gates that ask whether the paid sketch retains the
coupling advantage over the stronger diagonal baseline fail.

## Interpretation

The result separates two possible causes.  The five-scalar estimation problem
was not the main failure: every primary seed-cell kept accumulated coefficient
decision regret below 25% of split-cost oracle descent, and the median was only
3.43%.  The paid split instead destroys the modest coupling-specific margin.
The original exact coupled oracle improved over exact online diagonal by about
5.19%; after reserving 10.16% of each batch for ideal sensing, the sketch is
1.65% worse than diagonal in aggregate.

There is a descriptive favorable phase: high target sensitivity with low
innovation noise gives an AUC ratio of 0.939807 and improves 92.77% of its
seed-cells.  This was not a registered primary population and cannot rescue
the failed gates.  Restricting future claims to that outcome-selected subset,
lowering the sensing share or weakening the diagonal comparator would violate
the stopping rule.

The correct retained finding is therefore a phase boundary, not a viable
general controller: actor-created critic-target coupling has exact-moment
value, but learning that coupling through a separately paid drift sensor is
not broadly worthwhile under the registered resource model.  A successor
ICML idea must obtain its control signal from mandatory training quantities
without a separate sensing tax, or optimize a different decision with larger
intrinsic dynamic value.  It must receive a new theory and feasibility audit;
this stopped controller must not be renamed and rerun.
