# Failure-mechanism audit and TSP-V3 decision

## Decision

The repeated negative experiments do **not** have one common cause.  They fall
into three scientifically different classes:

1. the benchmark contains no useful participation headroom;
2. the benchmark has headroom, but the online sensor does not estimate the
   finite-budget quantity used for the return comparison;
3. the scientific mechanism is positive, but a preregistered practical-effect
   threshold is missed narrowly.

Changing a threshold cannot repair classes 1 or 2.  Conversely, stopping the
entire participation program because of class 1 would discard the confirmed
PettingZoo result and the positive MaMuJoCo regime structure.  The TSP-V3
decision is therefore:

- preserve every stopped experiment and its original gate;
- retire the repeated paid-validation sliding-window controller from the
  manuscript path;
- retain the return-free, fully charged probe-then-commit architecture;
- calibrate a task's admissible participation catalogue on development data,
  freeze it before confirmation, and compare the resulting controller with
  the complete fixed catalogue on fresh confirmation seeds.

This is an algorithmic redesign, not a retrospective reinterpretation of a
failed experiment.

## Latest DEV-003 failure: quantified cause

`TSP-V2-MARL-ONLINE-DEV-003` executed correctly.  Its two task-level AUC
changes relative to the strongest fixed comparator were -8.9092% on MPE and
-26.8388% on MaMuJoCo.  The +1% AUC and -2% terminal-return gates were not
unusually strict relative to these deficits.

A read-only audit of all eight controller ledgers gives four causes.

### 1. The observation is dominated by short-horizon noise

The controller uses the risk change across one block as its reward.  Across
the eight controller runs, 48%--73% of observations exceed the registered
`progress_scale=0.005` in magnitude and are clipped.  Mean absolute raw
progress is 0.0105--0.0148, whereas the signed mean is only -0.00067--0.00467.
Thus the standard deviation is roughly 5--20 times the mean signal, and
clipping removes precisely the magnitude information needed to distinguish
participation levels.

The per-q progress ordering is also seed-unstable.  For example, in
independent MaMuJoCo the first seed gives the largest mean progress to q=2,
whereas the second gives it to q=8.  In shared MaMuJoCo the second seed gives
positive apparent progress to q=8 even though fixed-q return AUC selects q=1.
Immediate four-update progress is therefore not a reliable estimator of the
finite-budget return AUC objective.

### 2. The fixed sliding window creates permanent optimism

Every action history is truncated to eight observations, and the confidence
bonus uses the retained sample count.  With four actions, T=100,
`delta=0.05`, and `bonus_scale=0.25`, the bonus cannot fall below

```text
0.25 sqrt(2 log(2*4*100^2/0.05) / 8) = 0.388916.
```

Consequently the controller never enters a genuine exploitation phase.
Across two seeds it spreads 200 choices over q={1,2,4,8}: MaMuJoCo
independent uses (45,70,28,57), MaMuJoCo shared uses (67,52,42,39), MPE
independent uses (34,33,111,22), and MPE shared uses (108,30,45,17).  These
are nondegenerate choices, but not consistent identification of the fixed-q
return optimum.

### 3. Repeated sensing consumes learning budget

The before/after validation pair is correctly charged on every decision.  It
uses 20% of the MPE environment budget and 11.111% of the MaMuJoCo environment
budget.  This cost is real and must not be removed from an analysis.  It is,
however, avoidable: the previously confirmed probe-then-commit controller pays
one separated probe whose message fraction is at most 0.768%.

### 4. The resource queues are not the limiting failure

Environment queues remain identically zero in all eight runs.  Message queues
are finite and reserve feasibility protects both physical budgets.  The
Lyapunov accounting layer works as intended; the failed component is the
learning-value sensor placed inside that layer.

## Historical failures by mechanism

| Experiment family | Observation | Root cause | Consequence |
|---|---|---|---|
| EXP-014B certificate switch | Controller equals fallback in every cell | Cold-start certificate deadlock | Retire strict evidence-before-action switch |
| EXP-017A nonlinear controller | q=1 absorbing state; 50.14% overhead | Information starvation plus surrogate mismatch | Retire that online surrogate |
| T-020 nonlinear audit | Cellwise fixed-q oracle gain 0.3846% | No benchmark headroom | Do not retune a controller on that task |
| SMACv2 | q=8 wins both coupling regimes; oracle gain 0 | No adaptation headroom | Valid negative task, not an algorithm-debug target |
| MPE Speaker--Listener phase scan | Oracle gain 0.312% | Reversal exists but practical value is negligible | Do not require this task in a positive suite |
| simple_reference_v2 | Shared and mixture gains 2.694% and 1.301% vs gates 3% and 1.5% | Narrow practical-effect miss | Preserve as supportive development, not confirmation |
| MaMuJoCo fast extension | Correct regime decisions; mixture +3.550%; per-regime oracle gaps -5.874% and -13.267% | Correlation-only catalogue picks q=1 although development oracle picks q=2 in the shared regime | Calibrate the admissible catalogue before fresh confirmation |
| DEV-003 online progress | MPE -8.909%, MaMuJoCo -26.839% AUC | Noisy clipped proxy, permanent optimism, repeated sensing tax | Retire the V2 online progress controller |
| T-083A dynamic graph | Positive affine effect but formal gate/audit failure and zero temporal correlation in the main population | Claim/evidence mismatch | Keep outside the TSP participation claim |

The table shows that some old gates were conservative, but relaxing gates is
not the proposed remedy.  The successful path must first select tasks with
nonzero oracle headroom and then use an observable that estimates the correct
quantity without consuming a large fraction of the learning budget.

## TSP-V3 success criterion

The target claim is deliberately narrower and stronger:

> Under a registered prior over cross-agent dependence regimes, a fully
> charged return-free probe and a Lyapunov finite-budget catalogue selector
> improve expected learning value over every single fixed participation level.

The standard MARL evidence uses two distinct environments:

- confirmed PettingZoo MPE `simple_spread_v2`, whose complete fixed catalogue
  has oracle actions q=8 and q=1 and for which the controller already improves
  the strongest static q by 2.8117%;
- MaMuJoCo `HalfCheetah-v2/2x3`, where development data show positive 3.5504%
  mixture value and a shared-regime optimum at q=2.

For MaMuJoCo, TSP-V3 freezes the stability-screened online catalogue `{2,8}`
before any new confirmation data.  Fixed q in `{1,2,4,8}` remain in the
comparison.  Excluding q=1 and q=4 from the online catalogue cannot make the
fixed baselines weaker; it only encodes the development-stage controller
design.  The return-free dependence certificate then chooses between q=8 for
low dependence and q=2 for high dependence.

The next experiment is a development crossover using the preserved fixed-q
development curves and newly generated controller trajectories under the same
four development seeds.  It is not manuscript evidence.  It must pass exact
accounting, selection, per-regime fixed-envelope, mixture-value, cost, and
byte-identical replay gates before any fresh-seed confirmation is created.

