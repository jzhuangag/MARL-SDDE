# T-020 nonlinear adaptation-value ceiling

## Scope

T-020 is a read-only feasibility audit based on commit
`e73da9f54edcf539248757c6a93ff81acac8ad5c` and the existing EXP-017A pilot.
It creates no scientific trajectory, performs no significance test, submits no
Slurm/GPU job, and does not modify EXP-017A or T-019.

The endpoint source is
`/scratch/jzhuangag/exp017a-pilot-17a4c32/endpoints.csv`, SHA-256
`bc241c772d20b76c5f42f72bd8a5523bda2ba225e113811e695dd840007191f0`.
Only the 576 rows for q in `{1,4,16,32}` are used. The two pilot seeds are
design information only; no p-value, confidence interval, or formal claim is
computed.

## Correct strong fixed baseline

For each task × budget, the strong baseline is selected from all four fixed
participation levels using the geometric terminal prediction error aggregated
over both pilot seeds and every registered mixing, rho, and delay cell. q=1 is
eligible: its validity is independent of the old controller's absorbing state
because the proposed probe and learning participation are separate.

| Task | Budget | Selected q | Geometric terminal error |
|---|---|---:|---:|
| Acrobot | environment-binding | 16 | 32.51484 |
| Acrobot | message-binding | 1 | 32.62917 |
| CartPole | environment-binding | 32 | 33.29212 |
| CartPole | message-binding | 1 | 34.57329 |

This corrects the prospective non-q1 fallback table in
`exp017b_static_design.md`; T-020 does not rewrite that historical T-019
record.

## Cellwise fixed-q oracle ceiling

Within each of the 72 task × mixing × rho × delay × budget cells, a descriptive
oracle chooses the q with the smallest two-seed geometric terminal error. This
is deliberately outcome-aware and therefore only an upper-bound diagnostic,
not a deployable controller or formal comparator.

- oracle/fallback geometric ratio: `0.9961544268`;
- geometric improvement: `0.0038455732` = **0.384557%**;
- strictly improved cells: `21/72` = **29.1667%**;
- cells with at least 2% improvement: **6/72**;
- cells with at least 5% improvement: **0/72**;
- maximum cell improvement: `4.18839%`.

The oracle includes the fallback in its candidate set, so its remaining 51
cells tie rather than become worse. The very small aggregate gain is not a
power problem: it is the most favorable cellwise selection available from the
registered fixed-q outcomes.

## Full probe-cost ceiling

The complete public T-019 probe schedule is charged: q_probe=4, b_probe=1,
the first eight controller blocks, then every 32nd block. For every seed-cell
arm, T-020 solves for the largest number of learning updates satisfying both

\[
u\,C_m(q)+N_p(u)C_m(4)\leq M,
\qquad
u+N_p(u)\leq E,
\]

where `N_p(u)` is the scheduled probe count. Probe gradients are not learning
updates.

Endpoints do not identify the terminal error or AUC that would result after
removing learning updates. T-020 therefore reports an explicitly optimistic
ceiling: it deducts every probe resource but assumes terminal error, CVaR90,
and AUC remain as good as in the no-probe run. Actual post-probe performance
could be worse; it cannot be reconstructed without a new trajectory.

| Metric | Task×budget fallback | Cellwise oracle | Oracle/fallback |
|---|---:|---:|---:|
| Geometric terminal error | 33.24240 | 33.11457 | 0.996154 |
| Mean cellwise two-seed CVaR90 | 34.04697 | 33.92218 | 0.996335 |
| Pooled seed-cell CVaR90 | 36.14075 | 35.72691 | 0.988549 |
| Mean normalized prediction AUC | 0.352769 | 0.384981 | 1.091312 |
| Mean no-probe learning updates | 1262.25 | 1139.00 | — |
| Mean usable learning updates after probe | 1250.50 | 1129.00 | — |
| Usable/no-probe update fraction | 0.990691 | 0.991220 | — |
| Mean probe count | 10.000 | 9.611 | — |
| Mean probe message bytes | 1,392,800 | 1,338,863 | — |
| Mean probe environment steps | 10.000 | 9.611 | — |

For each cell, two-seed CVaR90 is simply the larger of its two errors; its mean
improves only 0.3665%. The pooled 144-row tail improves 1.1451%, while mean
normalized AUC worsens by 9.1312%. Thus even ignoring all learning degradation
from probe cost, there is no broad multi-metric adaptation ceiling.

## Gate audit

The old no-harm gate `controller/fallback <= 1.05` is necessary but not
nontrivial. A controller that always returns the fallback has ratio exactly
one and passes it without adapting. Any future design therefore needs a
separate nontriviality gate.

Before any nonlinear GPU pilot, the static fixed-q oracle after full probe
charging must show both:

1. at least 5% aggregate geometric improvement over the task×budget fallback;
2. strict directional improvement in at least 60% of registered cells.

EXP-017A supplies 0.3846% and 29.17%, so both gates fail. Under the requested
decision rule, **EXP-017B is permanently stopped**. Its thresholds, task set,
or cells must not be retrospectively changed to rescue it.

Machine-readable values are in `t020_adaptation_value_ceiling.json`.
