# Validation report: EXP-007A correlated delayed linear TD

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Experiment result

- **ID**: EXP-007A-linear-td-correlation
- **Type**: preregistered linear TD(0) simulation with paired Markov paths
- **Status**: completed
- **Primary duration**: approximately 673 seconds
- **Reproduction duration**: approximately 673 seconds
- **Executed source commit**: `63925d8`
- **Primary command**:
  `python run_linear_td_correlation.py --output-dir results/linear_td_correlation --num-seeds 32 --bootstrap-replications 2000 --workers 4`
- **Exit code**: 0
- **Anomalies**: none

All 134,784 registered seed/cell/budget rows completed and were finite. Every
update was charged under the message-equivalent cost \(4+q\).

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| Independent speedup | median \(N_{\rm eff}(32)=30.996\) | PASS |
| Correlation saturation | at \(\rho=0.9\), median \(N_{\rm eff}(32)=1.111\) | PASS |
| Participation transition | \(q^\star(0)=16,\ q^\star(0.9)=1\) | PASS |
| Material resource value | endpoint ratios 0.233 and 0.319 | PASS |
| Delay/step-size consistency | 108/108 matched cells satisfy \(\eta_{32}\le\eta_0\) | PASS |
| Accounting/numerical validity | 134,784/134,784 valid rows | PASS |
| Overall | all six registered gates | **PASS** |

The paired-bootstrap 95% intervals for the two endpoint ratios were
[0.144, 0.372] under independent trajectories and [0.219, 0.462] under
\(\rho=0.9\). Both effects are well separated from one.

## Effective participation

Median \(N_{\rm eff}(32,\rho)\) changed as follows:

| \(\rho\) | 0 | 0.25 | 0.50 | 0.75 | 0.90 | 1.00 |
|---:|---:|---:|---:|---:|---:|---:|
| \(N_{\rm eff}\) | 30.996 | 3.555 | 1.932 | 1.305 | 1.111 | 1.000 |

The seed-level 2.5%--97.5% empirical ranges were [27.64, 38.18] at
\(\rho=0\) and [1.054, 1.165] at \(\rho=0.9\).

For exchangeable agent noise, direct expansion gives

\[
\Omega(q)=
\Omega_{\rm off}
+\frac{\Omega_{\rm diag}-\Omega_{\rm off}}{q}.
\]

Post-registered diagnostic fits of mean trace LRV against \(1/q\) had
\(R^2\) values 0.99998, 0.99993, 0.99990, 0.99934, and 0.99943 for
\(\rho=0,0.25,0.5,0.75,0.9\), respectively. At \(\rho=1\), all agent
directions are identical and \(N_{\rm eff}=1\) exactly. This is strong
mechanistic agreement rather than a visual trend alone.

## Finite-budget participation phase

The oracle counts were identical across the three registered delay levels:

| Budget | \(\rho=0\) | 0.25 | 0.50 | 0.75 | 0.90 | 1.00 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 8 | 4 | 2 | 1 | 1 | 1 |
| 8,000 | 32 | 2 | 2 | 1 | 1 | 1 |
| 32,000 | 16 | 8 | 2 | 1 | 1 | 1 |

This is the desired resource phase transition: when trajectories are
independent, extra messages buy variance reduction and a large group is
optimal; when common Markov transitions dominate, that benefit saturates and
the same budget is better spent on more updates with one agent.

## Selection-bias diagnostic

The registered oracle surface selects \(q,\eta\) and reports its error on the
same 32 seeds. This can optimistically bias an oracle comparison. A
post-registered two-fold check therefore selected actions on seeds 0--15 and
evaluated seeds 16--31, then reversed the halves.

Both training halves selected \(q=16\) for \(\rho=0\) and \(q=1\) for
\(\rho=0.9\). Held-out endpoint ratios were:

- first-half selection, second-half evaluation: 0.354 and 0.471;
- second-half selection, first-half evaluation: 0.197 and 0.228.

The transition is not explained by same-sample oracle selection. These
diagnostics do not replace a fresh-seed confirmatory experiment.

## Critical delay audit

The formal delay gate passes, but it is not positive evidence that delay
adaptation works. In all 108 matched cells, the best step size at \(D=32\)
equaled the best step size at \(D=0\); the non-increase gate passes entirely
through equality. Oracle \(q\) was also identical at \(D=0,8,32\).

Moreover, oracle-error ratios \(D=32/D=0\) were close to one, and sometimes
below one because finite-path stochastic variation can make stale averaging
appear favorable. The registered TD regime therefore does not activate a
material delay penalty.

Consequently:

- EXP-007A validates correlation-limited effective participation and the
  communication-budget phase transition;
- it does **not** validate a delay-adaptive algorithm or an empirical SDDE
  claim;
- a separate delay-stress experiment must use a regime where the
  Lyapunov--Krasovskii stability boundary is measurably active.

## Reproducibility

- **Verdict**: REPRODUCIBLE
- **Method**: deterministic same-seed full rerun in an isolated output path.

All 13 artifacts matched byte-for-byte by SHA-256, including five MRP/system
tables, five result tables, `summary.json`, and three figures. The complete
experiment suite passed 35 tests; the only warning was an unrelated
`pyreadline` deprecation notice.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Endpoint effects persist in both held-out half-splits; full phase tables are reported. |
| Ecological fallacy | NOTE | Effective participation is an aggregate variance quantity, not an individual-agent trait. |
| Berkson's paradox | NOTE | No seed, \(q\), step size, delay, or budget cell was filtered. |
| Collider bias | NOTE | The registered oracle selection is acknowledged; held-out halves audit selection bias. |
| Base-rate neglect | NOTE | All six correlation levels retain equal registered representation. |
| Regression to the mean | NOTE | Seeds and thresholds were fixed before the primary run. |
| Survivorship bias | NOTE | All registered rows completed and were finite. |
| Look-elsewhere effect | NOTE | Smoke results were excluded; the six primary gates were preregistered. |
| Garden of forking paths | NOTE | The MRP, grid, seeds, metrics, and gates were unchanged after smoke. |
| Correlation versus causation | NOTE | Shared-transition probability is experimentally intervened on; deep-MARL generalization is not claimed. |
| Reverse causality | NOTE | Participation policies are evaluated on common exogenous paths; outcomes do not set \(\rho\). |

## Validated decision

Promote the following to the paper's main line:

1. cross-agent correlation replaces nominal \(N\) by an effective
   participation number;
2. the exchangeable LRV law yields saturation beyond linear speedup;
3. under communication budget, saturation creates a correlation-dependent
   optimal agent count.

Keep SDDE/Lyapunov--Krasovskii analysis as the delay/stability layer, but do
not yet claim experimental delay adaptivity. The next confirmation should use
fresh seeds, an information-matched fixed action protocol, and a deliberately
active delay-stability regime.
