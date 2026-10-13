# ICML 2027 learning-value upgrade: persistent plan

## Material Passport

- Origin: academic-research-suite / experiment-agent, authorized autonomous implementation.
- Date: 2026-10-13.
- Status: CPU development and independent byte replay passed; fresh confirmation lock ready.

## Decisions

1. Preserve TSP/, TSP-V2/, TSP-V3/, all historical gates and results.
2. Use the full q={1,2,4,8} catalogue; forbid task-specific q removal.
3. Avoid V2's clipped one-block reward, nonshrinking sliding-window bonus, and
   repeated validation tax. Pay one independent calibration stream.
4. Count both messages and TOTAL actor transitions, with parallel ticks logged
   separately. Never label parallel-clock ticks as total environment work.
5. First establish an observable score/true-risk interface in a tractable
   Markov TD model. The selector cannot access theta or rho.
6. Use known-mixing Gaussian whitening with exact t/chi-square intervals;
   do not label a normal approximation as a finite-sample certificate.
7. Freeze the CPU design in a separate commit before calibration outcomes.
8. If development passes, freeze a separate confirmation lock with fresh seeds.
9. After CPU confirmation, qualify the actual MARL learning-response interface
   before any multi-environment performance run. A controlled CPU pass is not a
   nonlinear MARL success claim.

## Required CPU outputs

Full-context risk/decision tables, coverage and interval widths, oracle regret,
interior-q selection, strong fixed comparators, ablations, source/config hashes,
an independent byte replay, and a complete decision preserving every cell.

## GPU evidence requirements

One frozen controller and complete q catalogue; discrete coordination and
continuous control; native independent, intermediate, and shared coupling;
full fixed catalogue plus correlation-only, myopic, and a specified adaptive
batch baseline; return versus total actor transitions and messages; separate
wall time and calibration cost; development before fresh confirmation.

No new manuscript claim is admitted until it is supported at the declared scope.

## Completed on 2026-10-13

- Preregistration source: f584d76.
- Ten targeted tests passed.
- Development: 360 contexts, 16 fresh seeds, 46,080 method rows; all gates pass.
- Five scientific files reproduce byte for byte.
- A separate 64-seed confirmation lock retains the original implementation,
  population, complete catalogue, and all gates.
- The myopic comparator is a one-step-then-commit ablation. A strong repeatedly
  replanning adaptive baseline is still required before an ICML algorithm claim.
