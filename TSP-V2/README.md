# TSP-V2: Lyapunov Learning-Progress Control

This directory is an independent successor to the frozen `TSP/` manuscript.
No file in `TSP/` is overwritten or reinterpreted.

The upgraded research question is:

> Under correlated Markov rollouts and simultaneous communication and
> environment budgets, can a learner use fully charged short training branches
> to estimate the task-dependent Lyapunov contraction of each participation
> level and select the action that minimizes certified terminal learning risk?

The old controller estimates dependence and assumes that participation affects
learning only through effective variance and the remaining update horizon.
TSP-V2 additionally estimates action-specific learning progress.  This is the
smallest principled change that can represent an interior optimum such as
`q=2` in MaMuJoCo while retaining the dependence and resource accounting that
worked on PettingZoo.

## Stage policy

1. Prove and unit-test the finite-catalogue learning-progress selector on CPU.
2. Run an outcome-isolated GPU interface test using new probe seeds only.
3. Freeze the controller, budgets, gates, and fresh development seeds before
   measuring policy return.
4. Run confirmation on disjoint seeds only after every development gate passes.

The existing PettingZoo, SMACv2, and MaMuJoCo outcomes are development
information for TSP-V2.  They are never relabelled as TSP-V2 confirmation.

