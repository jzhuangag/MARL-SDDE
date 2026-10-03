# EXP-017A read-only code and research audit

## Existing-code verdict

The existing `experiments/nonlinear_markov_td` code is useful mechanism
scaffolding but is not a standard nonlinear benchmark. EXP-013 uses Gaussian
or synthetic realizable teachers; EXP-014B uses a two-state binary MRP and its
recorded pilot failed because the conservative controller fell back on every
block. Those outcomes remain negative evidence and are not reused or relabeled.

Reusable components are limited to plain neural semi-gradient TD, pathwise
dual-budget accounting, delayed gradient queues, scalar dependence summaries,
and complete `(q,b)` traces. The new runner does not edit any old runner or
artifact.

## Benchmark choice

Gymnasium provides a maintained standardized RL interface. `CartPole-v1` has
a four-dimensional continuous observation and two actions; `Acrobot-v1` has a
six-dimensional continuous observation and three actions. Both are standard
Classic Control environments and have appeared in neural-TD evaluation work.
Their fixed behavior policies make prediction and Bellman error directly
measurable without adding actor--critic scope.

## Prior-art boundary

- DASA (Dal Fabbro et al., arXiv:2403.17247) gives delay-adaptive multi-agent
  stochastic approximation with independent agent Markov chains and N-fold
  speedup.
- AsyncMATD (Dal Fabbro et al., arXiv:2407.20441) gives asynchronous delayed
  multi-agent linear TD and N-fold speedup under independent observation
  processes.
- Neural TD convergence and Gym experiments already exist; nonlinear function
  approximation by itself is not novel.

Therefore EXP-017A can test external nonlinear breadth only. It cannot support
unrestricted unknown-mixing adaptation, global occupation optimality, or
general nonlinear MARL. The proposed distinction remains correlation-limited
participation under known mixing, dual resources, and heterogeneous delay.

## Primary sources checked

1. Gymnasium CartPole documentation:
   https://gymnasium.farama.org/environments/classic_control/cart_pole/
2. Gymnasium Acrobot documentation:
   https://gymnasium.farama.org/environments/classic_control/acrobot/
3. Towers et al., *Gymnasium: A Standard Interface for Reinforcement Learning
   Environments*, arXiv:2407.17032.
4. Dal Fabbro et al., *DASA*, arXiv:2403.17247.
5. Dal Fabbro et al., *Finite-Time Analysis of Asynchronous Multi-Agent TD
   Learning*, arXiv:2407.20441.
6. Tian, Paschalidis, and Olshevsky, *On the Performance of Temporal Difference
   Learning With Neural Networks*, arXiv:2312.05397.

The requested `academic-research-suite` skill was not installed in this Codex
workspace. This audit therefore used repository records and the primary
official/arXiv sources above; search snippets and secondary summaries were not
used as theorem evidence.
