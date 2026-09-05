# RCR-H1 preregistration: analytic survival scan

## Frozen scope

RCR-H1 is an outcome-free CPU feasibility scan for the proposed
reuse--correct--refresh controller.  It is not an RL experiment, formal
evidence, or permission to run a GPU benchmark.  The scan is frozen before any
scientific outcome is generated.

- configuration: `docs/rcr_headroom_config.json`;
- configuration SHA-256:
  `875E692868E6696E5C4DD13C029A3E5E88914BB61E164A422243C9EF4C7E9D36`;
- 2,048 stochastic scenarios and 393,216 causal packet events;
- 32 fresh analytic seeds beginning at 73001;
- 64 aggregate cells after seed aggregation;
- local CPU only; no HARL trajectory, HPC4, GPU, or prior formal data.

The grid crosses 4/12 agents, effective batch sizes 4/16, Markov persistence
0.8/0.95, stationary/bursty/rotating/mixed certificate paths, and refresh
budgets 1/8 and 1/4.  Sensitivity is linked to the square root of the divergence
proxy rather than independently tuned.  The controller uses
`V=sqrt(horizon)` and no outcome-selected hyperparameter.

## Comparators

The adaptive correction QP is compared with no correction, full correction,
and the globally optimal fixed per-agent tempering vector obtained from the
entire analytic path.  The refresh controller is compared with the best phase
of a fixed-period schedule after granting both methods the same pointwise
optimal correction rule.  A clairvoyant same-count refresh oracle is used only
to measure attainable gain, never as a deployable baseline.

## Frozen mandatory gates

All ten gates must pass:

1. every result is finite and structurally valid;
2. adaptive correction / best static-vector geometric risk ratio is at most
   0.95;
3. causal RCR / best fixed-period geometric risk ratio is at most 0.95;
4. causal RCR strictly improves at least 70% of nonstationary cells;
5. median captured oracle gain is at least 50%;
6. stationary causal/fixed risk ratio is at most 1.01;
7. causal/fixed risk ratio is at most 0.98 separately at both persistence
   values;
8. no refresh-budget overshoot;
9. scalar solver terminates within its registered 128-iteration bound;
10. any failed gate stops confirmation, standard RL, formal, and GPU work.

No threshold, seed, profile, or comparator may be changed after observing the
scan.  A positive scan establishes only analytic headroom.  The normalized
geometric-path certificate and the Markov/potential learning theorem must also
pass their separate proof audit before a standard MARL experiment is designed.
