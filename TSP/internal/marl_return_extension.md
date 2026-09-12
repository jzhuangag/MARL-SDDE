# TSP policy-learning return extension

## Publication role

The existing convergence figure is a fixed-policy evaluation experiment: its
discounted-return **estimation error** should decrease.  It must not be
relabelled as episodic return.  A genuine increasing-return figure requires a
policy-learning experiment and is treated as an external CTDE validation of
the same critic-control mechanism.

## One model, two meanings of agent

The cooperative game contains `n_game` strategic agents.  During training,
each rollout worker runs one complete joint policy

`pi_theta(a|o) = product_i pi_{theta_i}(a_i|o_i)`

on an independent or marginal-preserving coupled copy of the game.  The TSP
participation variable `q` counts these rollout workers, not strategic game
agents.  Their samples update a centralized critic; decentralized actors use
only local observations at execution.  Consequently, changing `q`, spacing,
or critic step size changes training only and does not add execution-time
communication.

## Theory interface

During actor epoch `r`, the centralized critic tracks the projected value
fixed point `w_r^star`.  The actor update moves that target between epochs.
The moving-policy tracking theorem in `main.tex` gives

`Z_n <= cbar^n Z_0 + ((1+chi)d + (1+1/chi)v^2)/(1-cbar)`,

where `v` bounds the target motion and `(c,d)` come from the same delayed
Markov Lyapunov certificate used to choose the critic resource action.  The
bounded-feature corollary transfers `Z_n` to one-step advantage-estimation
error.  This is a rigorous critic-tracking bridge, not a claim that the affine
theorem proves global convergence of a nonlinear actor.

## Benchmark construction

- Primary public task: PettingZoo MPE `simple_spread`, trained with MAPPO.
- Secondary task after the primary gate: `simple_reference` or a second
  cooperative benchmark supported by the pinned HARL stack.
- Strategic architecture: parameter-shared actors and centralized critic
  during training; decentralized deterministic actors during evaluation.
- Rollout-worker dependence: independent reset streams and common-random-number
  reset streams.  Across the frozen seed registry, independent workers receive
  cyclic permutations of the same seed multiset whereas shared workers receive
  the same seed in each run.  Thus each worker has exactly the same empirical
  marginal seed law in both regimes; only the joint coupling changes.  A
  runtime observation/return-marginal audit remains mandatory before any
  performance run.
- In the shared regime, each strategic actor samples its discrete actions by
  inverse CDF from one public uniform across rollout workers.  Every worker
  retains its correct categorical action marginal, while the exploration
  innovation is correlated across workers.
- Controlled action: rollout participation `q` and critic step `eta`; spacing
  is held at `b=1` in this nonlinear transfer unless a pause/skip implementation
  can charge every physical transition exactly.
- Delay: initially zero.  A delayed-gradient condition is added only after a
  FIFO implementation passes ownership and accounting tests.

## Equal-resource comparison

One learner iteration of rollout length `L` costs `L` parallel environment
ticks and `q L` actor transitions/messages.  For budgets `(B_msg,B_env)`, the
number of learner iterations is

`N(q) = min(floor(B_msg/(h+q L)), floor(B_env/L))`,

where `h` is the registered server overhead per rollout packet.

Every method receives the same two budget limits.  Curves use the fraction of
that method's feasible update horizon as the common horizontal coordinate;
the raw cumulative message and environment charges are logged separately so
the inactive clock is not silently relabeled as exhausted.
Evaluation episodes are charged separately and identically, then excluded
from the training-budget axis.  Wall-clock time is reported as a secondary
systems metric, not the primary learning objective.

## Comparators and separation discipline

Development seeds select one strong fixed `(q, eta)` over the full registered
regime mixture.  That pair is frozen before confirmation.  Confirmation uses
new seeds and compares:

1. Lyapunov-selected `(q, eta)`;
2. the development-selected strong fixed pair;
3. all-worker MAPPO;
4. one-worker MAPPO;
5. a per-regime oracle shown only as an upper-bound diagnostic.

The primary metric is area under deterministic team-return versus charged
budget fraction, normalized so larger is better.  Secondary metrics are final
return, critic loss, advantage error, selected actions, messages, actor
transitions, wall-clock time, and controller overhead.

## Gates before a confirmatory GPU run

1. The pinned upstream baseline produces finite deterministic evaluation
   returns and a non-flat learning curve.
2. The owned coupling wrapper preserves every rollout worker's marginal law.
3. Exact accounting holds for all `q` and both budgets.
4. On development seeds, the per-regime fixed-pair oracle has at least 5%
   aggregate return-AUC headroom over the strong global fixed pair and improves
   both dependence regimes.
5. The Lyapunov action agrees with the oracle direction in at least 75% of
   development cells without using returns as an input.
6. Controller overhead is below 5% of training time.
7. Runner, analyzer, config, upstream commit, seeds, and environment lock are
   frozen before confirmation.

The 20,000-transition local continuous-action MAPPO smoke on seed 91001 closed
only the logging interface.  It was too short to learn: deterministic
evaluation return moved from about -109 to -132.  It is not a scientific
result and is not eligible for a manuscript figure.

A second, discrete-action seed-91002 smoke ran for 100,000 actor transitions
with evaluations every 10,000 transitions.  Return improved sharply from
-168.18 to -104.12 by the second evaluation but remained noisy thereafter and
ended at -119.06.  Throughput was about 62 transitions/s on the local CPU.
This confirms that a single short run cannot provide the requested smooth
learning curve; millions of transitions and multiple seeds are required.

The owned fixed-action bridge and exact dual-budget accounting are now
implemented in `experiments/run_mappo_return_bridge.py`.  A 17-seed, eight-worker
reset audit verified exact equality of every worker's reset-observation hash
multiset across independent and shared coupling; shared runs had one unique
reset per run and independent runs had eight.  The audit is recorded in
`internal/marl_coupling_audit.json`.  For discrete policies, a public-uniform
inverse-CDF sampler preserves each categorical action marginal algebraically
while correlating exploration across rollout workers.  Performance headroom
remains to be tested on the frozen development grid.

The current outcome-free development-grid SHA-256 is
`3833280e6a49d395be2b2c73791b7cb0b94700c5b76dfa23309ff27e689475b2`; the
fixed-action runner SHA-256 is
`77ba365b3cba64c6befaf4176b3d0b8d8034df2cd56e84d680cc744e10f0f9cc`.
The gate analyzer SHA-256 is
`c38cca687103f9d8d29b662018f997c7e989e15645b3fcf12efed17f521e4390`,
and the Slurm payload SHA-256 is
`ff632f8bc3f1f4fcf13dac21e3d984b7b3ce01141fdda467dee34706252859be`.

## Compute decision

A credible MPE curve requires millions of actor transitions per seed.  At the
measured CPU rate, one million transitions takes about 4.5 hours before a
multi-method, multi-regime, multi-seed comparison.  Local CPU remains
appropriate for wrapper tests and short smoke runs; development and
confirmation should use HPC4 GPUs after the seven gates above are encoded in a
preregistration.

## TSP-MARL-DEV-001 outcome

The frozen 16-cell, one-seed development scan completed on HPC4 with every allocation at exit code `0:0`.
All runs were finite, exactly charged against both budgets, checksum-valid, and based on clean pinned source trees.
The global strong fixed action was `(q,eta)=(8,0.0005)`.
The per-regime oracle used the same action for independent streams and `(q,eta)=(1,0.0005)` for shared streams.
The aggregate oracle AUC headroom was `5.032546%`, which passed the registered 5% threshold, but strict oracle improvement occurred in only the shared regime.
The frozen `oracle_improves_each_coupling_regime` gate therefore failed, and the registered stopping rule prohibits controller design and confirmation from this experiment.
The development curve is retained only in the internal scratch record and is not a manuscript figure.
Full validation and provenance are recorded in `marl_return_d0_validation.md`.

## TSP-MARL-DEV-002 outcome

A separately preregistered Pareto question retained the global strong fixed
`q=8` action under independent rollout innovations and asked for improvement
only in the shared regime where DEV-001 identified an internal optimum at
`q=1`.  A fully charged, non-learning 128-block critic-residual probe and the
Lyapunov variance--message score selected `q=8` for all four independent seeds
and `q=1` for all four shared seeds without observing evaluation return.

All 24 registered MAPPO cells completed.  Relative to fixed `q=8`, the
controller's return AUC changed by -0.4134% in the independent regime, +6.3388%
in the shared regime, and +2.9627% in the equal-weight mixture.  These values
passed the frozen -2%, +5%, and +2.5% development thresholds, respectively;
all selection, exact-accounting, probe-cost, overhead, coupling, and source
integrity gates also passed.  DEV-002 therefore authorizes only a separate
confirmation preregistration with new seeds.  Its development curve is not a
manuscript figure.  Full validation is recorded in
`marl_probe_commit_d1_validation.md`.
