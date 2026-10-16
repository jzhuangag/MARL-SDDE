# T-060A MinAtar fixed-q CPU feasibility pilot preregistration

## Purpose and scope

T-060A asks a single prospective question: does the stationary
correlation-participation phase retain at least 5% held-out learning value in a
standard high-dimensional reinforcement-learning environment when the
nonlinear representation is frozen and the trainable TD head remains within
the proved affine Markov-SA class?

This is a feasibility pilot, not paper evidence.  It contains no adaptive
controller, fingerprint probe, learned task selection, formal seed, GPU job,
or HPC4 write.  Passing permits a separate controller preregistration with new
seeds.  Failure permanently stops this MinAtar controller route under the
registered tasks and thresholds.

## Environment and learner

The task family contains Asterix, Breakout, and Seaquest from the official
`MinAtar==1.0.15` core API.  All three are retained.  A fixed policy samples
uniformly from the full six-action set used by the original benchmark.
Sticky-action probability is 0.1.  Difficulty ramping is disabled before any
outcome to define a time-homogeneous Markov reward process.  Previous executed
action is included in the encoder input, so sticky dynamics do not create an
unobserved state variable.

The frozen nonlinear encoder is the T-059 random convolution/ReLU/pooling/tanh
map with task-specific public seeds.  Only its 33-dimensional regularized TD
head is learned.  The update uses discount 0.97, regularization 0.08, a frozen
spectral step-size rule, delays 0 and 8, and half-tail Polyak--Ruppert (PR)
averaging.  The primary endpoint is prediction error relative to two
independent high-precision reference moment banks, weighted by their pooled
feature covariance.

## Dependence, actions, and costs

For every master seed and task, the runner generates one common and sixteen
private complete streams.  Actor `i` uses the common stream with probability
`sqrt(rho)`, otherwise its private stream.  Thus each actor keeps the official
fixed-policy marginal law and a pair shares a stream with probability `rho`.
The registered grid is q in `{1,4,16}`, rho in
`{0,.1,.3,.5,.7,.9,1}`, overhead in `{8,32}`, and delay in `{0,8}`.

For each overhead, the common message budget is `(overhead+16)*8192` and the
common environment budget is `16*8192` actor transitions.  The usable horizon
is the smaller integer budget horizon minus the registered delay.  Therefore
every larger-q sample, server overhead, and delay reserve is charged.  No
probe cost exists because T-060A contains no probe.

## Split-sample value ceiling

The first sixteen pilot seeds select a task-by-overhead strong fixed q and a
cellwise fixed-q oracle.  Those choices are frozen and evaluated only on the
disjoint final sixteen pilot seeds.  This avoids evaluating a cellwise oracle
on the same noise used to select it.  No pilot seed can later become a
controller or formal seed.

The controller route is authorized only if every V1--V9 gate passes:

1. exactly 8,064 unique endpoints and all registered cells;
2. finite positive risks and exact message/environment/delay accounting;
3. two-bank reference agreement, positive symmetric drift, condition number at
   most 500, and lifted delayed spectral radius below one for every task;
4. held-out oracle/strong-fixed geometric risk ratio at most 0.95;
5. strict held-out improvement in at least 60% of cells;
6. held-out oracle/strong ratio at most 0.98 on each task;
7. nonincreasing selected q with rho on at least 75% of task-overhead-delay
   paths;
8. exact selection/validation seed isolation;
9. no controller field or controller outcome in the pilot.

An exact clean rerun is required before a pass is acted upon.  Reproduction is
an additional authorization condition even though it is assessed only after
the immutable pilot run.

## Frozen decision rule

Any failed scientific, validity, reference, cost, or reproduction condition
sets `controller_pilot_authorized=false`.  Games, seeds, horizons, features,
thresholds, split, risks, and task aggregation cannot be modified after the
preregistration commit to rescue T-060A.  A passed pilot permits only a new
CPU controller design and power audit; it does not itself justify a GPU run.
