# T-059 theorem-aligned MinAtar bridge

## Status

T-059 is an outcome-free implementation and environment audit.  It does not
generate a fixed-q comparison, choose a benchmark after seeing a comparison,
allocate pilot or formal seeds, or authorize a controller, GPU, HPC4, or
`/project` write.  The next scientific result requires a separate immutable
preregistration commit.

## Scientific bridge

The primary learner is a frozen nonlinear state encoder followed by a learned
linear temporal-difference (TD) head.  For a fixed encoder
`phi_psi(s,a_prev)`, regularization `lambda`, and discount `gamma`, the delayed
head update is

`w[t+1] = w[t] + eta * mean_i(phi_i * (r_i + gamma * phi'_i^T w[t-D] - phi_i^T w[t-D]) - lambda * w[t-D])`.

Conditional on the frozen encoder, this is an affine Markov stochastic
approximation in `w`.  Therefore the T-048/T-056 finite delayed risk machinery
applies to the trainable head without claiming convergence of a jointly
trained neural network.  The bridge tests a rich nonlinear representation and
a standard reinforcement-learning environment while preserving the theorem's
actual parameter scope.

The encoder has eight fixed random 3-by-3 convolutional filters, ReLU, 2-by-2
adaptive average pooling, a fixed random 32-dimensional projection, tanh, a
constant coordinate, and unit-norm output.  Previous executed action is
included as a six-dimensional one-hot input because MinAtar's default sticky
action mechanism makes it part of the Markov state.  The online participation
arithmetic remains `O(qd)` and the delayed head memory is `O((D+1)d)`; no
Hessian, covariance inverse, preconditioner, or online neural backpropagation
is used.

## Frozen environment family for the prospective design

- official `MinAtar==1.0.15` core `Environment` API;
- Asterix, Breakout, and Seaquest, all retained as one benchmark family;
- 10-by-10 binary observations and the public full six-action set;
- uniform fixed behavior policy, sticky-action probability 0.1;
- difficulty ramping disabled to obtain a time-homogeneous fixed-policy task;
- terminal TD successor is zero and the next sample begins from the public
  environment reset distribution;
- discount 0.97 and positive head regularization 0.05 are prospective defaults.

Disabling difficulty ramping is a stationarity definition, not an
outcome-selected task modification.  The game mechanics, observation law,
rewards, terminal events, spawning randomness, and sticky actions remain the
official implementation.

## Marginal-preserving dependence

Each master seed generates one common and sixteen independent private complete
MinAtar streams.  Actor `i` uses the common stream with probability
`sqrt(rho)` and its own private stream otherwise.  Consequently every actor
has the unchanged fixed-policy MinAtar marginal law and two actors share the
complete stream with probability `rho`.  Prefixes implement q in `{1,4,16}`
under common random numbers.  Probe streams, if later authorized, must be
independent of all learning streams.

## Prospective fixed-q value gate

Before any adaptive controller is implemented, a separately preregistered CPU
pilot must compare only q in `{1,4,16}` over correlations
`{0,.1,.3,.5,.7,.9,1}`, delays `{0,8}`, and overheads `{8,32}`.  It must use
independent high-precision reference TD moments and report the feature-
covariance-weighted error of the half-tail Polyak--Ruppert head average.  Every
actor transition, message, delay reserve, and later probe must be charged.

The optimistic cellwise fixed-q oracle must improve over a pilot-selected
task-by-overhead strong fixed q by at least 5% in aggregate and strictly
improve at least 60% of registered cells.  Reference stability, independent
reference agreement, finite results, full cost accounting, taskwise
directionality, and exact rerun are mandatory.  Failing the value gate stops
this MinAtar controller route; games or thresholds may not be replaced after
viewing the outcomes.

## Compute decision

The official environments deliver roughly 56,000--99,000 transitions per
second per CPU process in the local interface audit.  Encoder and moment
calibration therefore remain CPU tasks.  GPU execution is not justified until
the fixed-q value gate passes and a later multi-seed learned-controller
workload is statically shown to benefit from it.

