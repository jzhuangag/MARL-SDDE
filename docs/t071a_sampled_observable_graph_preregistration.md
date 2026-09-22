# T-071A sampled observable nonstationary graph preregistration

## Purpose

T-071A asks whether an implementable, low-complexity controller can retain a
material fraction of the exact dynamic collaboration value established by
T-070A. It is a sampled local-CPU pilot and cannot become formal evidence.

The configuration, 32 pilot seeds, source hash, comparators, and gates are
frozen before any T-071A trajectory is generated.

Frozen configuration SHA-256:
`29cf65e9267720d27b9dfd10df9ed0845e51d75f8adc1800010e05280cbdede8`.

## Observable controller

At each of six decision blocks, every agent spends four transitions on a short
TD-observation fingerprint. The first two observations form a selection
fingerprint and the last two form an independent validation fingerprint. None
of these four transitions is a learning update.

For recipient \(i\), the server scores seven candidates: local, or one of three
delayed donors with weight 0.5 or 1.0. Selection minimizes squared distance to
the selection fingerprint. The selected transfer is accepted only if its mean
validation-half squared loss is no larger than that of the same-data local
shadow. Otherwise recipient \(i\) resets to its shadow.

This scan requires exactly \(6\times4\times7=168\) scalar candidate scores per
endpoint. It computes no Hessian, covariance matrix, inverse, or preconditioner.
The step size remains fixed at the stability-screened T-070A value so this pilot
isolates graph-identification value. Joint online step-size control is not
authorized until this gate passes.

## Stochastic model and comparison

Observations follow a vector Gaussian AR(1) process with the frozen temporal
and cross-agent correlations and the public piecewise target schedules. All
policies use common random numbers.

The primary baseline is the frozen T-070A cellwise best static personalized
graph. It uses all 240 transitions for learning and pays no sensing cost. Other
comparators are no-probe local learning, the fully charged local shadow, full
sharing, and a fully charged pathwise clairvoyant dynamic oracle.

The primary risk is personalized parameter mean-square error averaged over all
24 blocks. Terminal risk is descriptive.

## Frozen decision

All mandatory gates S1--S12 are specified in the JSON preregistration. Any
failure stops formal seeds and nonlinear/GPU experiments under this experiment
identifier. Thresholds, seeds, schedules, or comparison rules must not be
changed after observing results.
