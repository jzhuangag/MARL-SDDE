# EXP-013B: preregistered realizable nonlinear Markov-TD confirmation

## Material Passport

- Artifact: confirmatory experiment protocol
- Status at registration: no confirmatory outcomes observed
- Registration commit: to be recorded before execution
- Pilot provenance: EXP-013A, explicitly excluded from confirmatory evidence
- Hardware: local CPU

## Confirmatory question

Under a fixed communication budget, does the resource-optimal participation
level of delayed nonlinear Markov TD move from many agents under independent
sampling to a small subset under strongly shared sampling?

## Frozen data-generating process

The state process, 16-by-16 tanh student, discount, delay queue, learning rate,
and cost model are exactly those in the realizable EXP-013A diagnostic:

- four-dimensional stationary Gaussian AR(1), \(\lambda=.8\);
- fixed teacher network of the same architecture as the student;
- reward
  \(R_t=V^\star(S_t)-.9V^\star(S_{t+1})+\xi_t\),
  with unit-variance Gaussian reward noise;
- hidden common/private source selection with pair-sharing probability
  \(\rho\);
- semi-gradient TD(0), learning rate .03;
- message budget \(B=64{,}000\), server overhead \(h=64\), and
  \(T_q=\lfloor B/(h+q)\rfloor\) updates;
- \(q\in\{1,4,16,32\}\), \(D\in\{0,8\}\), and
  \(\rho\in\{0,.25,.5,.9\}\).

The confirmatory run uses 32 fresh seeds beginning at 20270701.  The teacher
seed remains fixed at 20270522 because it defines the task, not a replication.
No learning rate, budget, architecture, noise variance, or gate may change
after execution begins.

## Primary endpoint and paired contrasts

The endpoint is terminal validation MSE to the known teacher on 4,096 fresh
stationary states.  For seed \(s\), define the delay-averaged paired log
contrasts

\[
L_{0,s}
=\frac12\sum_{D\in\{0,8\}}
\log\frac{\operatorname{MSE}_{s,0,D,32}}
{\operatorname{MSE}_{s,0,D,1}},
\]

\[
L_{.9,s}
=\frac12\sum_{D\in\{0,8\}}
\log\frac{\operatorname{MSE}_{s,.9,D,4}}
{\operatorname{MSE}_{s,.9,D,32}}.
\]

Ratios are \(\exp(\bar L)\).  One-sided 99% upper confidence limits resample
the 32 seeds as clusters with 20,000 bootstrap draws and RNG seed 20270702.
Both delays remain together within every resampled seed.

For each seed and correlation, the resource oracle minimizes mean log MSE
over the two delays, with ties resolved toward smaller \(q\).  Report all
per-seed choices and their medians.

## Frozen gates

The experiment passes only if all five gates hold:

1. every run and terminal MSE is finite;
2. the 99% upper limit for \(\exp(\bar L_0)\) is below .70;
3. the 99% upper limit for \(\exp(\bar L_{.9})\) is below .85;
4. the median oracle choice is at least 16 for \(\rho=0\) and at most 4 for
   \(\rho=.9\);
5. the geometric MSE ratio is below .80 for \(q=32\) versus \(q=1\) at each
   individual delay when \(\rho=0\), and below .90 for \(q=4\) versus \(q=32\)
   at each individual delay when \(\rho=.9\).

Intermediate-correlation choices, exact effect sizes, loss maxima, and
delay-specific curves are descriptive unless covered by gate 5.  Failed gates
remain failures; no alternative aggregation or seed exclusion is permitted.

## Interpretation boundary

Passing supports a controlled nonlinear resource-allocation mechanism and
shows that it is not an artifact of linear features.  It does not establish
global neural-TD convergence, performance in a standard MARL benchmark, or
online estimation of \(\rho\).  Theorem 9 supplies the fixed-parameter
nonlinear variance identity; the affine Markov-TD theorem remains the rigorous
finite-time convergence result.

