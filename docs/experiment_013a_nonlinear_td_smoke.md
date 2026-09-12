# EXP-013A: nonlinear Markov-TD CPU feasibility smoke

## Material Passport

- Artifact: feasibility plan
- Evidence status: implementation smoke only; not preregistered scientific
  evidence
- Hardware: local CPU
- External environment dependency: none beyond the installed PyTorch

## Question

Does correlation-limited participation remain visible when the value function
is represented by a nonlinear neural network rather than a linear feature
map?

## Model

Each hidden source follows a stationary four-dimensional Gaussian
autoregression

\[
S_{t+1}=\lambda S_t+\sqrt{1-\lambda^2}\,\varepsilon_t,
\qquad \lambda=.8.
\]

There is one common path and 32 private paths.  Agent \(i\) selects the common
transition with probability \(\sqrt\rho\) and otherwise selects its private
transition, so two agents share a transition with probability \(\rho\).

A fixed teacher \(V^\star\) defines

\[
R_t=V^\star(S_t)-\gamma V^\star(S_{t+1}).
\]

Therefore \(V^\star\) is the exact discounted value function.  A
two-hidden-layer tanh network is trained by semi-gradient TD(0).  Aggregate
gradients are queued for \(D\) server steps before application.

## Smoke grid

- \(\rho\in\{0,.9\}\)
- \(D\in\{0,8\}\)
- \(q\in\{1,4,16,32\}\)
- two implementation seeds
- common message budget and server overhead

The first implementation attempt used a 40,000-unit budget and a
32-by-32 network but exceeded the three-minute local smoke timeout before
writing artifacts.  It was terminated at the registered hard timeout.  Because
this phase is explicitly implementation-only, the feasibility configuration
uses an 8,000-unit budget and a 16-by-16 network; any formal experiment must be
preregistered independently.

The smoke records final teacher MSE, maximum training loss, update count, and
finite execution.  Its purpose is to choose a stable learning-rate and budget
range for a later preregistered nonlinear experiment.  No paper claim may use
the smoke results.

## Pilot correction for the cost confound

The first completed smoke used server overhead \(h=8\) and budget \(B=8{,}000\).
It was numerically stable, but the resource rule

\[
T_q=\left\lfloor B/(h+q)\right\rfloor
\]

gave \(888\) updates for \(q=1\) and only \(200\) for \(q=32\).  The resulting
preference for small \(q\), including when \(\rho=0\), is therefore not a clean
test of correlation-limited participation.

A second implementation-only smoke uses \(h=64\) and \(B=64{,}000\), giving
984, 941, 800, and 666 updates for \(q=1,4,16,32\).  This retains a common
resource budget while reducing the update-count ratio from 4.44 to 1.48.
The correction was chosen from the update-count formula, before observing the
second smoke outcomes.  This remains feasibility work and is not admissible as
paper evidence.

## Gradient-variance mechanism audit

End-to-end training can still mix gradient variance, optimization bias, delay,
and the number of updates.  The final feasibility check therefore freezes one
randomly initialized nonlinear value network and estimates the covariance
trace of its averaged TD semi-gradient over independent stationary Markov
transitions.  It uses

- \(\rho\in\{0,.25,.9\}\);
- \(q\in\{1,2,4,8,16,32\}\); and
- 2,048 Monte Carlo replicates per configuration.

Because two agents use the identical common transition with probability
\(\rho\), their marginal gradients are identically distributed and their
cross-covariance is exactly \(\rho\Sigma_g\).  Consequently, at the fixed
network parameter,

\[
\frac{\operatorname{tr}\operatorname{Cov}(\bar g_q)}
     {\operatorname{tr}\operatorname{Cov}(g_1)}
=\rho+\frac{1-\rho}{q}.
\]

The smoke compares the measured ratio with this prediction.  It tests the
nonlinear parallel-variance mechanism only; it is not a convergence result and
cannot be cited as formal evidence without an independent preregistration.

## Realizable end-to-end diagnostic

The analytic teacher in the first training smoke need not lie exactly in the
16-by-16 student class, so terminal teacher MSE may be dominated by
approximation error.  A final implementation-only diagnostic uses a fixed
network of exactly the same architecture as the teacher and adds unit-variance
zero-mean reward noise:

\[
R_t=V^\star(S_t)-\gamma V^\star(S_{t+1})+\xi_t.
\]

The common/private source selection is applied to the complete transition,
including \(\xi_t\), so the same hidden-sharing correlation mechanism is
preserved.  The test reuses the \(h=64\), \(B=64{,}000\) cost model and the
two-seed \((\rho,D,q)\) grid.  It is intended only to decide whether a
preregistered end-to-end nonlinear study is identifiable under the proposed
synthetic design.
