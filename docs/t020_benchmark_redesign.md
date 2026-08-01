# T-020 outcome-free nonlinear benchmark redesign scan

## Research-source boundary

The requested `academic-research-suite` skill is not installed in this Codex
workspace. This scan therefore uses only primary papers, official environment
documentation, and upstream source code:

- [MinAtar paper, arXiv:1903.03176](https://arxiv.org/abs/1903.03176);
- [official MinAtar repository](https://github.com/kenjyoung/MinAtar);
- [upstream MinAtar environment wrapper](https://raw.githubusercontent.com/kenjyoung/MinAtar/master/minatar/environment.py),
  whose default sticky-action probability is 0.1;
- [upstream Asterix source](https://raw.githubusercontent.com/kenjyoung/MinAtar/master/minatar/environments/asterix.py),
  which samples spawn direction and treasure/enemy type;
- [official Gymnasium FrozenLake documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/),
  where slippery transitions choose the intended and two perpendicular
  directions with probability 1/3 each;
- [Machado et al., sticky-action ALE protocol, arXiv:1709.06009](https://arxiv.org/abs/1709.06009).

No endpoint outcome was used to select a successful candidate, budget, rho,
delay, or cell. No environment was stepped and no scientific trajectory was
generated.

## Required marginal-preserving coupling

For every candidate, define one complete exogenous random stream `U0` and iid
private streams `Ui`, each containing behavior-policy randomness, transition
randomness, reward randomness, resets, and sticky-action draws. Agent i uses
`U0` with probability `sqrt(rho)` and `Ui` otherwise. All streams have the
same law, so each agent retains exactly the standard fixed-policy Markov law;
two agents share the common stream with probability rho. No observation noise
is added.

This construction must be applied to complete Markov streams, not separately
to observations. Mixing remains known or independently certified; common
randomness is not a substitute for a mixing certificate.

## Outcome-free budget-ray algebra

For model parameter count `p`, participation q, and batching b, use public
communication cost

\[
C_m(q)=c_0+4pq
\]

and usable horizon

\[
H(q,b,D)=\max\left\{0,
\min\left(\left\lfloor\frac{M}{C_m(q)}\right\rfloor,
\left\lfloor\frac{E}{b}\right\rfloor\right)
-\left\lceil\frac{D_{90}}{b}\right\rceil\right\}.
\]

The equicorrelated variance factor is

\[
v(q,\rho)=\rho+\frac{1-\rho}{q}.
\]

At low rho, increasing q must materially reduce `v`; at high rho the benefit
must saturate at rho. In a message-binding approximation proportional to
`v(q,rho) C_m(q)`, the continuous internal optimum is

\[
q^*(\rho)=\sqrt{\frac{(1-\rho)c_0}{4\rho p}}.
\]

This gives an outcome-free architecture/cost check. For a 3,169-parameter
one-hot FrozenLake MLP, `c0/(4p)=5.17` and q*(0.1)=6.82. For a 1,521-parameter
small globally pooled MinAtar CNN, the values are 10.77 and 9.85. Both put the
low-rho message optimum inside `{4,16}`. A conventional large MinAtar CNN
would make per-agent payload dominate and collapse the optimum toward q=1;
such an architecture fails before any experiment.

Prospective rays must be derived from a public target horizon `H0`, not pilot
outcomes:

- message ray: `M = H0 C_m(4)`, `E = 2 H0`;
- environment ray: `E = H0`, `M = H0 C_m(32)`;
- delay rays: public `D90/H0` fractions such as 0, 0.05, and 0.20.

These formulas ensure that message cost can produce an internal q and delay
changes usable horizon. They do not guarantee that a task retains 5% oracle
value after its task×budget fixed baseline is selected.

## Candidate scan

| Candidate | Endogenous stochasticity | Low/high-rho variance shape | Internal message q* | Delay changes horizon | ≥5% post-baseline oracle certified | Decision |
|---|---|---|---|---|---|---|
| FrozenLake-v1 8×8, `is_slippery=True` | Three documented transition directions, each probability 1/3; sparse stochastic terminal reward | yes algebraically | yes for the small public MLP | yes | **no**; fixed policy, spectral/mixing constants, and full bound are not yet frozen | retain only as calibration candidate |
| MinAtar Asterix-v1, sticky=0.1 | sticky actions plus random spawn side, slot, and treasure/enemy type | yes algebraically | yes only with the specified small pooled CNN | yes | **no**; no independent mixing/value certificate is available | highest-priority nonlinear candidate, still infeasible |
| MinAtar Breakout/SpaceInvaders | sticky action supplies stochasticity, but much of the game update is deterministic | algebra holds but stochastic learning-value floor is weak | architecture-dependent | yes | no | reject as primary tasks |
| Slippery CliffWalking/Gridworld | stochastic transitions can be public and exact | yes | possible | yes | not certified; small-state/transient dominance risk | secondary analytic calibration only |

No candidate currently passes all five feasibility requirements. FrozenLake
has the cleanest public transition law but sparse rewards and a small state
space. Asterix has genuine transition/reward randomness and a neural state,
but its fixed-policy mixing certificate and 5% task×budget oracle bound remain
open. Therefore the scan does not authorize a new benchmark identifier.

## Proposed learning-value surrogate

For cached candidate `(q,b)`, propose the scalar lower confidence value

\[
\underline V_t(q,b)=
\eta s_{t,L}
-\frac{\bar L\eta^2}{2}\left[
s_{t,L}+\tau_U\sigma^2_{t,U}v(q,\rho_U)\right]
-\eta\delta_{t,U}\frac{D_{90}}{b}
-\pi_m C_m(q)-\pi_e b-\beta_t(q,b).
\]

Inputs and observability:

- `s_t,L`: lower confidence bound on squared mean TD-gradient signal, using
  only training gradients;
- `sigma²_t,U`: upper confidence bound on within-probe gradient dispersion;
- `rho_U`: independently charged probe certificate;
- `tau_U`: public known-mixing/certificate bound;
- `delta_t,U`: upper confidence bound on observable stale/current gradient
  difference;
- `D90`, message/environment prices and remaining budgets: public runtime
  quantities;
- `Lbar`: a prospectively verified smoothness/spectral/clipping bound;
- `beta_t`: time-uniform scalar confidence radius.

Held-out prediction error, Bellman evaluation data, true rho, common/private
source labels, and another policy's outcomes are forbidden inputs. Candidate
evaluation is a cached 12-element vector: time O(|Q||B|)=O(12), online memory
O(1), with no Hessian inverse, covariance inverse, or preconditioner.

A safety shield compares every candidate LCB with the task×budget fallback and
allows adaptation only when cumulative certified surplus remains nonnegative;
otherwise it executes the fallback. By induction this preserves nonnegative
surplus if all confidence bounds and `Lbar` are valid. It is not yet an actual
terminal-error no-harm theorem: ReLU smoothness and the time-uniform confidence
certificate remain unproved. The paired empirical no-harm gate is still
required, together with a separate nontriviality gate.

Ten CPU tests pass for the existing ceiling, full probe accounting, q=1
fallback eligibility, variance saturation, internal-q cost condition, delay
horizon, vectorized surrogate, and safety-ledger invariance. These tests
validate algebra and implementation shape, not the missing statistical
certificate.

## Redesign status

- static current-benchmark oracle gate: failed;
- candidate all-condition scan: failed;
- surrogate formula/observability/complexity: specified;
- complete no-harm certificate: failed/open;
- CPU algebra tests: passed;
- scientific trajectories: zero;
- new GPU pilot preregistration: not authorized.
