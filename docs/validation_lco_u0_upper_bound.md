# LCO-U0 controlled-sensing upper-bound validation

## Decision

The action-dependent-only sensing interface is stopped.  The perfect paid
sensor passes 13 of 14 frozen gates but fails U11: its median capture of the
exact-phase adaptation headroom is 0.559262, below the frozen 0.60 threshold.
The threshold is not rounded or changed.  No implementable noisy sensor can be
more informative than this upper-bound model under the same rule that only an
optimistic update produces a geometry observation.

This does not say that dynamic optimism lacks value.  It says that making the
scarce stabilizing action also carry the full identification burden imposes a
material information tax.  LCO-S0 and LCO-V0 exposed this tax empirically;
LCO-U0 shows that it persists even under perfect observations and a
Bayes-optimal constrained belief policy.

## Frozen analytic result

- execution commit: `263ef8b`;
- configuration SHA-256:
  `ca199339a419ef66e556769fdf4eea7e0dc3d0771fa19770b417f7d7813f523a`;
- result SHA-256:
  `fa961fc4062b7ed52286c7d718fe2a015954e89d38442deb60c507aa71569fef`;
- 120 analytic cells, 360 sparse occupation-measure LPs, no trajectory seed;
- age truncations 128/256/512.

The numerical certificate is clean: the largest age convergence gap is
(3.11\times10^{-15}), flow residual (2.50\times10^{-16}), calibration
residual (2.11\times10^{-15}), and normalization residual
(4.00\times10^{-15}).  Every cell obeys

\[
L_{\rm exact}\leq L_{\rm perfect\ paid\ sensing}\leq L_{\rm fixed},
\]

with zero budget overshoot and zero stationary-potential calls.

| Frozen headroom metric | Threshold | Observed | Result |
|---|---:|---:|---|
| Mean dynamic upper gain | at least 0.03 | 0.042747 | pass |
| Dynamic cells with gain at least 0.02 | at least 0.75 | 0.75 | pass |
| Low-persistence mean upper gain | at least 0.025 | 0.032498 | pass |
| Low-budget mean upper gain | at least 0.015 | 0.024013 | pass |
| Median exact headroom capture | at least 0.60 | 0.559262 | **fail** |

The loss is structural in precisely the difficult regimes.  Mean capture is
about 0.427 at persistence 0.8 versus 0.704 at persistence 0.95, and 0.456 at
budget 0.25 versus 0.631 at budget 0.5.  At normalized step 0.2 the absolute
upper gain is only 0.004248.  The stronger overall 4.27% mean is driven by
larger-step and higher-budget cells, not a uniform resolution of the sensing
problem.

## Mainline consequence

No further HMM probe frequency, information bonus, lookahead depth, or
action-dependent sensor is authorized on this model.  The next viable
interface must obtain geometry information from observations already produced
by every ordinary asynchronous update.

The resulting unified candidate is **passive clocked secant sensing**:

> Asynchronous coordinate updates create rotational instability, but those
> same coordinate perturbations expose cross-agent gradient responses.  Use
> the resulting secants to identify local potential/rotational geometry at
> ordinary-update cost, then use Lyapunov resource debt to buy optimism only
> for stabilization.

For a local linear game (F(x)=Ax), consecutive current-gradient pairs obey

\[
\Delta g=A\Delta x
\]

within a stationary local phase.  The alignment and orthogonal residual of
\(\Delta g\) relative to \(\Delta x\) separate symmetric contraction from
rotational coupling without forming (A).  Online arithmetic is (O(d)), and
optimism no longer has to be wasted to observe geometry.  In stochastic MARL,
the proof obligations are persistent excitation, phase-change contamination,
Markov gradient noise, asynchronous block availability, and a gain-weighted
mistake bound connected to the existing last-iterate supermartingale bridge.

This passive interface needs its own outcome-free identifiability and oracle
headroom audit before another development run.  LCO-U0 does not authorize
formal seeds, standard MARL, GPU, or HPC4.
