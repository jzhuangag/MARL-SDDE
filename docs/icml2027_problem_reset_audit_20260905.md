# ICML 2027 problem-level reset audit

Date: 2026-09-05

Status: **the bounded fresh search is closed without authorizing a new
algorithm or experiment.**  This is an internal research decision.  It is not
an impossibility claim for all asynchronous MARL problems.

## Decision rule

After the closure of participation control, dynamic collaboration graphs,
perishable-update backpressure, stale-gradient correction and coupled
actor--critic drift control, a new candidate was required to satisfy all of
the following before algorithm construction:

1. a genuinely multi-agent and asynchronous theorem object;
2. an intrinsic equal-resource learning advantage over the strongest relevant
   online baseline;
3. an online signal already present in mandatory training data or metadata;
4. a Lyapunov variable that changes the executed learning action rather than
   serving only as an after-the-fact proof device;
5. a plausible standard deep-MARL interface.

No efficacy trajectory, pilot seed, GPU job or HPC4 operation was run during
this audit.

## Candidates screened

### 1. Asynchronous constrained-MARL trust-region allocation

The proposed action was a vector of per-agent trust-region/update masses,
selected by a Lyapunov drift-plus-penalty problem with delayed constraint
queues.  This was rejected before experimentation.  Its proof reduces to
delayed stochastic primal--dual optimization plus a MARL performance-difference
bound; asynchrony alone does not create a new multi-agent theorem object.
Recent constrained-MARL work already covers shared constraints, primal--dual
learning and sequential local trust regions.  Adding delay and a per-agent
mass vector would not create enough separation for the intended paper.

### 2. Factor-local correction of mixed-version trajectories

This was the only candidate with a potentially order-of-magnitude statistical
gap.  Independent agent clocks produce a behavior policy

\[
\mu_h(a\mid o)=\prod_j
\pi_{j}^{v_{j,h}}(a_j\mid o_j),
\]

so full joint importance weights can have a second moment exponential in the
number of agents and horizon.  The proposed algorithm would retain only
causally relevant likelihood-ratio factors and use a Lyapunov variance queue to
choose a correction mask.

The model-free adaptive algorithm fails its first theory gate.

#### Exact result that survives

Consider an unrolled factored dynamic Bayesian network in which target and
behavior laws share the initial, transition, observation and reward kernels
and differ only in predictable action kernels.  Let a bounded functional
\(F\) be measurable on nodes \(U\), and let

\[
\mathcal C(U)=
\{(j,h):A_{j,h}\in\operatorname{An}(U)\}.
\]

Under support and predictable-version assumptions,

\[
\mathbb E_{\pi}[F]
=
\mathbb E_{\mu}\!\left[
F\prod_{(j,h)\in\mathcal C(U)}
\frac{\pi_{j,h}(A_{j,h}\mid O_{j,h})}
     {\mu_{j,h}(A_{j,h}\mid O_{j,h})}
\right].
\]

The proof eliminates non-ancestor kernels in reverse topological order.  The
identity already corrects state-occupancy shift, but only because it retains
every past action ancestor.  A same-time or immediate-neighbor ratio is not
enough.

For an agent policy gradient, one must first decompose

\[
\nabla_{\theta_i}J
=\sum_{t,c,u\ge t}\gamma^u
\mathbb E_{\pi}[\psi_{i,t}R_{c,u}].
\]

Each retained score--reward term requires the action ancestors of
\(\{O_{i,t},A_{i,t},R_{c,u}\}\).  In a long-horizon interacting Markov game,
this cone generally expands to most or all agents.

#### Why the adaptive mask is not certified

The known graph identifies structural non-ancestors, whose influence is
exactly zero.  It does not bound the strength of a true ancestor.  Realized
logged ratios also cannot control behavior on unobserved actions or target-only
states.  Two target policies can agree on every ratio observed in a packet and
have nearly maximally different downstream rewards.

A valid bound for omitting an ancestor would require at least one of:

- known transition or Dobrushin influence coefficients;
- uniform-over-observation policy TV/KL certificates;
- an independently sampled split;
- a verified model-based counterfactual bound.

None is free in the intended model-free deep-MARL setting.  Moreover, choosing
the mask after observing the same trajectory's likelihood ratios or TD
advantages creates selection bias.  The mask must be predictable before the
trajectory is revealed or be chosen from independent data.  The latter
reintroduces the sensing/sample-splitting cost that closed earlier routes.

Using the full exact ancestor cone leaves no nontrivial Lyapunov mask action;
using an approximate cone without the information above leaves an unproved
bias term.  Therefore the performance-bound gate fails and the outcome-free
oracle experiment is not authorized.

This proposal also collides with an existing repository route:
`Reuse--Correct--Refresh` already used factorized teammate likelihood-ratio
tempering, a virtual-queue price and an online correction QP.  Its frozen
headroom audit did not justify a new mainline.

### 3. Asynchronous joint-policy deployment coherence

Scheduling which policy versions are activated on different agents may have a
real systems value, but standard MARL benchmarks normally assume atomic policy
deployment.  The value estimate would require cross-play or a model that is
not available from mandatory training data.  This candidate was rejected as a
deployment/systems problem without an established equal-resource learning
advantage.

## Literature boundary

The search used primary records and is bounded rather than exhaustive.

- Foerster et al. introduced multi-agent importance sampling and policy
  fingerprints for changing teammate policies at ICML 2017.
- Rowland et al. developed a general conditional-importance-sampling framework
  at AISTATS 2020.
- Zawalski et al.'s MA-Trace extends V-trace to decentralized joint policies
  and distributed MARL.
- Rebello et al. already prove variance reduction for decomposed importance
  sampling in factored action spaces under explicit structural assumptions.
- Schmitt et al. analyze the correction/no-correction bias--variance tradeoff
  for shared replay.
- Zhao et al. already establish localized multi-agent policy optimization under
  stated structural conditions.

Consequently, importance sampling, factorization, locality, clipping, a
virtual queue, or mixed policy versions are not individually novel.  A valid
new paper would need the complete asynchronous Markov-game identification and
learning theorem; the proposed model-free mask does not have it.

## Research decision

The project must not create another controller identifier, change a benchmark
until it favors the method, weaken a comparator, or launch GPU work from this
screening.  A negative Gate A is cheaper and more informative than another
failed pilot.

Within the user's fixed intersection of asynchronous MARL, stochastic
optimization, online Lyapunov design, standard deep benchmarks and a strong
positive result, **no currently defensible new mainline has passed the entry
conditions**.

The next scientifically coherent deliverable is a theory/limits manuscript
audit built from already proved assets: adaptive change of measure, the
unknown-mixing impossibility, finite-budget threshold sandwich, delay and dual
budget dependence, and the empirically documented cost of paid adaptation.
That manuscript is only a candidate, not an ICML-ready claim.  It must first
show that its theorem chain is both correct and materially stronger than a
collection of independent negative observations.  No new performance
experiment is authorized until that audit identifies one theorem-facing
positive regime and a matching strong comparator.

## Immediate CPU-only work

1. build a theorem dependency graph for the limits manuscript candidate;
2. recheck every assumption and quantifier in AC-7--AC-9, the unknown-mixing
   impossibility and the threshold sandwich;
3. decide whether the results compose into one minimax information--learning
   frontier or merely several unrelated lemmas;
4. only if they compose, specify a single frozen positive-regime confirmation
   and a standard-benchmark falsification test;
5. otherwise stop the ICML packaging and redirect the correct subset to a
   narrower theory or signal-processing venue.

This work needs no GPU.

## Subsequent finite-commit theorem audit

A later bounded audit did close a coherent binary finite-commit limits chain:
learning excess reduces to testing under fresh deployment, expected safety is
exactly represented by a likelihood-ratio-augmented coupled occupation
program, and a zero-probe baseline-as-sensing switch obtains an anytime Ville
safety guarantee with a conditional matching upper bound.  These results are
recorded in the 2026-09-05 limits-manuscript documents.

They do not reopen the asynchronous-MARL mainline.  The final outcome-free
oracle ceiling also failed: median ideal headroom was zero and only 21.7625%
of 30,502 feasible T-018 cells reached 10%, versus the frozen 10%-median and
60%-prevalence gate.  No new scientific experiment, GPU run, or controller
identifier is authorized.
