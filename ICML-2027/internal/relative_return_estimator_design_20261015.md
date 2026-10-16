# Variance-sensitive relative-return estimator: frozen qualification

## Material Passport

- Type: original mathematical derivation and prospective CPU estimator qualification.
- Status: implementation and operational qualification complete; scientific outcomes unopened at registration.
- Scope: finite-horizon native episodic team return; resettable deterministic discrete-action MPE simulator.
- Provenance: parent `e72f7df`; frozen files in `relative_return_source_lock_20261015.json`.
- Exclusions: not a MARL optimizer, not an async queue theorem, not performance superiority, not an ICML-readiness decision.

## Substantive change

The previous distribution-free vector-gradient radius exceeds the signal at
16 episodes. Replacing its constants is insufficient. This design estimates
the *difference of actual team returns* directly and integrates out a rare
event: the first action at which two nearby policies disagree. It does not
substitute a critic loss, PPO surrogate, or training-reward trend for return.

The reference is a fresh, complete, sequential HAPPO update. Alternative
frozen policies interpolate the complete pre/post-update actor parameters at
scales 0, 0.5, and 1.5; reference scale 1 is the eligible zero-score action.
These counterfactual policies are estimation targets, not a proposed Adam
update or a claim that interpolation improves HAPPO.

## Exact expectation identity

Condition on policies, training history, and an independently sampled initial
state. Couple each actor's categorical actions maximally. At a common state,
let `c_t` be the product of the actors' categorical overlap probabilities.
Generate a whole common trajectory using each actor's normalized minimum
probabilities. Its prefix distribution is the ordinary coupled process
conditioned on survival, with survival factor `S_t = product_{s<t} c_s`.
Define `w_t = S_t(1-c_t)` and `W = sum_t w_t <= 1`.

Sample a branch index with probability `w_t/W`. At that state sample the
product maximal coupling conditioned on at least one actor disagreement,
and run a valid coupled suffix under the two policies. Let `D_t` be their
suffix return difference. The estimator is `X = W D_T`, or zero if `W=0`.

For any fixed complete common path,

`E[X | common path] = sum_t w_t E[D_t | common prefix t, disagreement]`.

Integrating the common path reproduces the disjoint first-disagreement
partition of the ordinary coupling. Trajectories with no disagreement have
identical return, so `E[X] = J(candidate)-J(reference)`. Future common-path
states used in choosing T do not bias the identity: the factor `w_t/W`
cancels W pointwise, and those future states are integrated out.

The code samples the first *actor* disagreement with masses
`product_{j<i} overlap_j * (1-overlap_i)`. Earlier actors are common, that
actor uses the disjoint residual distributions, and later actors use ordinary
maximal coupling. Forcing every actor to disagree would be biased and is not
implemented. Prefix states are reconstructed from the same environment seed
and exact action history, and checked byte-for-byte before branching.

## Second-moment comparison

Let Y be the ordinary maximally coupled full-episode return difference with
the same conditional suffix coupling. Its second moment is

`E[Y^2] = E_common sum_t w_t E[D_t^2 | prefix t, disagreement]`.

The new estimator has

`E[X^2] = E_common W sum_t w_t E[D_t^2 | prefix t, disagreement] <= E[Y^2]`.

Both have the same mean, hence `Var(X) <= Var(Y)`. If a *known uniform*
bound `W <= wbar` is available, then
`Var(X) <= wbar Var(Y) - (1-wbar)(E Y)^2 <= wbar Var(Y)`.
Observed maximum weights are not uniform bounds. This result does not order
X against inverse-CDF common random numbers; both paired methods are tested.

The implementation pays H steps for the common trajectory and two H-step
replays, thus at most 3H versus 2H for paired rollouts. Under a known bound
`wbar < 2/3`, the transition-cost-adjusted variance improves theoretically.
Policy forwards and wall-clock remain additional costs, and are measured.
No claim of universally better variance-per-second follows.

## Confidence interface and remaining limitation

For independent samples and a *known* support interval of width R, use
Maurer--Pontil empirical Bernstein with

`r = sqrt(2 sample_variance log(4/delta)/m) + 7 R log(4/delta)/(3(m-1))`.

Allocate failure probability across frozen candidates and decision times.
Independence is at episode level, not time-step level. A resettable simulator
supplies independent initial states and action randomness here; continuous
Markov streams require a separate justified bias/dependence treatment.

Small empirical variance alone does not remove the support term. Neither
sample maxima, a fitted critic, nor a normal-SE lower score is a rigorous
replacement for R. Qualification reports empirical ranking and precision
separately from this still-open nonvacuous confidence gate. Without that gate,
the previous launch-promise theorem cannot be instantiated as a deployable
certified PPO/Adam controller.

## Prospective protocol and costs

Eight contexts: two native tasks, two entirely fresh seeds, training stages
4 and 16. Each has three non-reference policies, 64 independent score samples
per method, and a disjoint 256-episode paired validation. Validation is never
passed into candidate construction, estimator, or selection. Source hashes,
exact transition/forward ledgers, identity-zero controls, and one deterministic
context replay are mandatory. The finite two-agent Markov game is checked by
exact enumeration, not simulation. The methods, gates and seeds are in JSON.

The 64-sample three-candidate probes alone cost 14,400 joint transitions per
context for the new method. This is 9x a stage-4 training prefix or 2.25x a
stage-16 prefix. Therefore qualification cannot establish affordable *per-update*
deployment merely by passing relative variance gates. Required precision and
amortization over future decisions must be solved separately; the full probe
cost is reported rather than hidden. No successor GPU run is authorized by
this estimator qualification.

Operational seed 74109999: one stage, two score and two validation samples;
only execution, sample counts and ledgers were inspected, not reward means.
It completed in 5.92 active CPU seconds. All three methods returned exact zero
on the identity control. Five estimator tests passed, including exact Markov
expectation reconstruction and conditional joint-coupling marginals.

Commands after the preregistration commit:

```
conda run --no-capture-output -n ust2 python ICML-2027/experiments/run_relative_return_qualification.py --lock ICML-2027/internal/relative_return_source_lock_20261015.json --output ICML-2027/results/relative_return_scientific_20261015
conda run --no-capture-output -n ust2 python ICML-2027/experiments/run_relative_return_qualification.py --lock ICML-2027/internal/relative_return_source_lock_20261015.json --context 0 --output ICML-2027/results/relative_return_registered_replay_20261015
```

## Prior-work boundary

Conditional Monte Carlo, coupling, measure-valued policy-gradient estimators,
and variance-sensitive confidence bounds predate this work. The current
first-disagreement construction is an internally derived estimator, not a
verified novelty claim. Carvalho et al. (arXiv:2107.09359) and Carvalho--Peters
(arXiv:2203.03917) study measure-valued policy gradients. Maurer--Pontil
(arXiv:0907.3740) supplies the confidence interface. The 2021 and 2022 papers
have different author lists and are not silently merged. A wider novelty
audit is needed before any paper claim about this specific construction.
