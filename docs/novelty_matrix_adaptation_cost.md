# Novelty matrix: adaptive participation cost

## Citation-integrity rule

Only primary paper, proceedings, journal, arXiv, or publisher pages were
used. The machine-readable companion is
`docs/novelty_matrix_adaptation_cost.json`.

| Work | Markov data | Agent correlation | \(q\) changes information and cost | Delay | Uniform safety | Adaptation/ID lower bound | Matching algorithm |
|---|---|---|---|---|---|---|---|
| Khodadadian et al., *Federated RL: Linear Speedup Under Markovian Sampling*, ICML 2022 | yes | no; separate-agent sampling | fixed number of agents | no | no | no correlation-ID lower bound | federated TD/Q upper bounds |
| Yin et al., *Gradient Diversity*, AISTATS 2018 | no Markov model | gradient similarity | batch size affects scaling | no | no | convergence lower bound from low diversity | diversity mechanisms, not cost-aware ID |
| Alfarra et al., *Adaptive Learning of the Optimal Batch Size of SGD*, 2020 | iid SGD | gradient variance, not cross-agent Markov correlation | batch size changes work | no | no | no safety/Markov ID lower bound | adaptive batch strategy |
| Wu et al., *Conservative Bandits*, ICML 2016 | iid/adversarial bandit | no | arm pull has standard unit cost | no | cumulative baseline constraint | conservative-regret lower bound | near-matching conservative algorithm |
| Gangrade et al., *Safe Linear Bandits over Unknown Polytopes*, COLT 2024 | iid linear bandit feedback | no | action changes reward/risk observations | no | smooth roundwise violations | safety-efficacy hardness | near Pareto-optimal DOSS |
| Kanarios et al., *Cost Aware Best Arm Identification*, RLC 2024 | iid arms | no | arm-dependent test cost | no | no baseline no-harm | cost-aware BAI lower bound | asymptotic CTAS; two-arm CO |
| Kaufmann et al., *Complexity of Best-Arm Identification*, JMLR 2016 | iid arms | no | standard pulls | no | no | fixed-confidence/budget change of measure | matching two-arm procedures |
| Carpentier & Locatelli, *Tight Lower Bounds for Fixed Budget BAI*, COLT 2016 | iid arms | no | standard pulls | no | no | fixed-budget error lower bound | comparison algorithms |
| Moulos, *Optimal Best Markovian Arm Identification*, NeurIPS 2019 | rested Markov arms | arms independent | action selects one arm | sampling dynamics | no | Markov BAI lower bound | Track-and-Stop within factor four |
| Karthik et al., *Best Arm Identification in Restless Markov Bandits*, 2022/2023 | restless Markov arms | arms modeled separately | action selects observed arm | restless evolution | no | Markov change-of-measure lower bound | partial/special-case matching |
| This EXP-015A route | yes | explicit common factor | yes | yes | explicit safety deficit relative to all-agent | exact fixed-design threshold; adaptive case open | horizon-aware fixed-design ETC |

Primary links:

- https://proceedings.mlr.press/v162/khodadadian22a.html
- https://proceedings.mlr.press/v84/yin18a.html
- https://arxiv.org/abs/2005.01097
- https://proceedings.mlr.press/v48/wu16.html
- https://proceedings.mlr.press/v247/gangrade24a.html
- https://rlj.cs.umass.edu/2024/papers/Paper193.html
- https://www.jmlr.org/papers/v17/kaufman16a.html
- https://proceedings.mlr.press/v49/carpentier16.html
- https://papers.nips.cc/paper/8798-optimal-best-markovian-arm-identification-with-fixed-confidence
- https://arxiv.org/abs/2203.15236

## Novelty gate

The fixed-design theorem does not collapse algebraically to a generic
two-arm bandit: the covariance eigenvalues depend jointly on \(q\theta\) and
\(\lambda^b\), while the feasible sample count depends on both
\(n(h+q)\) and \(nb+D\). This is a real structural distinction.

The full paper novelty gate is nevertheless only **provisionally passed**.
The literature already contains cost-aware BAI, conservative exploration,
and Markov-arm identification separately. Without AC-7--AC-9 (adaptive,
unknown-mixing lower and upper bounds), the current result should be framed as
a precise mechanism theorem and negative/phase-transition study, not as a
completed ICML-level matching theory.

