# T-017 theorem-by-theorem novelty confrontation

## Verification method

The requested `academic-research-suite` workflow was unavailable in this
workspace. To preserve citation integrity, this audit used only the complete
papers obtained from the authors' arXiv records or official
journal/proceedings pages, checked title/authors/venue/year and theorem text,
and recorded the result in `citation_verification_t017.json`. Search snippets
and secondary summaries were not used as evidence.

## Required comparison set

| Work | Controlled Markov | Changing observation dimension | Unknown covariance | \(q\): information + learning | Dual budgets | Delay / usable horizon | Objective | Finite-budget worthwhile threshold | Safety | Asymptotic matching |
|---|---|---|---|---|---|---|---|---|---|---|
| Nitinawarat & Veeravalli (2015), controlled sensing with non-uniform cost | yes: finite-state controlled Markov observation law | no | model is known under each hypothesis | control selects sensing law and scalar cost, not downstream learning participation | one accumulated control cost | no | sequential multihypothesis decision | nonasymptotic risk constraints; no learning-value threshold | no baseline-relative safety | yes, information per expected cost as risks vanish |
| Garivier & Kaufmann (2016), Track-and-Stop | no: iid one-arm observations | no | arm parameters unknown, not joint covariance | allocation changes ID information only | pull count | no | fixed-confidence best-arm ID | sample-complexity lower bound, not paid ID versus later learning gain | no | yes as \(\delta\to0\) |
| Moulos (2019), best Markovian arm | yes: independent rested finite-state Markov arms | selected arm only | one-parameter transition families | arm choice changes ID information, not common-factor learning participation | pull count | rested evolution, not delayed usable updates | fixed-confidence best-arm ID | no post-ID learning-value threshold | no | factor-four Markov Track-and-Stop result |
| Saad, Blanchard & Verzelen (2023), covariance-adaptive BAI | iid over time | yes: subsets/simultaneous rewards | yes: unknown covariance | query subset changes joint information/query count, not subsequent Markov learning speed | query complexity | no | fixed-confidence BAI | ID query complexity only | no | near-optimal for specified classes, up to logarithmic factors |
| Vannella, Proutiere & Jeong (2023), MAMAB BAI | iid Gaussian factored rewards | fixed factor/semi-bandit observation pattern under joint actions | no unknown common covariance mechanism | joint action changes BAI information, not adaptive agent count for downstream learning | samples | no | multi-agent/factored best joint-action ID | no paid-ID-versus-learning threshold | no | MF-TaS matches its approximated lower bound asymptotically |
| MARL-SDDE AC-7 | yes: scalar stationary Gaussian Markov factor with controlled stride | yes through predictable \(q_t\) | hypotheses differ in covariance; known \(\lambda\) | observation information only at AC-7 level | message + environment | exact resource/usable-horizon accounting | regime ID lower bound | no | no | no algorithmic matching claim |
| MARL-SDDE T-017 repaired route | yes | yes | restricted unknown mixing route only; unrestricted closed negatively | yes: \(q\) also sets correlation-limited terminal risk | message + environment | yes; delay removes usable learning updates | ID **and** post-ID oracle regret | yes: \(B_N,B_S,B_{\rm oracle}\) sandwich on separated class | explicit baseline-relative deficit | threshold sandwich only; full occupation matching open |

## Theorem-level inheritance audit

### AC-7 change of measure

Nitinawarat--Veeravalli already allow arbitrary causal randomized control
kernels, factor the controlled Markov history law, cancel the identical
control kernels in a likelihood ratio, optimize KL information per control
cost, and prove asymptotic lower/upper cost results. Garivier--Kaufmann and
Moulos likewise use decision-time change of measure and allocation
constraints. Therefore these generic ingredients are **inherited**, not a
novel theorem template.

The exact T-016 specialization remains useful but must be described as such:
the spatial rotation, changing-\(q\) scalar innovation, two hypothesis-specific
Kalman filters, irregular \(\lambda^{b_t}\), both KL directions, and
dual-budget stopping instantiate that machinery for this Gaussian model.
They do not justify claiming invention of adaptive controlled sensing.

### Dimension-changing covariance identification

Saad--Blanchard--Verzelen already study simultaneous subset observations with
unknown covariance and covariance-adaptive BAI query complexity. Thus
“unknown covariance plus changing query dimension” is **not** by itself a
novelty claim. Vannella--Proutiere--Jeong also show that multi-agent/factored
action structure and asymptotically matched BAI lower bounds already coexist.

### AC-8 negative theorem

The T-017 contribution is narrower: when temporal mixing is unrestricted up
to non-mixing, every adaptive finite-budget history at \(\lambda=1\) is a
garbling of one Gaussian latent draw. The exact positive Le Cam error floor
and its uniform \(\lambda_T\uparrow1\) extension close the unrestricted
unknown-mixing route negatively. None of the five required papers supplies
this common-factor mixing-boundary theorem.

### AC-9 threshold sandwich

Pure identification cost is already well covered by all five literatures.
The defensible distinction is that identification changes a subsequent
learning optimizer. Here \(q\) changes both covariance information and the
correlation-limited learning risk; \(b\) changes both information rank and
environment consumption; delay subtracts usable commit updates; a wrong
decision incurs oracle regret; and low-regime behavior is constrained by a
baseline-relative safety deficit. The resulting “when is adaptation worth
paying for?” thresholds are not consequences of a pure BAI stopping theorem.

This distinction supports only the compact separated threshold theorem. It
does not prove that the controller asymptotically matches the entire
controlled-belief optimum.

## Novelty verdict

- **Rejected as novel:** causal action-kernel cancellation; controlled
  Markov information-per-cost; generic binary change of measure; BAI
  allocation lower bounds; covariance-adaptive subset querying; multi-agent
  factored BAI matching.
- **Defensible contribution candidate:** the coupled correlation-learning
  value / dual-cost / delay / safety threshold, plus the unrestricted
  unknown-mixing impossibility theorem.
- **Mandatory wording:** known mixing or a public/certified separation
  \(\lambda\leq1-\gamma\); threshold matching on a compact separated class;
  no global occupation-optimality claim.

## Primary sources

1. Nitinawarat and Veeravalli, “Controlled Sensing for Sequential
   Multihypothesis Testing with Controlled Markovian Observations and
   Non-Uniform Control Cost,”
   *Sequential Analysis* 34(1), 2015,
   DOI `10.1080/07474946.2014.961864`.
2. Garivier and Kaufmann, “Optimal Best Arm Identification with Fixed
   Confidence,” COLT 2016, PMLR 49.
3. Moulos, “Optimal Best Markovian Arm Identification with Fixed
   Confidence,” NeurIPS 2019.
4. Saad, Blanchard, and Verzelen, “Covariance Adaptive Best Arm
   Identification,” NeurIPS 2023, DOI `10.52202/075280-3204`.
5. Vannella, Proutiere, and Jeong, “Best Arm Identification in Multi-Agent
   Multi-Armed Bandits,” ICML 2023, PMLR 202.
