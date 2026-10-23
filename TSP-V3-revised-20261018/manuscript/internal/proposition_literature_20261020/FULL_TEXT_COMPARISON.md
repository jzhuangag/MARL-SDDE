# Three propositions: full-text-grounded comparison and admission decision

Date:2026-10-20. Baseline102ec6a. Scope is the three candidate claims in
mechanism_analysis_20261020/proposed_additions.tex, not the entire prior literature.
This is an AI-assisted technical reading and proof review, not external peer review
or an exhaustive priority/plagiarism clearance. No new scientific experiment.

## Reading corpus and access integrity

Four additional works were read in their technical full-text passages, not inferred
from abstracts. Their complete original PDFs are preserved as local reading caches
with SHA256/preflight sidecars; copyrighted third-party PDFs are NOT redistributed
in the manuscript package or committed. The earlier six-work learning/resource
comparison is retained unchanged as historical evidence; its prospective-execution
paragraph is superseded by the completed direct validation.

1. Esary, Proschan and Walkup(1967), *Association of Random Variables, with
   Applications*, Ann.Math.Statist.38(5):1466–1474,
   DOI10.1214/aoms/1177698701. Primary article copy hosted by CMU:
   https://www.stat.cmu.edu/~brian/720/week02/esary-proschan-walkup-1967.pdf
   Read Definition1.1; Section2 propertiesP2–P4 and Theorem2.1, their proofs;
   Section3 indicator/monotone-function criteria. Three-count preflight PASS10/10/10.
2. Russo and Zou(2016), *Controlling Bias in Adaptive Data Analysis Using
   Information Theory*, AISTATS/PMLR51:1232–1240.
   https://proceedings.mlr.press/v51/russo16.html
   Read main Sections1.1,2.1–2.6,3.1 and corresponding supplementary proofs of
   Propositions1,3,4,6. Main/supplement preflight PASS9/9/9 and3/3/3.
   The PDF footer says41; both the official article metadata and volume51 index
   identify51. Preserve the footer discrepancy; use the publisher's series record,
   not an invented DOI or silently altered PDF. This is a resolved representation
   discrepancy, not a claim that the original PDF was error-free.
3. Chaloner and Verdinelli(1995), *Bayesian Experimental Design: A Review*,
   Statist.Sci.10(3):273–304, DOI10.1214/ss/1177009939.
   Primary article copy in Berkeley's course archive:
   https://people.eecs.berkeley.edu/~jordan/courses/260-spring09/readings/chaloner-verdinelli.pdf
   The PDF contains a JSTOR cover followed by the article. Visually read the
   introduction, Section1.3 Eqs.(1)–(2), Section1.5 budget constraints and opening
   Section2 normal-model design. Section/equation locators are used, not trusted
   local PDF page anchors: all three counts agree33/33/33 but the preflight is
   UNAVAILABLE due to a stale-xref warning. Rendered source images are retained.
4. Manjunath and Wilhelm(2021), *Moments Calculation for the Doubly Truncated
   Multivariate Normal Density*, J.Behav.DataSci.1(1):17–33,
   DOI10.35566/jbds/v1n1/p2.
   https://jbds.isdsa.org/jbds/article/view/9
   https://isdsa.org/jbds/fulltext/v1n1/p2/
   Read published full-text Sections2–3 Eqs.(2),(6),(11),(16)–(18) and Section5
   univariate/partially truncated reduction Eqs.(24)–(26); compare the earlier
   arXiv1206.5387 version without conflating its2012 date with publication2021.
   Published PDF preflight PASS17/17/17. Rectangular truncation is their scope;
   a general ellipsoidal likelihood event is not rectangular by relabelling it.

Access failures are not full-text passes: ProjectEuclid Blackwell1953 and
Tallis1963 return an Incapsula challenge/iframe; the Tallis1961 Wiley PDF route
returns the abstract landing page. Berkeley's Blackwell1951 file gave an empty
response. No challenge, paywall or rate limit was bypassed. The Lieb–Loss university
library PDF contains nine front-matter pages only; it is not the proof of
Theorem1.14. These sources remain leads, NOT purportedly completed proof readings.
The openly read primary articles already establish the overlaps below; blocked
leads prevent an unqualified claim of original general Gaussian truncation theory.

## Proposition-by-proposition comparison

| Candidate | Existing result and exact correspondence | Remaining V3-specific content | Decision |
|---|---|---|---|
| P1: signed same-stream risk | Selection changes a selected statistic's expectation: Russo–Zou Sections1.1/2.3/2.4 and supplement. Independent coordinates are associated: Esary et al.Theorem2.1. Odd Gaussian moments on even events vanish by symmetry. These are not new principles. | Substitute the actual continuing learner's coefficients into its conditional risk; whiten the centered ordered-covariance likelihood quadratic, remove cross products, and obtain the two-regime sign controlled by k1²−k0². This identifies cases where no positive carryover penalty is needed. | Admit as an explicitly attributed, model-specific continuing-risk corollary, not a new general selection/association theorem. Exact sign algebra and scope checked below. |
| P2: probability-sensitive Gaussian moment bounds | Truncated-normal first/second moments are classical; the zero-mean scalar reduction of Manjunath–Wilhelm's moment calculation yields the central/tail integrals. Fixed-mass upper/lower level-set optimization has an elementary exchange proof, given in full. | Combine these scalar constants with P1 and frozen learner sensitivities to replace the generic square-root penalty by two different one-sided orders. Constants are sharp for the scalar event problem, not all finite probes. | Do NOT introduce a standalone new Gaussian tail theorem. Integrate the tool and its proof in the supplement; combine P1+P2 in one specialized corollary. |
| P3: finite-catalogue separation | Russo–Zou Section2.6 states the perfect-information value as E[max f]−max E[f], which becomes Iπ−Oπ with loss sign reversed. Chaloner–Verdinelli Section1.3 already evaluates design by a subsequent decision's expected utility and includes resource constraints. Randomized fixed actions are convex mixtures. | Instantiate the comparison using the ORIGINAL full-budget catalogue and the EXECUTED gate's paid-horizon/carryover population certificate. The analytical all128-design calculation gives an objective-specific separation test and its negative cases. | Classical decision corollary, NOT a new value-of-information theorem or universal irreplacability result. Include the criterion/cost decomposition and complete29/128 versus0/128 diagnostics. |

## Proof review and disallowed extrapolations

- P1: the precision difference is positive semidefinite because inversion reverses
  Loewner order. Whitening/diagonalizing preserves Gaussian independence. Posterior
  mean is a linear functional; coordinate sign changes remove its cross products.
  Independence makes the conditional covariance argument's extra tower covariance
  zero. Complementation gives the hypothesis-one sign. Fixed thresholds/ties and
  pre-data weights are required. The result is non-strict unless nondegeneracy is
  supplied; the integrated text says nonnegative/nonpositive, not always strict.
- P2: for g=E[1_E|m], optimize under0<=g<=1 and E[g]=p; the pointwise product
  (f−c)(g_star−g)>=0 gives the fixed-mass extremizer without restricting E to m.
  Gaussian integration by parts gives h+/h− exactly. Their derivatives t²−1 and
  1−u² have the needed signs for delta<=1/4. The sign-reversed high-regime term
  uses the CENTRAL h− bound, not the tail h+ bound. Delta=0 uses continuity.
- P3: a finite minimum and BOTH strictly positive prior weights are essential for
  the common-minimizer equivalence. Regime-only oracle Oπ is not an oracle over
  state-adaptive policies; Vπ=0 cannot rule those out. Remaining-horizon loss is
  not assumed nonnegative for arbitrary affine transients. No unknown-prior or
  minimax conclusion is inferred from balanced-prior separation.
- The analytic Z0=1/Z1=(1+Y)/2 example proves failure of independent-mixture risk
  transport, not failure of BOTH conditions in the executed AND gate, not failure
  for the registered grid, and not superiority over cost-optimal branch actions.

The previous separate deterministic enumeration/nine quadrature fixtures are
retained, not rerun as extra experiments. They share the audited program-risk
primitive and are not an independently reimplemented risk oracle or expert proof.

## Evidence preserved and publication judgment

All128 designs:120 feasible,8 infeasible;44 harmful,32 beneficial and44 zero
transfers. Both admission counts113. Balanced-prior(pi=.5) certificates29/128,
all at scales512/2048, with99 uncertified cases; minimax0/128. No new endpoint
replaces the original primary endpoints. The low-running-mean32 negative result,
23/256 adverse cell means, identical ablation and fixedq2 pooled advantage remain.

The substantive addition to this PAPER is the explicit continuing-learner
mechanism/cost/comparator connection and its complete original-grid diagnosis.
None of the three candidates warrants advertising a new foundational probability
or decision theorem. The bounded corpus does not establish universal publication
priority; the revision makes no such claim. No new experiment is needed to state
these proved, narrowly scoped consequences. A claim of incremental executed-gate
benefit would require a separate prospective protocol and human-approved budget.
