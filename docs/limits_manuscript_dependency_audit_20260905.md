# Limits-manuscript dependency audit

Date: 2026-09-05

Status: **conditional GO for a narrowly scoped parallel-Markov-learning
theory manuscript after two remaining minimax links are proved; NO-GO for
relabelling it as an asynchronous strategic-MARL algorithm paper.**

## The one coherent statistical question

Consider a learner with a finite message budget, a finite environment budget
and delayed usability.  It can collect observations from a variable number of
Markov streams at variable strides.  Cross-stream dependence controls the
value of adding agents, but the dependence regime is unknown and must be
learned using the same finite resources.

The coherent candidate question is:

> What is the minimax price of adapting parallelism to an unknown dependence
> regime, and when is identification amortized by the downstream learning
> gain before delay and dual budgets expire?

This question joins the positive and negative results through one normalized
quantity.  Let \(j\) index a separated dependence regime, \(V_j^*(s)\) be the
known-regime oracle risk at budget-ray scale \(s\), and let

\[
\mathfrak U(s)
=
\inf_{\pi\in\Pi_{\rm pred}}
\max_j \frac{R_j^\pi(s)}{V_j^*(s)} .
\]

The policy class contains common unknown-regime predictable rules and includes
history-dependent participation and stride decisions.  It does not reveal the
regime to the learner.

On a compact separated, certified-mixing class, the existing explore--test--
commit rule gives the valid upper bound

\[
1\le \mathfrak U(s)
\le 1+O(\log s/s).
\]

Its lower side \(\mathfrak U(s)\ge1\) currently follows only because a learner
that knows the regime has more information.  It is not an
information--learning lower bound and does not match the controlled-belief
occupation value.  When the mixing class reaches arbitrarily close to one,
uniform regime
identification has a nonzero error floor.  Thus the separated positive theorem
and the unrestricted negative theorem identify a plausible shared boundary,
but they do not yet solve its complete minimax frontier.

A more appropriate target quantity is the safety-constrained oracle regret

\[
\mathcal V_B(\epsilon)=
\inf_{\pi\in\Pi_B:\,
\sup_\nu[R_\nu^\pi-R_\nu^{\rm all}]_+\le\epsilon}
\sup_\nu\left(R_\nu^\pi-R_\nu^\star\right).
\]

## Dependency graph

```text
Gaussian common-factor Markov observation model
  |
  +-- spatial rotation + hypothesis-specific Kalman innovations
  |      |
  |      +-- AC-7 stopped adaptive KL chain rule
  |              |
  |              +-- two-direction data-processing lower bound
  |              |       |
  |              |       +-- controlled-belief occupation lower bound
  |              |
  |              +-- finite-budget necessary identification threshold B_N
  |
  +-- Bhattacharyya fixed-block upper test
  |      |
  |      +-- sufficient identification cost
  |
  +-- exact finite-budget downstream risk catalogue
  |      |
  |      +-- theorem-derived strict fallback and S_mean safety slack
  |      +-- risk sensitivity V*(s-c)-V*(s) <= Lc/s^2
  |
  +-- compact separation: lambda <= 1-gamma, Delta_theta >= Delta_min,
  |      nonzero oracle gap, bounded delay, finite stable catalogue
  |      |
  |      +-- B_N <= B_S <= class-constant threshold upper bound
  |      +-- T-039 first-order oracle-normalized matching
  |
  +-- lambda -> 1 endpoint + TV contraction + Le Cam
         |
         +-- AC-8 unrestricted unknown-mixing impossibility
```

## Claim ledger

| Object | Defensible status | Boundary that must remain explicit |
|---|---|---|
| AC-7 adaptive change of measure | proved for the stationary Gaussian common-factor model with known mixing | not a general Markov-game likelihood theorem |
| adaptive occupation information lower bound | proved as a relaxation induced by every correct policy | no implemented solver or finite-budget equality to the optimum |
| unknown-mixing result | proved negative when the class approaches unit mixing | positive results require known mixing or a public gap from one |
| necessary/sufficient threshold sandwich | proved on a compact separated class | constants require a nonvanishing oracle gap and uniform conditioning |
| theorem-derived fallback | proved for a supplied exact finite-budget risk catalogue | controls expected `S_mean`, not pathwise no-harm |
| first-order adaptation matching | proved as \(1+O(\log s/s)\) on the separated class | not second-order controlled-occupation matching |
| learning-risk-to-testing reduction | proved for predictable sensing followed by finite fresh deployment | requires a separated finite deployment catalogue and noncancellation; does not cover continuous interleaved learning |
| safety-constrained occupation achievability | open | the current ETC upper bound does not match the adaptive occupation program |
| minimum safety slack \(\epsilon^\star(B)\) | open | strict no-harm can force fallback and has no nontrivial frontier theorem |
| delayed affine Markov-TD transfer | formal mechanism evidence exists | does not make the observation lower bound a general TD lower bound |
| full matrix delayed stability coupling | open AC-10 | cannot claim general adaptive \((q,b,\eta)\) convergence |
| nonlinear/general observation extension | open AC-11 | fixed-gradient covariance is not nonlinear training convergence |
| SDDE-to-discrete bridge | open AC-12 | SDDE may be interpretation only |

## Why this is not asynchronous strategic MARL

The present model has multiple Markov observation streams, not distinct agents
with coupled strategic policy blocks in one Markov game.  The action
\((q_t,b_t,\eta_t)\) controls the statistical experiment; \(\eta_t\) does not
alter the AC-7 observation law.  The delay \(D\) reduces the usable post-probe
horizon.  It is not a random packet-arrival process with stale teammate
policies.

There is no theorem here for:

- CTDE policy-gradient convergence;
- mixed policy versions inside a joint trajectory;
- asynchronous actor or critic applications;
- decentralized execution under stale teammate policies;
- Nash, coarse-correlated-equilibrium or Markov-potential-game gap.

Changing terminology would not supply these missing objects.  Therefore this
package cannot honestly satisfy a fixed scope of “asynchronous MARL update
control” without a new theorem and new evidence.

## Experimental evidence and its exact scope

EXP-016B is strong formal evidence for the stated threshold mechanism:

- implementation frozen before formal outcomes;
- 192 independent paired formal seeds;
- 2,752,512 rows;
- all P1--P12 gates passed;
- byte-identical independent reproduction;
- 54.43% relative improvement in the primary Gaussian threshold population;
- 10.85% relative improvement in the affine delayed Markov-TD transfer layer;
- 77/96 registered scenarios met the directional/practical condition.

These outcomes validate “identification can be feasible before adaptation is
learning-value positive.”  They do not validate a standard nonlinear MARL
controller.  The later standard-task feasibility program was correctly
stopped: its best registered task supplied only 1.404% oracle adaptation value
and the aggregate supplied 0.704%, below the frozen 5% gate.

This is not a power issue.  It is a lack of strong-baseline headroom on the
tested standard tasks.

## Publication decision

### ICML 2027 asynchronous-MARL paper

**NO-GO on current evidence.**  The theorem object, algorithm and benchmarks
do not implement asynchronous strategic MARL.  Adding that wording would
overclaim, while another controller search has already failed its entry gate.

### Narrow theory / signal-processing manuscript

**Conditional GO only after the two remaining central minimax links above are
closed.**  A coherent
working title is:

> **When Is Parallel Markov Learning Worth It? Correlation, Delay, and the
> Minimax Cost of Adaptation**

The contribution contract is one story:

1. dependence changes the attainable value of parallel Markov data;
2. learning that dependence consumes the same finite resources as learning;
3. a controlled change-of-measure lower-bounds that opportunity cost;
4. a separated-class rule is first-order oracle matching;
5. uniform adaptation becomes impossible near the non-mixing boundary;
6. formal Gaussian and affine experiments verify the predicted finite-budget
   learning-value phase.

This is a plausible TSP/theory submission package if the missing links close.
It is not an “absolute accept” guarantee and should not be advertised as one.

The present upper algorithm also needs redesign.  Individual all-agent
observations have positive regime KL, so a separately paid fixed probe is not
information-theoretically necessary.  The appropriate candidate is a
**baseline-as-sensing anytime likelihood switch**: execute the safe baseline,
update the exact Kalman likelihood ratio from the observations that baseline
learning already produces, and switch only after directional evidence and
remaining-budget value jointly qualify.  Its unavoidable cost is the
high-regime oracle opportunity lost while waiting, not a fictitious mandatory
sensor charge.  This candidate remains parallel Markov stochastic
optimization; it does not restore asynchronous strategic-MARL scope.

## Remaining authorized CPU work

1. retain the proved learning-excess-to-testing reduction only for finite
   fresh deployment; do not extend it to interleaved online learning without
   a new noncancellation or coupled-law argument;
2. derive and analyze the baseline-as-sensing anytime likelihood switch,
   including its occupation cost and remaining-budget stopping boundary;
3. formulate the joint two-law occupation program imposed by one common
   predictable policy kernel; prove that it lower-bounds expected-safe rules;
4. characterize strict no-harm feasibility or the minimum necessary
   \(\epsilon^\star(B)\);
5. prove or explicitly assume every risk-sensitivity and compactness constant
   used by T-039; remove any circular definition involving the oracle gap;
6. reconcile the older `theorem_derived_fallback.md` statement “AC-9 not
   closed” with the later split into AC-9a--d;
7. audit the occupation lower bound's flow and stopping constraints as a
   theorem, not only prose;
8. assemble a contribution-led manuscript and appendix in the narrow scope;
9. run a fresh citation-integrity pass and the existing theorem regression
   suite before delivery.

No new efficacy experiment, GPU job or HPC4 work is authorized by this audit.
