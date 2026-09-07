## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free algorithm and theorem derivation
- Origin Date: 2026-09-07
- Verification Status: ABSTRACT DELAYED LINEAR-ALIGNMENT THEOREM PROVED; PURSUIT REALIZATION OPEN
- Version Label: optimistic_delayed_factor_lyapunov_v1

# Optimistic delayed factor learning inside the Lyapunov controller

## Why this is the final statistical interface candidate

A hard lower-confidence shield can reject every useful refresh until all local
models are already known.  An unrestricted plug-in critic avoids that deadlock
but its dynamic-oracle term depends on uncertainty for both the selected action
and an unobserved comparator.  A locally linear optimistic model is the clean
middle ground: uncertainty causes targeted exploration, the communication
queue prices that exploration, and the Lyapunov index still jointly chooses
the cache edge and delayed packet weight.

This is not a new objective layered on top of the previous algorithm.  It is a
statistically instantiated version of the same launch decision

\[
 u_p=(a_p,\alpha_p),\qquad
 a_p\in\{\varnothing\}\cup E_p(i),\quad
 0\leq\alpha_p\leq\bar\alpha,
\]

and the same composite Lyapunov function `V F + H + Q^2/(2 nu)`.

## Delayed local alignment model

For every launch `p`, the centralized local-factor critic produces predictable
features `x_p(a) in R^d` for the null action and at most `Delta_i` cache
refreshes.  Assume `||x_p(a)||_2<=L_x`.  The scalar feedback observed when the
chosen trajectory packet returns is

\[
 Y_p=A_p(a_p)+\xi_p,
 \qquad A_p(a)=x_p(a)^\top w_*+b_p(a),
\tag{1}
\]

where `A_p(a)` is the conditional owner-gradient alignment used in the exact
Lyapunov drift, `xi_p` is conditionally `sigma`-sub-Gaussian at receipt, and
`|b_p(a)|<=epsilon_p^app(a)` is a declared approximation/drift allowance.  The
reference vector used to form `Y_p` must be computed from data completed before
launch; its error is part of `epsilon^app`.  No receipt observation may be
inserted retroactively into its launch context.

The receipt delay is allowed to depend on launch history, owner, action, and
service state, but it must be non-informative about the unrealized innovation:
conditional on the launch filtration and chosen action, revealing whether a
packet has returned cannot select on `xi_p`.  This condition is automatic for
externally generated service times independent of trajectory noise.  Without
it, the completed-packet regression can be selection biased and (3) is not a
valid confidence radius.

Let `C_p` contain launches whose feedback has arrived before launch `p`, and

\[
 \Lambda_p=\lambda I+\sum_{k\in C_p}x_k(a_k)x_k(a_k)^\top,
 \qquad
 \widehat w_p=\Lambda_p^{-1}
       \sum_{k\in C_p}x_k(a_k)Y_k.
\tag{2}
\]

Assume `||w_*||<=S`.  A valid radius is

\[
 \begin{aligned}
 \beta_p={}&\sigma\sqrt{2\log\frac{
       \det(\Lambda_p)^{1/2}}
       {\det(\lambda I)^{1/2}\delta_p}}
       +\sqrt\lambda S
       +\left(\sum_{k\in C_p}
                    [\epsilon_k^{app}(a_k)]^2\right)^{1/2},\\
 r_p(a)={}&\beta_p\|x_p(a)\|_{\Lambda_p^{-1}}
             +\epsilon_p^{app}(a).
\end{aligned}
\tag{3}
\]

The last term in `beta_p` follows from
`||Phi^T b||_(Lambda^-1)<=||b||_2`; it is not silently treated as stochastic
noise.  Standard self-normalized martingale concentration gives, with the
allocated probability,

\[
 |x_p(a)^\top\widehat w_p-A_p(a)|\leq r_p(a)
\tag{4}
\]

simultaneously over the finite local action set.  A summable allocation over
launches makes the event simultaneous in time.

## Optimistic Lyapunov decision

Define

\[
 A_p^U(a)=\min\{A_{\max},
                  x_p(a)^\top\widehat w_p+r_p(a)\}.
\tag{5}
\]

Here clipping and gradient bounds supply the public physical bound
`A_p(a)<=A_max`.  The cap is essential under persistent model
misspecification: it prevents a growing historical-bias radius from making the
largest possible packet gain, and hence the communication queue cap, diverge.
It does not invalidate optimism because the true alignment is already below
`A_max`.

Substitute `A_p^U(a)` for the alignment in the exact index

\[
 J_p(a,\alpha;A)
 =-m_p(a;A)\alpha+\frac{C_p(a)}2\alpha^2
  -B_p(a)+Q_pc_p(a).
\tag{6}
\]

For every candidate, compute the same scalar minimizer

\[
 \widehat\alpha_p(a)=
 \Pi_{[0,\bar\alpha]}
 \left(\frac{m_p(a;A_p^U(a))}{C_p(a)}\right),
\]

then select the candidate with minimum optimistic index.  The optimism is
one-sided in exactly the useful direction: larger alignment lowers predicted
learning drift.  It does not override the cache reset, delay penalty, or
communication queue price.

## Theorem 1: selected-action Lyapunov regret

On event (4), for every feasible launch-measurable comparator pair
`(a_p^o,alpha_p^o)`, the selected pair satisfies

\[
 J_p(a_p,\alpha_p;A_p)-J_p(a_p^o,\alpha_p^o;A_p)
 \leq 2V\bar\alpha r_p(a_p).
\tag{7}
\]

### Proof

Equation (4) and the physical cap imply `A_p(a)<=A_p^U(a)` and
`A_p^U(a)-A_p(a)<=2r_p(a)`.  Because `alpha>=0`, the optimistic index is no
larger than the true index for every action-weight pair.  Exact minimization
therefore gives

\[
 \begin{aligned}
 J_p(a_p,\alpha_p;A_p)
 &\leq J_p(a_p,\alpha_p;A_p^U)
       +2V\bar\alpha r_p(a_p)\\
 &\leq J_p(a_p^o,\alpha_p^o;A_p^U)
       +2V\bar\alpha r_p(a_p)\\
 &\leq J_p(a_p^o,\alpha_p^o;A_p)
       +2V\bar\alpha r_p(a_p).
 \end{aligned}
\]

Unlike a symmetric plug-in comparison, (7) contains no confidence radius for
the unselected comparator action.

## Theorem 2: bounded feedback delay

Suppose every chosen alignment observation returns within at most `D` later
launches, `lambda>=L_x^2`, and first ignore the explicit approximation terms in
(3).  Then

\[
 \sum_{p=1}^N\|x_p(a_p)\|_{\Lambda_p^{-1}}
 \leq
 \sqrt{2N(D+1)d\log\left(1+
            \frac{NL_x^2}{\lambda d}\right)}.
\tag{8}
\]

### Proof

Partition launches by their residue modulo `D+1`.  When launch `p` is reached,
every earlier selected context in its own residue class has returned.  Hence
`Lambda_p` dominates the ridge Gram matrix formed only by earlier contexts in
that residue, and inverse order reverses.  Apply the elliptical-potential
lemma within each residue and sum to obtain

\[
 \sum_{p=1}^N\|x_p(a_p)\|_{\Lambda_p^{-1}}^2
 \leq 2(D+1)d\log(1+NL_x^2/(\lambda d)).
\]

The condition `lambda>=L_x^2` makes every norm at most one.  Cauchy--Schwarz
proves (8).

Combining (7)--(8) with the exact discrete drift theorem adds the normalized
statistical term

\[
 2\bar\alpha\beta_N
 \sqrt{\frac{2(D+1)d}{N}
       \log\left(1+\frac{NL_x^2}{\lambda d}\right)}
\tag{9}
\]

to the average stationarity bound, plus the normalized sum of the explicit
current approximation terms.  Thus random computation delay need not be the
primary wall-clock metric: it has a direct statistical role because it delays
learning which cache edge has positive drift value.

## Complexity and resource accounting

The ridge state is shared across the local edge model.  A completed packet
requires one Sherman--Morrison rank-one inverse and log-determinant update in
`O(d^2)`.  Evaluating at most `Delta_i+1` confidence widths costs
`O(Delta_i d^2)`, followed by the existing scalar closed forms and
`O(Delta_i)` scan.  With a fixed small local feature dimension,
the cost is independent of the total number of agents.  There is no Hessian,
global covariance over agents, graph QP, or combinatorial subset search.

Because `A_p^U<=A_max`, the maximum effective packet gain in the existing
queue-cap lemma is finite even when an explicitly retained misspecification
term does not vanish.  Persistent approximation error may leave a nonzero
learning floor in (9), but it cannot by itself authorize unbounded average
communication.

The trajectory used to produce `Y_p` is the same fully charged owner rollout
that supplies the learning packet.  Counterfactual features are critic
computations on completed replay; they are not free environment samples.
Only a selected non-null refresh consumes policy bytes.  Compute time and
memory remain measured secondary outcomes.

## Pursuit realization contract

The theorem can be instantiated without claiming a distribution-free neural
confidence set by freezing a local neural feature encoder within each fitting
epoch and fitting a finite-dimensional linear factor head only from subsequently
completed packets.  Candidate contexts, ridge state, and reference gradients
are then predictable.  Encoder drift, Bellman approximation, controlled-kernel
shift from `pursuit_controlled_kernel_bridge_20260907.md`, and factor residual
all enter `epsilon^app` explicitly.

Before any GPU efficacy experiment, a CPU interface qualification must verify:

1. launch/receipt causality and delayed covariance updates;
2. nonzero signed alignment feedback and counterfactual feature diversity;
3. empirical coverage of the declared linear-head radius on held-out completed
   packets, reported as calibration rather than a distribution-free theorem;
4. bounded degree and measured `O(Delta)` candidate count;
5. exact policy-byte charging and a nontrivial communication queue;
6. an oracle-value ceiling against age, mismatch, periodic, fixed-local, and
   strong online baselines before training outcomes are used to tune gates.

Failure of this interface stops the Pursuit efficacy run.  Passing it permits
a separately preregistered pilot; it does not turn the tabular confirmation
into standard-MARL evidence.

## Bounded novelty statement

The self-normalized ridge confidence set is inherited from
[Abbasi-Yadkori, Pal, and Szepesvari (NeurIPS 2011)](https://proceedings.neurips.cc/paper_files/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html),
and delayed linear-bandit feedback is established prior art, including
[Vernade et al. (ICML 2020)](https://proceedings.mlr.press/v119/vernade20a.html).
Virtual queues are likewise not new to constrained reinforcement learning; see,
for example,
[Wei, Liu, and Ying (AISTATS 2022)](https://proceedings.mlr.press/v151/wei22a.html).
None of these ingredients is claimed as a standalone contribution.

The prospective contribution is the coupled object: an optimistic estimator
of the signed learning value of refreshing one teammate policy cache, embedded
inside the same Lyapunov drift that selects the refresh edge and delayed owner
packet weight, with a pathwise policy-byte queue and a controlled Markov-game
kernel.  It differs from asynchronous cooperative linear-MDP data sharing
[Min et al. (ICML 2023)](https://proceedings.mlr.press/v202/min23a.html), where
homogeneous learners communicate data for one MDP, and from
[dynamic coordination graphs](https://proceedings.mlr.press/v157/siu21a.html),
whose graph factorizes values/actions rather than deciding which training-time
policy version a rollout worker receives.  This distinction remains a bounded
novelty hypothesis, not a claim of exhaustive priority; it requires a fresh
systematic check before manuscript submission.

Three 2026 asynchronous-RL preprints make the boundary sharper.  [GAC](https://arxiv.org/abs/2603.01501)
projects updates to control alignment between consecutive stale gradients;
[staleness--learning-rate scaling laws](https://arxiv.org/abs/2607.01083)
derive conditions coupling one policy's rollout lag and step size; and
[SAT](https://arxiv.org/abs/2607.18722) adapts a trust region using sampled
policy mismatch.  Therefore neither gradient alignment, stale-packet weighting,
nor asynchronous-RL stabilization is a defensible novelty claim by itself.
These works concern one asynchronously trained policy rather than distinct
interacting actor blocks, do not select a directed teammate policy-cache edge,
and do not enforce an average policy-byte budget through the learning
Lyapunov drift.  The paper must demonstrate the value of that full coupling,
not merely outperform an uncorrected asynchronous optimizer.
