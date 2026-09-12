# Marked-Poisson wall-clock gate

Status: a bounded oracle-model conversion, a one-packet barrier phase and a
fresh-query separation from fully utilized frozen-policy shadow batching are
closed.  Broader partially asynchronous and speculative comparators remain
outside the separation.

## Declared asynchronous oracle

Applied update events follow an exogenous marked Poisson process.  The total
event rate is

\[
\Lambda=\sum_i\lambda_i,
\qquad
p_i=\lambda_i/\Lambda.
\]

The mark `I_k` identifies the policy block returned at event `k`.  Conditional
on the pre-application history, marks are iid with probabilities `p_i` and
completion timing is independent of the trajectory innovation.  The packet
was queried at a predictable version `b_k` satisfying `0<=k-b_k<=D`.  This is a
standard delayed-oracle model, not a claim about arbitrary simulator timing.
Variable episode length, trajectory-dependent service and unbounded version
lag are outside this theorem.

Let `tau_K` be the elapsed time of the `K`-th applied event.  Then

\[
\tau_K\sim\operatorname{Gamma}(K,\Lambda),
\qquad
\mathbb E\tau_K=K/\Lambda. \tag{1}
\]

Consequently the event-time rate-balanced descent coefficient `c_star` from
`clocked_async_mpg_rate_balanced_steps.md` becomes

\[
\mathsf R_{\rm async}=\Lambda c^\star \tag{2}
\]

per expected unit time.  Combining (2) with the Markov packet and bias/noise
terms gives an expected stopping-time complexity simply by replacing `K` by
`Lambda*E[tau_K]`.  A high-probability clock statement can use standard Gamma
concentration, but has not yet been added to the main theorem.

## One-packet synchronous barrier

For the first bounded comparison, synchronous round `r` launches one packet
per agent and waits for all of them.  With independent exponential service
times of rates `lambda_i`, its expected round length is exactly

\[
\mathbb E T_{\max}
=\sum_{\varnothing\ne A\subseteq[n]}
\frac{(-1)^{|A|+1}}{\sum_{i\in A}\lambda_i}. \tag{3}
\]

For equal rate `lambda`, (3) is `H_n/lambda`, where `H_n` is the harmonic
number.  If a simultaneous policy-gradient step is certified with step
`1/L_sync`, its deterministic descent coefficient per expected second is

\[
\mathsf R_{\rm sync}
=\frac{1}{L_{\rm sync}\mathbb E T_{\max}}. \tag{4}
\]

The certified coefficient ratio is `R_async/R_sync`.  This comparison charges
all launched packets but allows only the registered one packet per agent in a
barrier round.

## Explicit symmetric rate--coupling phase

Take equal service rates, diagonal block smoothness `L_0`, symmetric
cross-smoothness `kappa`, and

\[
L_\Sigma=L_0+(n-1)\kappa.
\]

The rate-balanced scale is the positive root of

\[
nL_0c+(1+\delta)D^2nL_\Sigma^2c^2=1. \tag{5}
\]

Using `L_sync=L_Sigma`, the asynchronous certificate is faster than the
one-packet barrier certificate exactly when

\[
\frac{L_0}{L_\Sigma H_n}
+\frac{(1+\delta)D^2}{nH_n^2}<1. \tag{6}
\]

Equation (6) is a nonempty, interpretable phase boundary.  More agents and
barrier order-statistics enlarge the asynchronous region; interaction-weighted
event delay shrinks it.  At large delay the transition is on the order of
`D=H_n*sqrt(n/(1+delta))`.  This is a certified-bound comparison, not an exact
risk ordering.

## Strong-comparator obstruction

Equation (6) is insufficient for a paper claim because a stronger synchronous
implementation can keep fast actors working after their first completion,
collect additional packets at the frozen round policy and use them as a larger
batch.  Such shadow batching converts nominal waiting time into variance
reduction.  On separable linear-quadratic problems it can match the statistical
value of asynchronous samples, so no universal asynchronous advantage over
fully utilized synchronization is possible from throughput alone.

The required fresh-query lower/separation family is now proved in
`clocked_async_mpg_fresh_query_separation.md`.  Even an unlimited deterministic
batch at the same frozen policy cannot distinguish two smooth potential
instances with the same initial gradient; a second fresh asynchronous query
can.  The witness fully charges service and has a strict expected-time gap
against a fixed all-agent barrier.

This closes the bounded analytic gate only.  A partially asynchronous learner
that releases converged blocks or applies within-round updates is no longer a
global-barrier method, and a speculative multi-policy sampler changes the data
law.  Both remain valid empirical comparators where implementable.  The
one-state continuous-action witness also cannot substitute for a positive
multi-state Markov-game confirmation.

## Validation

The exact inclusion--exclusion barrier time, coefficient conversion and
symmetric phase are implemented without sampled outcomes.  Five tests verify
the harmonic-number identity, the zero-delay closed form, monotone loss of
certified advantage with delay, finite heterogeneous-rate coefficients and
strict input validation.
