# T-044 Polyak--Ruppert averaged phase law after T-043A

## Decision

T-043A and its resource-normalized AUC diagnostic both show zero adaptation
value for the frozen constant-step last-iterate family. The failure is not a
reason to tune those tasks. It identifies an estimator mismatch: last-iterate
constant-step risk has a steady-state noise floor, so fewer updates can act as
implicit early stopping and reverse the intended resource comparison.

T-044 retains the same correlation--delay--budget mechanism but changes the
theorem-facing estimator to Polyak--Ruppert (PR) averaging. PR averaging is
classical and is not claimed as novelty. Its role is to align the achievable
upper bound with T-038's sample-information lower bound, for which additional
observations cannot hurt.

## Exact finite-horizon identity

For the delayed additive recursion

\[
e_{t+1}=e_t-\eta A e_{t-D}+\eta\xi_t,
\]

let \(C\) be the T-037 lifted companion, \(S\) select the current iterate,
and \(J=S^\top\) inject innovations. For burn-in \(b<N\), define

\[
\bar e_{b:N}=\frac1{N-b}\sum_{t=b+1}^{N}e_t.
\]

Its exact mean is

\[
\mathbb E\bar e_{b:N}
=\frac1{N-b}\sum_{t=b+1}^{N}SC^t x_0.
\]

The impulse multiplying innovation \(\xi_s\) is

\[
H_s^{(b,N)}=
\frac{\eta}{N-b}
\sum_{t=\max\{b+1,s+1\}}^{N}SC^{t-1-s}J.
\]

For arbitrary stationary lag covariance \(K_{s-r}\),

\[
\operatorname{Cov}(\bar e_{b:N})
=\sum_{s,r=0}^{N-1}H_s^{(b,N)}K_{s-r}H_r^{(b,N)\top}.
\]

This is a finite-horizon identity; it uses neither Gaussianity nor an iid
replacement. Delay remains inside \(C\), and the full cross-agent Markov lag
covariance remains inside \(K\).

## Message-budget phase

For exchangeable agents, the leading PR covariance carries the exact factor

\[
\rho+\frac{1-\rho}{q}.
\]

Under message budget \(B\) and per-update cost \(h+q\),
\(N(q)\asymp B/(h+q)\). Up to task- and mixing-dependent constants, the
leading risk is therefore

\[
\frac1B\left(\rho+\frac{1-\rho}{q}\right)(h+q).
\]

For \(0<\rho<1\), its continuous optimum is
\(q^\star=\sqrt{h(1-\rho)/\rho}\), clipped to the public catalogue. Thus the
same estimator has a large-q optimum under weak correlation and `q=1` under
perfect correlation. This is the clean asymptotic participation phase that
constant-step terminal risk failed to expose on T-043A.

## Claim boundary

The exact identity is proved for additive Markov innovations. For
multiplicative TD, the T-042 weighted Poisson decomposition must be applied to
the PR readout and its two increment terms still require small-gain
absorption. A future standard-task scan may proceed only under a separate
preregistration with PR averaging frozen before evaluation. T-043A remains a
permanent failure and cannot be relabeled as evidence for T-044.

The closest recent comparator already studies inference for PR-averaged
linear stochastic approximation under Markov noise. Therefore the paper's
novelty must remain the sharp multi-agent correlation--participation phase,
delay/resource accounting, predictable minimax lower bound, and adaptation
cost—not PR averaging itself.
