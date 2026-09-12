# T-049 marginal-preserving bridge to standard Markov tasks

## Purpose

T-049 replaces the scalar task proxy used by T-045A with exact vector objects
from a public finite-state reinforcement-learning task. For a frozen
epsilon-soft evaluation policy, it computes the projected temporal-difference
drift, the stationary edge-gradient process, and every finite-lag covariance
matrix used by the T-048 scheduled Polyak--Ruppert identity.

## Exact task covariance

Let \(S_t\) be the regenerative fixed-policy Markov chain and let \(\xi_t\)
be the projected temporal-difference update at its fixed point. For state
\(s\) and successor \(u\), define

\[
 F_{su}=\mathbb E[\xi_t\mathbf1\{S_{t+1}=u\}\mid S_t=s],
 \quad m_s=\sum_uF_{su}.
\]

If \(d\) is the stationary distribution and \(P\) is the regenerative
transition matrix, then

\[
 K_0=\mathbb E[\xi_t\xi_t^\top],
\]

and for \(k\ge1\),

\[
 K_k=\mathbb E[\xi_{t+k}\xi_t^\top]
 =\left[
 \sum_{s,u}d_sF_{su}(P^{k-1}m)_u^\top
 \right]^\top.                                         \tag{1}
\]

Equation (1) follows by conditioning on the edge
\((S_t,S_{t+1})\) and then applying the Markov property. It retains
nonreversibility and matrix-valued temporal correlation.

## Marginal-preserving cross-agent coupling

For each independent learning replicate, draw a common standard-task
trajectory \(U\), private trajectories \(V_i\), and switches
\(Z_i\sim\operatorname{Bernoulli}(\sqrt\rho)\). Actor \(i\) uses \(U\) when
\(Z_i=1\) and \(V_i\) otherwise. Every actor trajectory therefore has exactly
the upstream task law, rather than only matching its first two moments.

For prefix participation sets \(S_s,S_r\), direct expansion gives

\[
 \operatorname{Cov}(\bar\xi_s,\bar\xi_r)
 =\left[
 \rho+(1-\rho)\frac{|S_s\cap S_r|}{q_sq_r}
 \right]K_{s-r}.                                      \tag{2}
\]

Indeed, two distinct actors share \(U\) with probability
\((\sqrt\rho)^2=\rho\); the same private trajectory contributes only for the
same actor. For prefixes, (2) is the exact T-047 multiplier
\(\rho+(1-\rho)/\max(q_s,q_r)\).

The switch is fixed within a learning trajectory, so temporal correlation is
not silently destroyed. It is independently redrawn across experimental
replicates. Probe replicates and learning replicates are disjoint.

## Algorithmic and empirical role

The T-049 bridge permits an exact CPU phase scan on standard Gymnasium kernels
without claiming that a scalar second-largest-eigenvalue-modulus proxy equals
the vector task risk. It also defines a sampled coupling whose actor marginals
are unchanged. The upcoming static scan must still be independently
preregistered before evaluating any schedule value, and a sampled controller
remains unauthorized until that scan passes its practical-value gates.
