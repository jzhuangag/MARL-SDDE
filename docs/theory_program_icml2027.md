# ICML 2027 theory program: correlation-limited collaboration

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: analysis
- Origin Date: 2026-07-30
- Verification Status: PARTIALLY_VERIFIED
- Version Label: research_program_v1

## Autonomous decision

The generic claim “delay-adaptive multi-agent stochastic approximation under
Markov data” is not sufficiently novel. DASA already proves delay adaptivity,
average-delay dependence, and \(N\)-fold speedup under independent agent
Markov chains; AsyncMATD proves linear speedup for asynchronous multi-agent TD.
Federated RL under Markov sampling also established linear speedup earlier.

The viable ICML-level gap is narrower and sharper:

> Cross-agent dependence invalidates the usual \(N\)-fold speedup law. Under a
> fixed communication budget, the useful number of agents is a
> correlation-, optimization-state-, and delay-dependent control variable.

The paper should not claim novelty for delay, Markov sampling, participation,
or Lyapunov analysis separately. Its novelty must be the combined theorem that
replaces \(N\) by a cross-agent **effective participation** quantity, proves a
finite-budget participation phase transition, and gives an implementable
low-complexity controller with a near-oracle guarantee.

## Recommended title and scope

Primary working title:

**Beyond Linear Speedup: Correlation-Limited Collaboration in Delayed
Multi-Agent Markov Learning**

Algorithm-focused alternative:

**Correlation- and State-Adaptive Participation for Delayed Multi-Agent
Markov Learning**

The first is stronger if the theorem is completed; the second should be used
only if an observable controller passes strong baselines on nonlinear tasks.
“MARL” should not appear in the title unless the final experiments and theorem
cover more than centralized policy evaluation.

## Model

Let a predictable subset \(S_k\subseteq[N]\) contribute delayed stochastic
directions:

\[
\theta_{k+1}=\theta_k-\eta_k\frac{1}{|S_k|}
\sum_{i\in S_k}G_i(\theta_{k-\tau_{i,k}},Z_{i,k}).
\]

Assume:

1. the mean operator is strongly monotone and Lipschitz (or use a separate
   nonconvex stationarity theorem);
2. the joint process \(Z_k=(Z_{1,k},\ldots,Z_{N,k})\) is geometrically mixing,
   but its coordinates need not be independent;
3. delays are predictable and bounded for the base theorem, with an
   average-delay extension;
4. the chosen subset is predictable with respect to past observations;
5. stochastic directions have bounded conditional moments.

For \(S\), define the long-run covariance functional

\[
\Omega(S)=\sum_{h=-\infty}^{\infty}
\mathbb E\!\left[
\bar\xi_S(Z_0)^\top P\,\bar\xi_S(Z_h)
\right],
\qquad
\bar\xi_S=\frac1{|S|}\sum_{i\in S}\xi_i.
\]

This quantity captures temporal Markov correlation and cross-agent
correlation simultaneously. It is the correct replacement for the independent
noise term \(\sigma^2/|S|\).

For an exchangeable global-plus-cluster factor model,

\[
\Omega(S)=
\Omega_g+
\Omega_c\sum_{r=1}^{C}\left(\frac{n_r(S)}{|S|}\right)^2+
\frac{\Omega_\epsilon}{|S|}.
\]

Thus global common noise is irreducible, balanced cluster selection controls
the cluster term, and only the idiosyncratic component gives linear variance
reduction.

## Theorem ladder

### Theorem 1: delayed Lyapunov recursion

Construct a Lyapunov--Krasovskii functional

\[
\mathcal V_k=
\|\theta_k-\theta^\star\|_P^2+
\lambda\sum_{j=k-D}^{k-1}
\|\theta_{j+1}-\theta_j\|^2.
\]

The target conditional recursion is

\[
\mathbb E_k[\mathcal V_{k+1}]
\le
(1-c_1\mu\eta_k)\mathcal V_k
+c_2\eta_k^2\Omega(S_k)
+c_3L^2\eta_k^2\Psi(\tau_k)\mathcal V_k
+R_{\rm mix}(k),
\]

under an explicit stability condition on \(\eta_k,D,L,\mu\). The proof must
handle subset choice using only past data; otherwise selection bias invalidates
the martingale/mixing step.

### Theorem 2: correlation-limited speedup

For fixed \(q\) and constant stable step size, derive

\[
\mathbb E[\mathcal V_K]
\le
\exp[-c\mu\eta K/\chi_D]\,\mathcal V_0
+\frac{C\eta}{\mu}\Omega(q)
+R_{\rm mix},
\]

where \(\chi_D\) is an explicit delay factor. Define

\[
N_{\rm eff}(q)=\frac{\Omega(1)}{\Omega(q)}.
\]

Then \(N_{\rm eff}(q)\le q\), with strict saturation whenever the common
long-run covariance is nonzero. This theorem is the core “beyond linear
speedup” result and directly exposes the independence assumption behind prior
\(N\)-fold results.

### Theorem 3: finite-budget participation phase transition

With message-equivalent cost \(c(q)=c_0+q\) and budget \(B\), only

\[
K(q)=\left\lfloor\frac{B}{c_0+q}\right\rfloor
\]

updates are available. Substitute \(K(q)\) into Theorem 2 and minimize the
bound over \(q\) and \(\eta\). Prove:

- in a bias-dominated transient, increasing \(q\) can hurt because it reduces
  the number of contraction steps;
- in a variance-dominated regime with independent agents, larger \(q\) helps;
- with common correlation, the variance benefit saturates and the optimal
  \(q^\star\) is finite;
- \(q^\star\) changes across optimization state and dependence strength,
  yielding explicit sufficient conditions or thresholds.

This formalizes the EXP-006A phase diagram rather than merely fitting it.

The scalar, no-delay special case should be stated as an exact proposition,
not hidden in an appendix. For

\[
x_{t+1}=(1-\mu\eta)x_t-\eta\bar\xi_{S,t},
\quad
a=1-\mu\eta,
\]

the exact finite-budget risk is

\[
R_B(q)=
a^{2K(q)}V_0+
\eta^2\Omega(q)
\frac{1-a^{2K(q)}}{1-a^2}.
\]

For \(q_b>q_a\), whenever the larger group has a lower noise term, it is
preferred exactly when

\[
V_0 <
\frac{
\eta^2\!\left[
\Omega(q_a)(1-a^{2K(q_a)})
-\Omega(q_b)(1-a^{2K(q_b)})
\right]
}{
(1-a^2)\left[
a^{2K(q_b)}-a^{2K(q_a)}
\right]
}.
\]

Substituting
\(\Omega(q)=\sigma^2[\rho+(1-\rho)/q]\) makes the threshold collapse toward
zero as common correlation \(\rho\) approaches one. This gives a clean,
reviewer-checkable phase-transition statement before the delayed generalization.

### Theorem 4: observable controller

Maintain scalar upper surrogate \(U_k\) and an upper confidence estimate
\(\widehat\Omega_k^+(S)\) from charged probes. Select the registered
\((q,\eta)\) minimizing the blockwise bound. The desired result is

\[
\mathcal R_k(\widehat q_k,\widehat\eta_k)
\le
(1+\varepsilon_k)
\min_{q,\eta}\mathcal R_k(q,\eta)
+\Delta_k,
\]

with high probability, where \(\varepsilon_k,\Delta_k\) are explicit in probe
length, mixing time, candidate-set size, and surrogate slack. EXP-006C tests a
point-estimate prototype; a theorem-quality algorithm should use conservative
confidence inflation so that \(U_k\) is actually an upper surrogate.

## Proof difficulty and minimum viable theorem

- The exact scalar phase transition is low difficulty and should be completed
  first; it is also an algebraic unit test for every later bound.
- The fixed-subset delayed recursion with joint Markov mixing is medium-to-high
  difficulty. The main technical risk is controlling delayed iterate/noise
  dependence without accidentally reintroducing cross-agent independence.
- The predictable adaptive-subset theorem is high difficulty because the
  selected sample is endogenous. A sample-splitting probe/exploitation design
  is the safest proof route.
- A nonconvex theorem or nonlinear function approximation extension is
  additional work and should not be promised until the strongly monotone/linear
  result is closed.

The minimum theoretically credible ICML package is Theorems 1--3 plus a
matched estimator and strong experiments. Theorem 4 would materially raise the
ceiling, but a weak or incorrect adaptive theorem is worse than omitting it.

## SDDE role

Use the stochastic delay differential equation

\[
d\Theta(t)=
-\bar G_{S(t)}(\Theta(t-\tau(t)))\,dt
+\sqrt{\eta(t)}
\Sigma_{S(t)}^{1/2}\,dW(t)
\]

as a principled diffusion representation, with
\(\Sigma_S\) equal to the Green--Kubo long-run covariance of the joint Markov
process. A Lyapunov--Krasovskii functional explains stability and the
state/correlation participation tradeoff.

For ICML, the SDDE cannot be the only guarantee. The paper must return from the
diffusion model to a discrete-time finite-sample bound or provide an explicit
approximation error. Otherwise reviewers can reasonably object that the
algorithm is discrete while the proof addresses only a surrogate.

## Complexity

The proposed controller is intentionally first-order:

- rolling centered moments: \(O(Wq_p)\);
- six participation counts by 17 scalar step sizes: 102 cached risk lookups
  per block;
- scalar Lyapunov state plus at most \(D+1\) delayed-state entries;
- no \(d\times d\) covariance, inverse, eigendecomposition, neural policy,
  actor--critic, or preconditioner.

The current synthetic implementation is CPU-only. GPU becomes useful only for
nonlinear deep TD/control benchmarks, not for the algorithm itself.

## ICML evidence bar

Synthetic results alone are insufficient. A credible submission needs:

1. exact phase-diagram validation against the theorem;
2. an online-controller experiment with charged probes and strong
   oracle-information fixed baselines;
3. linear TD policy evaluation on standard Markov chains with deliberately
   controlled cross-agent common factors;
4. at least one nonlinear benchmark with shared-environment or common-mode
   correlation, realistic delay traces, and communication-matched comparisons;
5. ablations for state adaptation, correlation adaptation, delay handling,
   probe cost, cluster-balanced selection, and misspecified correlation;
6. wall-clock/communication measurements and confidence intervals over seeds.

If the observable controller cannot beat the strongest fixed baseline, the
ICML main claim should be the correlation-limited speedup theorem and
participation phase transition, not adaptive control. That version can still
be strong, but it needs an exact or near-exact optimizer of the theoretical
bound and broader empirical confirmation.

## Preliminary prior-art boundary

The following authoritative records were checked on 2026-07-30. A fresh
multi-resolver citation-integrity audit is still required before manuscript
delivery.

- Dal Fabbro et al., *DASA: Delay-Adaptive Multi-Agent Stochastic
  Approximation*, arXiv:2403.17247 (2024): delay-adaptive multi-agent SA,
  independent agent Markov chains, average-delay rate, \(N\)-fold speedup.
- Dal Fabbro et al., *Finite-Time Analysis of Asynchronous Multi-Agent TD
  Learning*, arXiv:2407.20441 (2024): bounded asynchronous delay and linear
  speedup for multi-agent TD under independent agent observation processes.
- Khodadadian et al., *Federated Reinforcement Learning: Linear Speedup Under
  Markovian Sampling*, ICML 2022, PMLR 162:10997--11057: Markov sampling,
  multiple local updates, and linear speedup.
- Fraboni et al., *Clustered Sampling: Low-Variance and Improved
  Representativity for Clients Selection in Federated Learning*, ICML 2021,
  PMLR 139:3407--3416: clustered client selection for lower aggregation
  variance, but not the delayed cross-agent Markov participation law above.
- Cummins et al., *Controlling Participation in Federated Learning with
  Feedback*, L4DC 2025, PMLR 283:174--186: control-theoretic participation for
  ADMM federated learning, but not correlation-limited Markov speedup.
