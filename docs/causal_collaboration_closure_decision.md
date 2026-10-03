# Causal collaboration: closure decision and reference theorem

2026-08-31; parent 7f4e90c. Internal research note, not a submission-ready
paper or an efficacy preregistration. The reference code is
experiments/dependence_delay_linear/predictable_collaboration_reference.py.

## Decision

**Modify the architecture; preserve all historical evidence.** Stop using
the frozen post-hoc debt penalty as the foundation of a safety theorem.
Retain the question of useful time-varying collaboration and negative
transfer, but commit the graph before new observations and keep an
independent local donor bank.

This closes the timing gap for the restricted reference below. It changes
the algorithm: collaboration affects personalized outputs, not donor
training states. It is centralized estimation under exogenous fixed-policy
sampling, not decentralized policy execution or interacting Markov-game
control. The graph is online; the retention parameter is fixed publicly.
The proof is discrete, without an SDDE approximation.

Two decisions must not be conflated:

- Mathematical interface: **GO for the restricted reference.** An executed
  output-risk bound and a composite Lyapunov drift are derived below.
- ICML novelty/efficacy: **NOT ESTABLISHED.** Ordinary online aggregation is
  inherited, general TD/RL is not covered, and no GPU/formal run is authorized
  by these qualification tests.

### Why stop repairing the retrospective argument?

Let x,s be previous post-mix states, d=x-s, old target theta, and new target
theta+Delta. With shared next-block update noise epsilon_f,
L=ax+(1-a)(theta+Delta)+epsilon_f and
S=as+(1-a)(theta+Delta)+epsilon_f. Direct expansion gives

\[
 (L-\theta-\Delta)^2-(S-\theta-\Delta)^2
 =a^2[(x-\theta)^2-(s-\theta)^2]-2a^2d\Delta+2ad\epsilon_f. \tag{1}
\]

Continuous AR noise makes the conditional mean of epsilon_f depend on the
old noise boundary. Target changes add another term, and a final action
has no free next-block monitor. The existing target-interval shield is
valid under its assumptions but the frozen availability scan was
conservative. None of this proves every recursive-transfer algorithm
impossible; it does invalidate using the current debt argument as a
complete submission foundation.

T-083A FAIL, its stationary cost, the nonzero-AR baseline mismatch and
NoiseCertificate duplication remain recorded and unchanged.

## 1. Model and observable algorithm

There are n agents, T blocks of m>=2 observations, bounded scalar targets
|theta_i,t|<=R, and 0<a<1. Targets are constant within a block, may change
between blocks, and sampling/target dynamics do not depend on collaboration.
Targets are predictable in the analysis filtration, not visible to code.

\[
Y_{i,t,s}=\theta_{i,t}+\xi_{i,t,s},\qquad
\xi_{\rm next}=\lambda\xi_{\rm prev}+u_{\rm next}.
\]

Noise continues across blocks. Known 0<=lambda<1 and public vbar bound the
marginal innovation variance by vbar(1-lambda^2). Innovations are jointly
Gaussian across agents, independent across observation times and of the
past. Cross-agent correlation is allowed. No unrestricted unknown-mixing
guarantee is claimed.

Each independent local model L_i,t is in [-R,R]. Before a block, candidate
vector v_i,t has the recipient's current local model in coordinate zero,
and the other agents' local models delayed by D additional blocks in the
remaining coordinates. Startup holds the initial donor values. Candidate
values and simplex weights w_i,t are measurable before the block. Donor
models never use collaborative outputs as training inputs.

Choose z_i,t=w_i,t^T v_i,t. Then collect the charged block and compute

\[
h_{i,t}=
\frac{\sum_{s=2}^{m}(Y_{i,t,s}-\lambda Y_{i,t,s-1})}
{(m-1)(1-\lambda)}
=\theta_{i,t}+\varepsilon_{i,t},\quad
\bar\nu=\frac{\bar v(1+\lambda)}{(m-1)(1-\lambda)}. \tag{2}
\]

Only within-block pairs are used. Hence epsilon is conditionally centered
and Gaussian with variance <=nubar, even at target-changing boundaries.

Execute and maintain the local bank:

\[
x_{i,t}=az_{i,t}+(1-a)h_{i,t},\quad
s_{i,t}=av_{i,t,0}+(1-a)h_{i,t},\quad
L_{i,t}=\operatorname{clip}_{[-R,R]}s_{i,t}. \tag{3}
\]

The evaluated outputs x and s are NOT clipped; only the next local bank is
clipped. The contrast theorem is not silently transferred to clipped
outputs. Set

\[
\ell_{i,t}(w)=(w^\top v_{i,t}-h_{i,t})^2,\quad
g_{i,t}=2(z_{i,t}-h_{i,t})v_{i,t},\quad
w_{i,t+1}=\Pi_{\Delta_n}(w_{i,t}-\alpha g_{i,t}). \tag{4}
\]

Initialize w_i,1=e_0. Use public, outcome-independent constants

\[
G^2=4nR^2(4R^2+\bar\nu),\quad D_\Delta=\sqrt2,\quad
\alpha=\frac{D_\Delta}{G\sqrt T}. \tag{5}
\]

Indeed E[||g||^2 | past]<=G^2, because ||v||^2<=nR^2 and
E[(z-h)^2 | past]<=(z-theta)^2+nubar<=4R^2+nubar.
True target, risk and old formal data are never controller inputs.

## 2. Exact executed-risk alignment and comparison theorem

For a predictable comparator u, set c=u^T v, x^u=ac+(1-a)h and d=z-c.
Expansion yields the PATHWISE identity

\[
(x-\theta)^2-(x^u-\theta)^2
=a^2[\ell_t(w)-\ell_t(u)]+2ad\varepsilon_t. \tag{6}
\]

Predictability centers the last term conditionally. This covers the last
block without buying a future monitor. If z is chosen using h, centering
fails in general; the unit counterexample takes z=epsilon.

Projection nonexpansiveness, expansion of ||w-alpha*g-u||^2 and convexity give

\[
\ell_t(w_t)-\ell_t(u)
\le \frac{\|w_t-u\|^2-\|w_{t+1}-u\|^2}{2\alpha}
+\frac{\alpha}{2}\|g_t\|^2. \tag{7}
\]

For predictable u_t with path length
P_T=sum_{t<T}||u_{t+1}-u_t||, telescope (7). Changing the comparator changes
each squared distance by at most 2D_Delta||u_{t+1}-u_t||. Therefore

\[
\sum_t[\ell_t(w_t)-\ell_t(u_t)]
\le\frac{D_\Delta^2+2D_\Delta P_T}{2\alpha}
+\frac{\alpha}{2}\sum_t\|g_t\|^2. \tag{8}
\]

Taking expectations using (6) proves

\[
\mathbb E\sum_t[(x_t-\theta_t)^2-(x_t^u-\theta_t)^2]
\le a^2\left[
\frac{D_\Delta^2+2D_\Delta\mathbb E P_T}{2\alpha}
+\frac{\alpha TG^2}{2}\right]. \tag{9}
\]

A fixed comparator has bound a^2 D_Delta G sqrt(T). For the local comparator
e_0, the initial distance is zero, so (7) yields the sharper
a^2 alpha T G^2/2 bound.

Scope: any fixed graph chosen before evaluation, including a deterministic
population-optimal graph for this independent bank, is covered. The bank
does not change under that counterfactual readout. An outcome-selected
hindsight graph need not be predictable: (8) still holds but its risk
translation (9) does not follow. The recursively trained fixed graphs of
T-083A are a different comparator class. Equation (9) is not universal
improvement, terminal-risk dominance, or a theorem for general TD.

## 3. Complete composite Lyapunov learning bound

For a stationary target, let e_t=L_t-theta and
Phi_t=||w_t-e_0||^2/(2 alpha). Choose

\[
\Psi_t=a^2\Phi_t+\frac{a^2}{1-a^2}e_{t-1}^2. \tag{10}
\]

This combines adaptation potential with local parameter error, not an
unproved debt queue. With r_t=(x_t-theta)^2, (6)-(7) imply

\[
\mathbb E[a^2(\Phi_{t+1}-\Phi_t)+r_t\mid F_t^-]
\le a^2e_{t-1}^2+(1-a)^2\bar\nu+a^2\alpha G^2/2. \tag{11}
\]

Since theta is in the clipping interval, clipping decreases its distance.
Thus E[e_t^2-e_{t-1}^2 | F_t^-] <=
-(1-a^2)e_{t-1}^2+(1-a)^2 nubar. Multiplication by a^2/(1-a^2) and addition to
(11) cancel the old-error term:

\[
\mathbb E[\Psi_{t+1}-\Psi_t+r_t\mid F_t^-]
\le\frac{1-a}{1+a}\bar\nu+\frac{a^2\alpha G^2}{2}. \tag{12}
\]

Nonnegativity of Psi and telescoping prove

\[
\frac1T\sum_{t=1}^T\mathbb E r_t
\le\frac{\mathbb E\Psi_1}{T}
+\frac{1-a}{1+a}\bar\nu+\frac{a^2\alpha G^2}{2}. \tag{13}
\]

Substituting (5), the adaptation remainder is O(T^-1/2). Fixed retention
and noise leave a noise floor: this is NOT convergence to zero. All
statements hold per recipient; averaging does not require agent independence.

For changing targets define Delta_t=theta_t-theta_{t-1}. Choose kappa>0 with
r=a^2(1+kappa)<1 and c_Delta=a^2(1+1/kappa). Young's inequality gives
conditional local output risk <=r e_{t-1}^2+c_Delta Delta_t^2+(1-a)^2 nubar,
where e_{t-1}=L_{t-1}-theta_{t-1}. Now choose

\[
\Psi_t^{\rm change}=a^2\Phi_t+
\frac{r}{1-r}(L_{t-1}-\theta_{t-1})^2.
\]

The identical cancellation proves

\[
\mathbb E[\Psi_{t+1}^{\rm change}-\Psi_t^{\rm change}+r_t\mid F_t^-]
\le\frac{c_\Delta\Delta_t^2+(1-a)^2\bar\nu}{1-r}
+a^2\alpha G^2/2. \tag{14}
\]

For example kappa=(1-a^2)/(2a^2) is admissible. Summation explicitly charges
target variation; arbitrary rapid changes do not imply vanishing risk.

## 4. Observable all-prefix cumulative safety allowance

For the local comparator, d_t=z_t-v_t,0 is observable and predictable.
Define S_k=sum_{t<=k}2ad_t epsilon_t and
V_k=sum_{t<=k}4a^2 d_t^2 nubar. For each real beta,
exp(beta*S_k-beta^2*V_k/2) is a nonnegative supermartingale. Mixing beta
against N(0,1/v_0) with public fixed v_0>0 gives

\[
M_k=\sqrt{\frac{v_0}{v_0+V_k}}
\exp\left(\frac{S_k^2}{2(v_0+V_k)}\right).
\]

Ville's inequality and a union bound over n agents imply, with probability
at least 1-delta, simultaneously for all agents and k<=T,

\[
|S_k|\le b(V_k)=
\sqrt{(V_k+v_0)[2\log(n/\delta)+\log(1+V_k/v_0)]}. \tag{15}
\]

Combining (6)-(7) and the zero initial local distance gives a bound on
REALIZED accumulated output-risk contrast:

\[
\sum_{t\le k}[(x_t-\theta_t)^2-(s_t-\theta_t)^2]
\le\frac{a^2\alpha}{2}\sum_{t\le k}\|g_t\|^2+b(V_k). \tag{16}
\]

The code uses v_0=1. The right side contains no target. Cross-agent
correlation does not invalidate the marginal martingales/union bound.
This is not zero harm per step or a prescribed hard budget. Its observed
gradient sum can be large; no practically tight O(sqrt(T)) realized
allowance is asserted without further control. Unlike a standalone
post-hoc monitor, it holds for this specified policy by construction, with
the additional learning theorem (12).

## 5. Delay, gain and complexity

For the same predictable u_t on delayed/fresh bank vectors, boundedness gives

\[
|(u_t^\top v_t^D-\theta_t)^2-(u_t^\top v_t^0-\theta_t)^2|
\le4R\|v_t^D-v_t^0\|_\infty. \tag{17}
\]

The expected post-update penalty is at most a^2 times the sum of (17).
No small delay penalty is promised under arbitrary source variation.

If a predictable comparator has expected accumulated advantage Gamma_T
over local, (9) proves positive expected gain only when Gamma_T exceeds the
stated regret bound. This is an inherited regret-to-gain deduction, NOT a
new matching lower bound or sharp phase transition. The noiseless unit
witness confirms the local vertex is not absorbing, not broad efficacy.

The dense scalar implementation costs O(n*m+n^2 log n) per block,
O(n^2+n(D+1)) persistent memory, and an O(n*m) input block. It has no matrix
inverse/eigensolve. It charges n*m actor transitions and n(n-1) directed
scalar donor payloads, even at zero weights. These are payload counts;
wire headers/server overhead must be frozen in a byte-budget experiment.
Do not call dense graph learning linear in n. Sparse/vector extensions
are not implemented or validated here.

The theorems compare the same number of charged blocks. They do not prove
optimal risk under two resource budgets or wall-clock speedup. Cross-agent
correlation is allowed in the guarantee, but no correlation-specific
adaptation optimality or sharp correlation phase boundary is proved.

## 6. Novelty boundary and next integration gate

The projected-gradient machinery is inherited from Zinkevich,
*Online Convex Programming and Generalized Infinitesimal Gradient Ascent*,
ICML 2003 ([author paper](https://www.martin.zinkevich.org/publications/ICML03.pdf)).
Multitask sharing, similarity-dependent regret and cheap mirror-descent
updates also appear in Cesa-Bianchi, Laforgue, Paudice and Pontil,
*Multitask Online Mirror Descent*, cited as the 2021 arXiv record revised in
2022 ([arXiv](https://arxiv.org/abs/2106.02393)).
Neither is claimed identical to this delayed readout. These two checks
establish inherited components, not an exhaustive novelty audit.

The source-verification JSON records positive primary/index matches and
the unresolved Zinkevich page-range conflict. No final bibliography entry
has been created. Existence/core attribution is verified; publication
bibliography delivery remains blocked on that metadata detail.

Proceed with ONE bounded integration package:

1. Establish a useful fixed-policy RL interface for this causal risk
   contract, or reject that reduction. An unbiased AR location statistic
   is not an ordinary bootstrapped TD target. Specify a meaningful
   contribution beyond ordinary multitask aggregation before naming a
   final ICML algorithm.
2. Only after interface/novelty/feasibility checks, freeze one CPU development
   protocol: correct static readout baselines, plain online aggregation,
   nonzero temporal correlation, heterogeneity/change, delay, measured
   allowance tightness and full costs. Preserve every outcome.
3. Successful unchanged development permits fresh independent confirmation,
   then a matched standard RL benchmark and a unified paper/proof appendix.
   Request GPU handoff only if needed.

If integration only repackages known aggregation, reject that candidate
instead of launching a large experiment to manufacture novelty. No new
experiment number bypasses the previous failures. The final ICML paper,
general TD proof and standard RL efficacy evidence remain unfinished.
