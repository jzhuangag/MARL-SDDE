# Unknown-mixing impossibility (AC-8)

## Scope and quantifiers

Fix two **strictly positive and distinct** common-factor variances
\(\theta _0\ne\theta _1\). A policy may choose finite \(q_t,b_t\)
predictably, randomize through a parameter-independent common kernel, stop at
a dual-budget-bounded time \(\tau_T\), and decide
\(\phi_T\in\{0,1\}\). Delay may reduce the usable post-decision horizon but
does not alter the observation law.

Let \({\cal L}_{j,\lambda}^{\pi_T}\) be the resulting stopped-history law.
For the unrestricted mixing class,

\[
 \inf_{\pi_T,\phi_T}\ \sup_{0\leq\lambda<1}\
 \max_{j\in\{0,1\}}
 P_{j,\lambda}^{\pi_T}\{\phi_T\ne j\}
 \ \geq\ e_*(\theta _0,\theta _1)-o_T(1)>0.                 \tag{AC8}
\]

Consequently no sequence is uniformly \(\delta_T\)-correct with
\(\delta_T\to0\). The infimum is over every policy and test obeying its
finite dual budgets, not merely a fixed probe block.

## Endpoint experiment

At \(\lambda=1\), stationarity gives one latent draw

\[
 C\sim N(0,\theta_j),\qquad Z_t=\sqrt{q_t}C+\varepsilon_t.
\]

All adaptive actions, observations, stopping, and the decision are a Markov
kernel \(K^{\pi_T}\) applied to \(C\). This remains true with predictable
irregular strides: strides cannot create a fresh latent innovation when the
coefficient is one. Total variation contracts under a Markov kernel, hence

\[
 \operatorname{TV}({\cal L}_{0,1}^{\pi_T},{\cal L}_{1,1}^{\pi_T})
 \leq \operatorname{TV}\{N(0,\theta_0),N(0,\theta_1)\}=:v_*<1.
\]

Le Cam's two-point inequality yields

\[
 \max_jP_{j,1}^{\pi_T}(\phi_T\ne j)\geq e_*:=\frac{1-v_*}{2}>0. \tag{1}
\]

Writing \(\theta_< =\min_j\theta_j\),
\(\theta_> =\max_j\theta_j\), and

\[
 x_*^2=\frac{\theta_<\theta_>\log(\theta_>/\theta_<)}
                 {\theta_>-\theta_<},
\]

the exact value used by the audit is

\[
 v_*=2\left[\Phi(x_*/\sqrt{\theta_<})-
             \Phi(x_*/\sqrt{\theta_>})\right].
\]

Strict positivity of both variances is necessary: a degenerate zero-variance
hypothesis can be singular with respect to a positive-variance latent law and
is not covered by this lower bound.

## Removing the endpoint does not restore uniformity

For any fixed \(T\) and policy with finite affordable history, the stopped
law is a finite mixture of nonsingular Gaussian observation laws and common
policy kernels. On every finite action/history branch its covariance entries
are continuous functions of \(\lambda\), including irregular powers
\(\lambda^{|S_t-S_s|}\). Truncation over the finite budget tree, followed by
dominated convergence, therefore gives

\[
 \operatorname{TV}({\cal L}_{j,\lambda}^{\pi_T},
                    {\cal L}_{j,1}^{\pi_T})\longrightarrow0
 \quad(\lambda\uparrow1),\qquad j=0,1.                       \tag{2}
\]

Choose \(\lambda_T<1\) after the finite policy is fixed so that both
distances in (2) are at most \(1/T\). The testing error at \(\lambda_T\) is
then at least \(e_*-1/T\), proving (AC8). This argument also covers policies
whose allowable history size and physical span grow with \(T\): the
adversary chooses a correspondingly closer \(\lambda_T\). The implementation
checks covariance-KL convergence for an irregular, changing-\(q\) design;
Pinsker converts it to the required TV convergence.

## What the theorem does and does not close

- **Closed negatively:** fully unknown mixing with a parameter class reaching
  arbitrarily close to one, under any finite affordable predictable design.
- **Still possible:** pointwise consistency for each fixed
  \(\lambda<1\). Repeated innovations exist and information can grow.
- **Open positive route:** a compact class
  \(\lambda\leq1-\gamma\) with public \(\gamma>0\), or a separate valid
  certificate imposing such a bound.

Thus AC-8 is no longer “open with a boundary warning.” It is a formal
negative theorem for unrestricted unknown mixing, with a precisely separated
restricted route.
