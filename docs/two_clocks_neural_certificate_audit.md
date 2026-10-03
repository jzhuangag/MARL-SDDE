# Neural-certificate scope audit for Two Clocks

Date: 2026-09-02.

Status: outcome-free analytic audit.  No trajectory, return, seed, benchmark
selection or GPU result is used.

## Question

The finite-game theorem bounds the change in agent `i`'s expected packet when
teammate `j` changes a categorical softmax policy.  If

\[
 \sup_s\|z_j(s;\vartheta)-z_j(s;\vartheta')\|_2
 \le K_j\|\vartheta-\vartheta'\|_2,
\]

then the theorem-facing off-diagonal coefficient is

\[
 L_{ij}^{\rm net}
 =\frac{H\sqrt{A_j}}2 C_{i,H}K_j.
\]

This audit asks whether that formula currently supplies a finite, useful and
complete certified step for the standard HARL actor planned in the nonlinear
benchmark bridge.

## Result

**It does not.**  The current standard neural actor is an empirical extension,
not a theorem-covered instantiation.  Four independent gaps prevent a valid
numerical `alpha_cert` from being reported.

### 1. An unconstrained multilayer actor has no global parameter-to-output constant

For a two-layer ReLU subnetwork

\[
 f_{W_1,W_2}(x)=W_2[W_1x]_+,
\]

take scalar `x=1`, `W_1=1`, `W'_1=1+epsilon` and
`W_2=W'_2=M`.  The parameter displacement is `epsilon`, whereas the output
displacement is `M epsilon`.  Letting `M` grow rules out any finite global
parameter-to-output Lipschitz constant over the unconstrained parameter
space.  Layer normalization does not repair the global statement: the final
linear weight and the learned normalization affine parameters are themselves
unbounded.

Consequently the standard `[128,128]` HARL actor needs an explicit compact
parameter set, spectral/row-norm constraints, bounded normalization affine
parameters and bounded inputs before a uniform `K_j` can be declared.  A
sampled Jacobian norm is a local diagnostic, not a replacement for this
coverage condition.

### 2. The existing bound is categorical, while the continuous layer is Gaussian

The planned MAMuJoCo layer uses a diagonal Gaussian actor with a trainable
`log_std`.  The categorical-softmax total-variation inequality is therefore
inapplicable.  For diagonal Gaussians a valid route starts from

\[
 \operatorname{KL}(P\|Q)=\frac12\sum_r\left[
 \frac{\sigma_r^2+(\mu_r-\mu'_r)^2}{(\sigma'_r)^2}
 -1+2\log\frac{\sigma'_r}{\sigma_r}\right]
\]

and Pinsker's inequality.  It requires public lower and upper variance bounds
and a uniform parameter-to-mean/log-standard-deviation envelope.  Those
objects are not currently part of the algorithm.  Freezing or projecting the
variance would be a new theorem-compatible architecture, not a silent reuse
of the categorical result.

### 3. Off-diagonal sensitivity alone does not determine the common step

The Lyapunov root also contains the owner-block smoothness `L_jj`:

\[
 L_{jj}\alpha+(1+\delta)D^2u_j\alpha^2\le1.
\]

The trajectory-law argument supplies an off-diagonal change-of-law bound.  It
does not by itself bound the change in the owner's score function and hence
does not instantiate `L_jj` for an unconstrained neural policy.  Substituting
a PPO clipping ratio, an observed Hessian norm or a sampled Fisher eigenvalue
would change a theorem-facing uniform constant into an empirical proxy.

### 4. Gradient clipping is not a free certificate

Clipping every returned packet would give a finite pathwise statistic bound
`C_(i,H)`.  It also changes the conditional mean of the packet.  A convergence
claim for the original policy gradient then needs an explicit clipping-bias
term or a tail condition proving that clipping is inactive with controlled
probability.  The previous `maximum_step_norm` in development code is not such
a theorem.

## Scientific decision

The architecture-specific nonvacuity gate closes **negatively** for the
unconstrained standard HARL actor.  Therefore:

1. the discrete theorem remains scoped to finite/tabular factorized softmax
   policies with declared finite constants;
2. standard neural MARL may be evaluated only as a clearly labelled empirical
   extension of the same single-flight timing mechanism;
3. the neural implementation must not be called pathwise certified unless a
   later, separately proved projected/spectral architecture supplies `K_j`,
   `L_jj`, Gaussian variance bounds where applicable, and clipping bias;
4. a practical measured teammate-KL or local-Jacobian proxy may be reported,
   but it cannot be used as evidence that the uniform theorem applies;
5. no new CPU sweep can resolve these analytic gaps by choosing a favorable
   network seed.

This negative scope result does not invalidate the finite-game theorem or its
two positive CPU confirmations.  It does reduce the strength of a future ICML
claim: standard-MARL experiments must stand as empirical evidence, while the
theorem-facing contribution remains the exact rate--coupling phase on the
declared finite-policy class.

## What would be required for a theorem-covered neural extension

A future extension must fix all of the following before outcome access:

- bounded observations or a certified feature map;
- spectral/row-norm projection for every linear layer;
- bounded layer-normalization affine parameters, or a replacement whose
  Lipschitz constant is explicit;
- for Gaussian policies, bounded variance and a parameter-to-variance map;
- a diagonal policy-gradient smoothness bound on the same compact set;
- a bounded packet construction plus an explicit truncation/clipping-bias
  bound;
- a numerical audit showing that the resulting positive-root step is not
  negligible relative to the declared optimizer scale.

Adding these restrictions only to rescue a benchmark outcome is forbidden.
