# Lower side of the clocked asynchronous phase

Status: exact robust-direction and service-clock lemmas.  These results close
two necessary lower-side interfaces; they are not yet a minimax lower bound
matching every constant in the Lyapunov upper theorem.

## Cross-agent indistinguishability

Let a returned block packet be `g_hat=s*e`, where `s>0` and `e` is a unit
vector.  Suppose the learner only has a valid current-gradient uncertainty
ball

\[
 \|g-\widehat g\|\le B.
\]

For an `L`-smooth potential and the update `alpha*g_hat`, the largest uniform
lower certificate available from this information is

\[
\begin{aligned}
 &\sup_{\alpha\ge0}\inf_{\|g-\widehat g\|\le B}
 \left[\alpha\langle g,\widehat g\rangle
 -\frac L2\alpha^2\|\widehat g\|^2\right]\\
 &\qquad=\frac{[s-B]_+^2}{2L}.
\end{aligned}
\tag{1}
\]

The inner infimum aligns the uncertainty against the packet, producing
`alpha*s*(s-B)-L*alpha^2*s^2/2`.  If `B<s`, its maximizer is
`alpha=(s-B)/(Ls)`.  If `B>=s`, zero is optimal and no positive step has a
uniform positive certificate.

This ambiguity has a distinct-agent potential-game witness.  At packet birth,
take teammate coordinate `y=0` and owner coordinate `x=0` in

\[
 \Phi_h(x,y)=s x+hxy-\frac L2x^2-\frac\mu2y^2,
 \qquad h\in\{-B,+B\},
\tag{2}
\]

with `mu>B^2/L` when global concavity is desired.  Both games return the same
birth gradient `partial_x Phi_h(0,0)=s`.  After the teammate moves to `y=1`,
the owner gradients are `s-B` and `s+B`.  The packet and its birth metadata
cannot distinguish the two games.  Equation (1) is therefore not a weakness
of the proposed Lyapunov proof: it is an information boundary created by an
interacting teammate update.

For the clocked learner, a valid choice is

\[
 B_k=\beta_{i,H}+\sum_{j\ne i}L_{ij}
 \|\theta_j^k-\theta_j^{b_k}\|,
\tag{3}
\]

where `beta_(i,H)` is the charged finite-horizon Markov-packet bias.  The ratio
`B_k/||g_hat_k||` is the unfavorable side of the rate--coupling phase.  The
main algorithm need not estimate this ratio online; the result explains why a
universal safe stale-direction speedup is impossible.

## Essential-agent service lower bound

Suppose policy block `i` needs at least `m_i` sequential fresh packets before
a declared full-policy accuracy criterion can hold.  If its packet completion
clock is Poisson with rate `lambda_i`, the expected arrival time of packet
`m_i` is `m_i/lambda_i`.  Therefore every stopping time that requires all
essential blocks satisfies

\[
 \mathbb E T\ge\max_i\frac{m_i}{\lambda_i}.
\tag{4}
\]

This follows because `T` is pathwise no earlier than each required arrival
time and the `m_i`-th Poisson arrival has a Gamma distribution with that mean.
It formalizes the slow-essential-agent limit: summing all actor rates cannot
replace the rate of a strategically necessary policy block.

The existing two-curvature fresh-query witness supplies a concrete `m_i=2`
subproblem and separates two adaptive asynchronous queries from arbitrarily
many duplicate packets at one frozen policy.  A final minimax theorem should
extend (4) to the stochastic accuracy-dependent packet requirement and compare
its constants with the upper wall-clock theorem.

## Validation

`wall_clock_phase.py` implements (1) with an optional public step cap and (4).
Tests compare the closed form with a 200,001-point direct search, exercise the
exact `B=s` boundary, verify the slow-block maximum and reject malformed
inputs.  These are deterministic algebra checks, not efficacy data.
