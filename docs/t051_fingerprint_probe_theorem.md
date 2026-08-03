# T-051 observable fingerprint-probe theorem

## Construction

T-051 keeps the T-049 marginal-preserving trajectory-switch coupling but
replaces the conservative moment-ratio ICC probe. In one independent probe
block, actor (i) receives the common standard-task trajectory (U) when
(Z_i=1), and a private trajectory (V_i) otherwise, with
(Z_i\sim\operatorname{Bernoulli}(\sqrt\rho)). Each actor sends one
fixed-length hash of its state path. The server observes only equality of
hashes; it does not observe (Z_i), (U), or the true correlation.

Hash collisions are assumed absent at the chosen digest length. Accidental
*trajectory* collisions are retained exactly.

## Lemma 1: exact independent-path collision probability

Let (P) and (\pi) be the public regenerative fixed-policy kernel and its
stationary law. For a fingerprint containing (L) transitions, define

\[
 c_L=\Pr\{(S_0,\ldots,S_L)=(S'_0,\ldots,S'_L)\},
\]

where the two paths are independent. If (w_0=\pi\odot\pi), then

\[
 w_{t+1}=w_t(P\odot P),\qquad c_L=w_L\mathbf1.          \tag{1}
\]

This follows by summing the squared probability of every state path. It is
an exact finite-kernel computation, not a mixing or birthday approximation.

## Theorem 1: observable match identity

For an unordered actor pair, let (M_{ij}=1) when their fingerprints agree.
Both actors use the common path with probability
((\sqrt\rho)^2=\rho), in which case they match surely. Conditional on not
both using the common path, their paths are independent with the same public
marginal law. Therefore

\[
 \mathbb E M_{ij}=\rho+(1-\rho)c_L
 =c_L+(1-c_L)\rho.                                    \tag{2}
\]

For (q_p\ge2), the within-block pair-match rate

\[
 W=\binom{q_p}{2}^{-1}\sum_{i<j}M_{ij}
\]

lies in ([0,1]) and has the same expectation (2). Pair indicators within a
block need not be independent.

## Theorem 2: fixed-block correlation certificate

Use (n) fully independent probe blocks and let \(\bar W_n\) be their mean.
Define

\[
 \widehat\rho=\Pi_{[0,1]}
 \frac{\bar W_n-c_L}{1-c_L},\qquad
 r_{n,\alpha}=\frac1{1-c_L}
 \sqrt{\frac{\log(2/\alpha)}{2n}}.                     \tag{3}
\]

Hoeffding's inequality applies across blocks, without assuming independent
pairs inside a block, and gives

\[
 \Pr\{|\widehat\rho-\rho|>r_{n,\alpha}\}\le\alpha.    \tag{4}
\]

Clipping cannot enlarge the estimation error. The interval obtained by
clipping the unprojected estimate plus or minus (r_{n,\alpha}) has the same
coverage.

## Theorem 3: plug-in participation regret

For a fixed catalogue, every leading stationary coefficient

\[
 K_q(\rho)=(h+q)\left(\rho+\frac{1-\rho}{q}\right)
\]

is affine. Its optimality set (I_q\subseteq[0,1]) is therefore a closed
interval computable from pairwise line intersections. The plug-in controller
chooses the minimizer at \(\widehat\rho\).

Let (d_q(\rho)=\operatorname{dist}(\rho,I_q)) and
\(K_\star(\rho)=\min_qK_q(\rho)\). Equations (3)--(4) imply

\[
 \mathbb E K_{\widehat q}(\rho)
 \le K_\star(\rho)+
 \sum_{q:K_q>K_\star}
 [K_q(\rho)-K_\star(\rho)]
 \min\{1,2e^{-2n(1-c_L)^2d_q(\rho)^2}\}.              \tag{5}
\]

Indeed, selecting a suboptimal (q) requires the estimate to reach its
optimality interval, a deviation of at least (d_q(\rho)). Summing the
coefficient-weighted event bounds proves (5). Close to a decision boundary,
the classification probability may be hard but the coefficient gap is also
small; (5) retains that fact instead of requiring exact regime labels.

If each probe block sends one (q_p)-agent fingerprint vector, its message
cost is (h+q_p), independent of fingerprint length. Its environment cost
is (L) parallel task transitions. For post-probe learning message budget
(B_L) and no-probe total budget (B=B_L+n(h+q_p)), the leading expected
risk ratio to a fixed baseline (q_0) is bounded by

\[
 \frac{B}{B_L}
 \frac{\text{right side of (5)}}{K_{q_0}(\rho)}.        \tag{6}
\]

Thus all probe messages are charged. A later static design must also make the
environment budget at least (nL) plus the selected learning horizon and
charge delay separately.

## Complexity and scope

Equation (1) costs (O(L|\mathcal S|^2)) offline. Online, each actor hashes
its length-(L) state path locally; grouping (q_p) fixed-size fingerprints
costs (O(q_p\log q_p)) per block and the action scan costs (O(|\mathcal
Q|)). No covariance matrix, Hessian, inverse preconditioner, or online model
solve is used.

The theorem is exact for the registered trajectory-switch coupling. It does
not claim that fingerprint matches estimate arbitrary notions of agent
dependence. A standard-task experiment must report this coupling explicitly
and retain unchanged per-agent Gymnasium marginals.
