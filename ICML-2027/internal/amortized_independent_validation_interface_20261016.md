# Independent selected-packet gain and joint full-cost interface

Status: prospective learning-response instrument, not a confidence certificate.
This does not change the selected amortized algorithm or erase its error floor.

At frame t freeze beta, candidate policies, coefficients, potential, resource
costs and meta(m,lambda,geometry). Collect20 independent episodes under beta;
m in{4,8,16} uses only a prefix. The last four episodes are inaccessible to
selection. Choose the current priced full-score argmax k_hat using the prefix;
only then open the tail for the chosen action and idle. For the unclipped
actual-return contrast X with frozen past-only potential,

    E[mean_tail X(k_hat) | history, prefix] =
        [J_H(pi_(k_hat))-J_H(beta)]/R.

Proof: conditional on the prefix, k_hat and all policies are fixed, while the
tail episodes are independent beta trajectories. Apply the likelihood-ratio
TD-control-variate identity to each. Training prefix argmax scores remain
selection-biased; this new held-out estimate is not a simultaneous UCB or a
finite-m realized no-harm guarantee. Across learning frames conditional means
vary with the learner, so frame/episode observations are not independent replicas
of a single task effect. Scientific seed is the comparison unit (three seeds).

The full candidate score does not see tail returns. After selection, the same
bank's local programs and critic may use all20 already paid episodes for FUTURE
learning. This does not leak tail data into the current receipt; the tail is not
benchmark evaluation data. Same-world benchmark evaluation remains on a wholly
different role/seed stream and never feeds either controller or program.

Tail cost is known before the bank as a function of candidate action a:

    C_tail(a) = 4 H sum_(changed i) (4*obsdim_i+16) logical bytes,
    F_tail(a) = 4 H * number_changed_actor_blocks / fixed_work_scale.

These are part of candidate costs alongside policy broadcasts. If the frozen
potential exists, its4H tail predictions are common sunk work. Thus the same
composite drift action prices the actual score-validation data path, and closing
queues includes it. Idle's tail contrast is identically zero, so no private
actor record/forward is charged for idle; already sampled public trajectories
and later full programs remain charged for EVERY action. No duplicate program
bytes are refunded. Tail diagnostics cannot adjust meta feedback/receipt gains.

The existing actual-return telescoping theorem applies with these augmented
costs. Constant return scale and objective-motion caveats are unchanged. The
conditional pointwise-feasible idle/min-probe reference remains available under
the prospective1.25 communication/1.25 proxy-compute targets for these shapes.
Theory compares eligible actions at our states, not a separate HAPPO trajectory.
The held-out mean measures selection response; only the separate full learning
curves test whether cumulative learning outperforms strong controls. Neither
one alone is admitted as ICML-level evidence.
