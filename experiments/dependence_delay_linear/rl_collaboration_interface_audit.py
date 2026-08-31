"""Exact finite-model interface audit; no sampled training or efficacy pilot.

Matrix solves below are audit oracles only, never inputs to a controller.
The cases refute invalid reductions; they do not select an experiment task.
"""

import json

import numpy as np


def validate_mrp(transition, reward, discount):
    p = np.asarray(transition, dtype=float)
    r = np.asarray(reward, dtype=float)
    if (p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] < 1
            or r.shape != (p.shape[0],) or not np.isfinite(p).all()
            or not np.isfinite(r).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0)
            or not 0 <= discount < 1):
        raise ValueError("invalid finite discounted Markov reward process")
    return p, r


def value_oracle(transition, reward, discount):
    p, r = validate_mrp(transition, reward, discount)
    return np.linalg.solve(np.eye(len(r)) - discount*p, r)


def return_moments(transition, reward, discount, horizon):
    """Moments of sum_{k=0}^{H-1} gamma^k r(S_k), conditional on S_0."""
    p, r = validate_mrp(transition, reward, discount)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("return horizon must be a positive integer")
    mean, second = np.zeros_like(r), np.zeros_like(r)
    for _ in range(horizon):
        next_mean = p @ mean
        second = r*r + 2*discount*r*next_mean + discount**2*(p @ second)
        mean = r + discount*next_mean
    return mean, second


def stationary_distribution(transition):
    p = np.asarray(transition, dtype=float)
    validate_mrp(p, np.zeros(p.shape[0]), 0.)
    equation = p.T - np.eye(len(p))
    equation[-1] = 1.
    rhs = np.zeros(len(p))
    rhs[-1] = 1.
    mu = np.linalg.solve(equation, rhs)
    if np.any(mu <= 0) or not np.allclose(mu @ p, mu):
        raise ValueError("audit requires a positive stationary distribution")
    return mu


def dirichlet_operator(transition, discount):
    p = np.asarray(transition, dtype=float)
    mu = stationary_distribution(p)
    validate_mrp(p, np.zeros(len(p)), discount)
    eye = np.eye(len(p))
    operator = (1-discount)*np.diag(mu)
    for s in range(len(p)):
        for next_s in range(len(p)):
            difference = eye[s]-eye[next_s]
            operator += .5*discount*mu[s]*p[s, next_s]*np.outer(difference, difference)
    return operator, mu


def td_label_counterexample():
    # Continuing one-state MRP. The baseline is not yet at its fixed point.
    p, reward, gamma = np.ones((1, 1)), np.ones(1), .9
    truth = float(value_oracle(p, reward, gamma)[0])
    local, donor, a = 0., 8., .8
    td_label = float((reward + gamma*p @ np.array([local]))[0])
    x, s = a*donor+(1-a)*td_label, a*local+(1-a)*td_label
    contrast = (x-truth)**2 - (s-truth)**2
    claimed = a*a*((donor-td_label)**2 - (local-td_label)**2)
    missing = 2*a*(donor-local)*(td_label-truth)
    return {"true_value": truth, "td_target_mean": td_label,
            "executed_value_risk_contrast": contrast,
            "incorrect_unbiased_label_prediction": claimed,
            "missing_bootstrap_bias_term": missing,
            "identity_residual_after_correction": contrast-claimed-missing}


def history_mismatch(horizon=32):
    """One-state deterministic learning per agent; J versus local I.

    The local update x <- .5*x + .5*theta is a valid contracted mean update.
    Evaluating I on J's pre-mix states is not running I on its own history.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("invalid horizon")
    theta = np.array([0., 1.])
    averaging = np.full((2, 2), .5)
    collaborative = np.zeros(2)
    independent = np.zeros(2)
    same_history_loss = own_history_loss = 0.
    for _ in range(horizon):
        proposal = .5*collaborative + .5*theta
        independent = .5*independent + .5*theta
        same_history_loss += np.sum((proposal-theta)**2)
        own_history_loss += np.sum((independent-theta)**2)
        collaborative = averaging @ proposal
    return {"blocks": horizon, "same_history_I_loss": float(same_history_loss),
            "counterfactual_I_loss": float(own_history_loss),
            "gap": float(same_history_loss-own_history_loss),
            "closed_form_gap": horizon/8-(1-4.**(-horizon))/6,
            "asymptotic_gap_per_block": 1/8}


def nonreversible_counterexample():
    p = .1*np.eye(3) + .9*np.roll(np.eye(3), 1, axis=1)
    reward, gamma = np.array([1., 0., -1.]), .9
    symmetric, mu = dirichlet_operator(p, gamma)
    a = np.diag(mu) @ (np.eye(3)-gamma*p)
    b = mu*reward
    truth = value_oracle(p, reward, gamma)
    wrong_optimum = np.linalg.solve(symmetric, b)
    return {"A": a.tolist(), "symmetric_dirichlet": symmetric.tolist(),
            "skew_operator_norm": float(np.linalg.norm(a-a.T)),
            "gradient_at_true_value": (symmetric@truth-b).tolist(),
            "value_error_of_symmetric_surrogate_minimizer": float(mu @ (wrong_optimum-truth)**2)}


def residual_ranking_counterexample():
    p = np.array([[.1, .9], [.9, .1]])
    gamma, mu = .9, np.array([.5, .5])
    b = np.eye(2)-gamma*p
    errors = np.array([[1., 1.], [.1, -.1]])
    return {"value_mse": [float(mu @ (e*e)) for e in errors],
            "squared_mean_Bellman_residual": [float(mu @ ((b@e)**2)) for e in errors]}


def markov_td_risk_gramian(transition, discount, step, horizon):
    """Exact conditional future-risk metric for homogeneous tabular TD(0).

    Zero rewards give V*=0 and error update B_ss' = I-eta e_s(e_s-gamma e_s')^T.
    Q_s^H is expected sum of squared parameter errors for H pre-update losses,
    conditional on current Markov state s. This is a model-known audit oracle.
    """
    p = np.asarray(transition, dtype=float)
    validate_mrp(p, np.zeros(len(p)), discount)
    if not 0 < step <= 1 or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("invalid TD audit step/horizon")
    eye = np.eye(len(p))
    maps = np.array([[eye-step*np.outer(eye[s], eye[s]-discount*eye[j])
                      for j in range(len(p))] for s in range(len(p))])
    q = np.zeros((len(p), len(p), len(p)))
    for _ in range(horizon):
        next_q = np.repeat(eye[None], len(p), axis=0)
        for s in range(len(p)):
            for j in range(len(p)):
                next_q[s] += p[s, j]*(maps[s, j].T @ q[j] @ maps[s, j])
        q = next_q
    return q


def oracle_line_transfer(metric, error, donor_error):
    """Exact one-donor QP using privileged error/metric; audit only."""
    q, e, donor = map(lambda x: np.asarray(x, dtype=float), (metric, error, donor_error))
    d = donor-e
    curvature = float(d@q@d)
    linear = float(d@q@e)
    beta = 0. if curvature <= 0 else float(np.clip(-linear/curvature, 0., 1.))
    changed = e+beta*d
    return beta, float(changed@q@changed-e@q@e)


def oracle_policy_cost_and_advantage(transition, discount, step, error, donor, state, horizon):
    """Enumerate a small oracle transfer policy, not a scalable experiment.

    Returns its expected total risk and sum of local-baseline advantages.
    At each node the remaining-horizon QP includes no transfer as beta=0.
    """
    p, e = np.asarray(transition, dtype=float), np.asarray(error, dtype=float)
    q = markov_td_risk_gramian(p, discount, step, horizon)[state]
    beta, advantage = oracle_line_transfer(q, e, donor)
    changed = e+beta*(np.asarray(donor)-e)
    cost = float(changed@changed)
    if horizon > 1:
        eye = np.eye(len(p))
        for j in range(len(p)):
            b = eye-step*np.outer(eye[state], eye[state]-discount*eye[j])
            future_cost, future_advantage = oracle_policy_cost_and_advantage(
                p, discount, step, b@changed, donor, j, horizon-1)
            cost += p[state, j]*future_cost
            advantage += p[state, j]*future_advantage
    return cost, advantage


def delayed_consequence_counterexample():
    # Public reversible, nonzero-memory MRP: eigenvalues of P are 1 and .5.
    p = np.array([[.9, .1], [.4, .6]])
    gamma, step, horizon = .9, .5, 64
    mu = stationary_distribution(p)
    conditional = markov_td_risk_gramian(p, gamma, step, horizon)
    q = np.einsum("s,sij->ij", mu, conditional)
    eigenvalues, vectors = np.linalg.eigh(q)
    before = vectors[:, 0]
    after = .8*vectors[:, -1]
    # Deliberate oracle-constructed counterexample, NOT a deployable action.
    immediate = float(after@after-before@before)
    future = float(after@q@after-before@q@before)
    beta_future, advantage_future = oracle_line_transfer(q, before, after)
    beta_now, _ = oracle_line_transfer(np.eye(2), before, after)
    euclidean_choice = before+beta_now*(after-before)
    return {"transition": p.tolist(), "stationary": mu.tolist(),
            "discount": gamma, "td_step": step, "horizon": horizon,
            "risk_metric_eigenvalues": eigenvalues.tolist(),
            "initial_error": before.tolist(), "post_transfer_error": after.tolist(),
            "immediate_squared_error_change": immediate,
            "expected_cumulative_squared_error_change": future,
            "oracle_future_risk_QP_beta": beta_future,
            "oracle_future_risk_QP_advantage": advantage_future,
            "oracle_immediate_error_QP_beta": beta_now,
            "future_advantage_of_immediate_error_QP": float(euclidean_choice@q@euclidean_choice-before@q@before),
            "scope": "zero-reward homogeneous tabular TD; exact Markov jump moments; oracle-selected witness, not efficacy"}


def report():
    p = np.array([[.7, .3], [.2, .8]])
    r, gamma, horizon = np.array([0., 1.]), .9, 8
    mean, second = return_moments(p, r, gamma, horizon)
    infinite_value = value_oracle(p, r, gamma)
    tail = -gamma**horizon*np.linalg.matrix_power(p, horizon)@infinite_value
    symmetric, mu = dirichlet_operator(p, gamma)
    td_operator = np.diag(mu) @ (np.eye(2)-gamma*p)
    return {
        "kind": "exact_public_model_interface_audit_not_efficacy",
        "td_direct_plugin": td_label_counterexample(),
        "bounded_return_interface": {
            "horizon": horizon, "finite_horizon_value": mean.tolist(),
            "return_variance": (second-mean**2).tolist(),
            "infinite_horizon_value": infinite_value.tolist(),
            "truncation_bias": (mean-infinite_value).tolist(),
            "tail_identity_residual": float(np.max(np.abs(mean-infinite_value-tail))),
            "uniform_tail_bound": float(gamma**horizon/(1-gamma)),
            "transitions_per_full_length_return": horizon,
            "qualification": "conditionally centered for finite-H target with fresh rollout; NOT automatically infinite-horizon TD"
        },
        "reversible_dirichlet": {"operator_identity_residual": float(np.max(np.abs(symmetric-td_operator))),
                                  "novelty": "inherited_geometry"},
        "nonreversible_dirichlet": nonreversible_counterexample(),
        "Bellman_vs_value_ranking": residual_ranking_counterexample(),
        "recursive_comparator": history_mismatch(),
        "future_risk_vs_immediate_error": delayed_consequence_counterexample(),
        "decision": {"readout_as_standalone_icml_candidate": "reject",
                     "reference_retained_as_baseline": True,
                     "direct_TD_plugin": "reject",
                     "new_efficacy_pilot_authorized": False,
                     "formal_authorized": False, "gpu_authorized": False}
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, sort_keys=True, allow_nan=False))
