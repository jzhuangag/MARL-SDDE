"""One substantive cost/precision redesign; not a native performance result.

Share one multi-epoch unit proposal across line-search candidates. Use a frozen
single-actor response basis plus an explicitly paid interaction remainder as
an alternative expected selection-risk bound. No population UCB is invented.
"""
import copy
import math
import numpy as np
import torch


def value_geometry(samples, action_coefficients, basis_indices, *, predictable_offset=None):
    """All representation choices must be fixed BEFORE this bank is opened.

    samples is the unclipped actual-return contrast matrix m x K. Basis indices
    identify pre-bank unit single-actor policies. Each candidate has a fixed
    nonnegative coefficient vector; it need not obey an additive game model.
    The nonadditive residual is measured, not assumed zero.
    """
    x = np.asarray(samples, dtype=float)
    a = np.asarray(action_coefficients, dtype=float)
    indices = tuple(basis_indices)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or a.shape != (x.shape[1], len(indices)):
        raise ValueError("finite bank and frozen response representation required")
    if not np.isfinite(x).all() or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("finite contrasts and nonnegative action coefficients required")
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= x.shape[1] for i in indices):
        raise ValueError("frozen valid basis indices required")
    offset = np.zeros(x.shape[1]) if predictable_offset is None else np.asarray(predictable_offset, dtype=float)
    if offset.shape != (x.shape[1],) or not np.isfinite(offset).all():
        raise ValueError("finite PRE-BANK offset required")
    z = x[:, list(indices)]
    residual = x - z @ a.T - offset
    factor = len(indices) * (float(a.max()) ** 2 if a.size else 0.)
    return {
        "full_vector": {"variance_statistic": float(x.var(axis=0, ddof=1).sum()), "remainder_statistic": 0.},
        "response_basis": {"variance_statistic": factor * float(z.var(axis=0, ddof=1).sum()),
                           "remainder_statistic": 4 * float(np.abs(residual).max(axis=1).mean())},
        "basis_dimension": len(indices), "candidate_count": x.shape[1],
        "representation_must_precede_bank": True, "is_high_probability_certificate": False,
    }


def expected_risk_feedback(geometry, family, episodes, young_scale):
    """Observable feedback whose expectation upper-bounds selection regret/V.

    It is NOT a pointwise certificate for the realized selected policy.
    Probe and family selection must precede the samples. Choosing the cheaper
    feedback post hoc does not preserve the all-option unbiased-loss argument.
    """
    if episodes < 2 or not math.isfinite(young_scale) or young_scale <= 0:
        raise ValueError("positive fixed scale and at least two independent episodes required")
    stats = geometry[family]
    return young_scale + stats["variance_statistic"] / (young_scale * episodes) + stats["remainder_statistic"]


def eligible_probe_bytes(observation_dimensions, ready_actor_indices, episodes, horizon):
    """Only changed actors need early private records; others' ratios equal one.

    Float32 observations, int64 actions and float64 behavior log probabilities.
    Public states/rewards are separately paid. This does not refund later full
    program transmission or pretend probe bytes were already delivered for free.
    """
    indices = tuple(ready_actor_indices)
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(observation_dimensions) for i in indices):
        raise ValueError("unique valid ready actors required")
    return episodes * horizon * sum(observation_dimensions[i]*4 + 16 for i in indices)


def interpolate_server_proposal(server_actor, unit_actor, alpha):
    """PPO-block line search, NOT 'Adam with learning rate alpha'.

    For alpha>0 retain only the server-owned unit-block moments. Parameter
    interpolation is a declared post-optimizer action; it executes no extra
    backward or Adam step. Idle retains the original server's moments.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("nonnegative finite line-search coefficient required")
    if alpha == 0:
        return copy.deepcopy(server_actor)
    if alpha == 1:
        return copy.deepcopy(unit_actor)
    result = copy.deepcopy(unit_actor)
    source = list(server_actor.actor.parameters()); target = list(result.actor.parameters())
    if len(source) != len(target) or any(a.shape != b.shape for a,b in zip(source,target)):
        raise ValueError("matching policy blocks required")
    with torch.no_grad():
        for old, new in zip(source,target):
            new.copy_(old + alpha * (new - old))
    if not all(torch.isfinite(p).all() for p in target):
        raise FloatingPointError("nonfinite interpolated policy")
    return result


def amortized_proposals(server_actor, arrays, alphas, epochs, *, budget=lambda: None):
    alphas = tuple(alphas)
    if not alphas or len(set(alphas)) != len(alphas) or 1. not in alphas or any(
            isinstance(a, bool) or not math.isfinite(a) or a < 0 for a in alphas):
        raise ValueError("unique finite nonnegative coefficients including the unit reference required")
    # Import only on native execution; this module itself creates no environment.
    from current_packet_native.integration import proposal
    unit, trace = proposal(server_actor, arrays, 1., epochs, budget=budget)
    variants = {alpha: interpolate_server_proposal(server_actor, unit, alpha) for alpha in alphas}
    entries = sum(p.numel() for p in unit.actor.parameters())
    def state_bytes(actor):
        tensors = list(actor.actor.state_dict().values())
        tensors += [v for s in actor.actor_optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
        return sum(t.numel()*t.element_size() for t in tensors)
    return variants, dict(unit_proposal=trace, actor_adam_calls=epochs,
        actor_backward_rows=len(arrays[0])*epochs,
        line_search_coefficients=list(alphas),
        interpolation_parameter_entries=entries*sum(alpha not in (0.,1.) for alpha in alphas),
        modeled_state_tensor_copy_bytes=state_bytes(server_actor)+sum(
            state_bytes(server_actor if alpha == 0 else unit) for alpha in alphas),
        primitive="one_server_PPO_block_then_parameter_line_search",
        worker_optimizer_imported=False, positive_step_optimizer_follows_unit_block=True)
