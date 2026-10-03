from __future__ import annotations

import numpy as np

from .composite_cache_lyapunov import (
    cache_mismatch_energy,
    donor_receipt_energy_increment,
    donor_receipt_increment_upper,
    refresh_reset_benefit,
    stale_edge_activation_margin,
    topology_motion_increment,
    topology_motion_positive_upper,
)


def test_refresh_benefit_equals_exact_energy_drop() -> None:
    current = {0: np.array([0.7, -0.2]), 1: np.array([0.1, 0.4])}
    caches = {
        (0, 1): np.array([0.2, 0.3]),
        (1, 0): np.array([-0.1, 0.5]),
    }
    weights = {(0, 1): 1.7, (1, 0): 0.8}
    before = cache_mismatch_energy(current, caches, weights)
    benefit = refresh_reset_benefit(current[0], caches[(0, 1)], weights[(0, 1)])
    refreshed = dict(caches)
    refreshed[(0, 1)] = current[0].copy()
    after = cache_mismatch_energy(current, refreshed, weights)
    assert np.isclose(before - after, benefit)


def test_donor_receipt_identity_matches_direct_energy_difference() -> None:
    current = np.array([0.7, -0.2])
    gradient = np.array([0.3, -0.5])
    caches = {1: np.array([0.2, 0.3]), 2: np.array([0.9, -0.4])}
    weights = {1: 1.7, 2: 0.8}
    step = 0.06
    direct_before = 0.5 * sum(
        weights[i] * np.sum((current - caches[i]) ** 2) for i in caches
    )
    next_current = current - step * gradient
    direct_after = 0.5 * sum(
        weights[i] * np.sum((next_current - caches[i]) ** 2) for i in caches
    )
    identity = donor_receipt_energy_increment(
        current=current,
        gradient=gradient,
        step=step,
        cache_by_recipient=caches,
        weight_by_recipient=weights,
    )
    assert np.isclose(direct_after - direct_before, identity)


def test_receipt_upper_bound_dominates_exact_increment() -> None:
    current = np.array([0.7, -0.2])
    gradient = np.array([0.3, -0.5])
    caches = {1: np.array([0.2, 0.3]), 2: np.array([0.9, -0.4])}
    weights = {1: 1.7, 2: 0.8}
    weighted = sum(weights[i] * (current - caches[i]) for i in caches)
    exact = donor_receipt_energy_increment(
        current=current,
        gradient=gradient,
        step=0.06,
        cache_by_recipient=caches,
        weight_by_recipient=weights,
    )
    upper = donor_receipt_increment_upper(
        step=0.06,
        gradient_norm_upper=float(np.linalg.norm(gradient)),
        weighted_cache_displacement_norm=float(np.linalg.norm(weighted)),
        total_outgoing_weight=sum(weights.values()),
    )
    assert exact <= upper + 1e-12


def test_activation_margin_encodes_non_absorption_condition() -> None:
    assert stale_edge_activation_margin(
        reset_benefit=3.0,
        learning_drift_disadvantage=0.1,
        learning_weight=4.0,
        communication_queue=0.5,
        message_cost=1.0,
    ) > 0.0
    assert stale_edge_activation_margin(
        reset_benefit=0.1,
        learning_drift_disadvantage=0.1,
        learning_weight=4.0,
        communication_queue=0.5,
        message_cost=1.0,
    ) < 0.0


def test_topology_motion_is_exact_fixed_universe_energy_difference() -> None:
    current = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    caches = {
        (0, 2): np.array([1.0, 0.0]),
        (1, 2): np.array([2.0, 1.0]),
    }
    old_weights = {(0, 2): 1.0, (1, 2): 0.0}
    new_weights = {(0, 2): 0.0, (1, 2): 1.0}
    before = cache_mismatch_energy(current, caches, old_weights)
    after = cache_mismatch_energy(current, caches, new_weights)
    jump = topology_motion_increment(current, caches, old_weights, new_weights)
    assert jump == 2.0
    assert after - before == jump


def test_active_edge_only_energy_cannot_silently_telescope() -> None:
    current = {0: np.array([0.0]), 1: np.array([0.0])}
    caches = {(0, 2): np.array([0.0]), (1, 2): np.array([3.0])}
    old_weights = {(0, 2): 1.0, (1, 2): 0.0}
    new_weights = {(0, 2): 0.0, (1, 2): 1.0}
    # No parameter update and no refresh occurs, yet changing which edge is
    # active creates a positive Lyapunov jump that an active-edge sum misses.
    jump = topology_motion_increment(current, caches, old_weights, new_weights)
    assert jump == 4.5
    assert topology_motion_positive_upper(
        policy_cache_diameter_upper=3.0,
        old_weight_by_edge=old_weights,
        new_weight_by_edge=new_weights,
    ) == jump


def test_topology_motion_upper_ignores_helpful_edge_removal() -> None:
    old_weights = {(0, 2): 1.0, (1, 2): 0.5}
    new_weights = {(0, 2): 0.0, (1, 2): 0.25}
    assert topology_motion_positive_upper(
        policy_cache_diameter_upper=10.0,
        old_weight_by_edge=old_weights,
        new_weight_by_edge=new_weights,
    ) == 0.0


def test_topology_refresh_and_receipt_energy_events_do_not_double_count() -> None:
    current = {0: np.array([0.7, -0.2]), 1: np.array([0.1, 0.4])}
    caches = {
        (0, 2): np.array([0.2, 0.3]),
        (1, 2): np.array([-0.1, 0.5]),
    }
    old_weights = {(0, 2): 0.5, (1, 2): 1.0}
    new_weights = {(0, 2): 1.5, (1, 2): 0.8}
    initial = cache_mismatch_energy(current, caches, old_weights)
    topology = topology_motion_increment(
        current, caches, old_weights, new_weights
    )

    refresh = refresh_reset_benefit(
        current[1], caches[(1, 2)], new_weights[(1, 2)]
    )
    refreshed_caches = dict(caches)
    refreshed_caches[(1, 2)] = current[1].copy()

    gradient = np.array([0.3, -0.5])
    step = 0.06
    receipt = donor_receipt_energy_increment(
        current=current[0],
        gradient=gradient,
        step=step,
        cache_by_recipient={2: refreshed_caches[(0, 2)]},
        weight_by_recipient={2: new_weights[(0, 2)]},
    )
    final_current = dict(current)
    final_current[0] = current[0] - step * gradient
    final = cache_mismatch_energy(final_current, refreshed_caches, new_weights)
    assert np.isclose(final - initial, topology - refresh + receipt)
