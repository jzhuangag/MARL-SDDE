from __future__ import annotations

import pytest

from .coupled_actor_critic_finite_time import coupled_finite_time_bound


BASE = {
    "event_count": 400,
    "coverage_window": 5,
    "lyapunov_gap": 2.0,
    "actor_progress_scale_min": 0.08,
    "actor_radius_upper": 0.03,
    "critic_gradient_radius_upper": 0.02,
    "critic_contraction": 0.5,
    "critic_step_comparator": 0.4,
    "critic_target_sensitivity_upper": 0.6,
    "critic_weight": 0.7,
    "cross_motion_constant": 1.2,
    "squared_update_path_length": 0.5,
}


def test_finite_time_bound_matches_theorem_algebra() -> None:
    result = coupled_finite_time_bound(**BASE)
    critic_floor = 0.02 / 0.5
    critic_mean = 4 * critic_floor**2 + 2 * 2.0 / (0.7 * 0.4 * 0.5 * 400)
    actor_floor = 8 * 0.03**2 + 8 * 0.7**2 * 0.6**2 * critic_mean
    activated = 4 * 2.0 / 0.08 + 2 * 400 * actor_floor
    full = (2 * 5 * activated + 2 * 5**2 * 1.2 * 0.5) / (400 - 5 + 1)
    assert result.critic_floor == pytest.approx(critic_floor)
    assert result.mean_squared_critic_radius == pytest.approx(critic_mean)
    assert result.mean_squared_actor_floor == pytest.approx(actor_floor)
    assert result.activated_gradient_sum == pytest.approx(activated)
    assert result.full_gradient_average == pytest.approx(full)


def test_more_events_reduce_zero_floor_optimization_term() -> None:
    values = dict(BASE)
    values.update(
        actor_radius_upper=0.0,
        critic_gradient_radius_upper=0.0,
        squared_update_path_length=0.0,
    )
    short = coupled_finite_time_bound(**values)
    values["event_count"] = 1600
    values["actor_progress_scale_min"] = 0.04
    long = coupled_finite_time_bound(**values)
    assert long.full_gradient_average < short.full_gradient_average


def test_statistical_radii_create_an_explicit_nonvanishing_floor() -> None:
    positive = coupled_finite_time_bound(**BASE)
    zero = coupled_finite_time_bound(
        **{
            **BASE,
            "actor_radius_upper": 0.0,
            "critic_gradient_radius_upper": 0.0,
        }
    )
    assert positive.mean_squared_actor_floor > 0.0
    assert positive.full_gradient_average > zero.full_gradient_average


@pytest.mark.parametrize(
    "mutation",
    [
        {"event_count": 0},
        {"coverage_window": 401},
        {"actor_progress_scale_min": 0.0},
        {"critic_contraction": 0.0},
        {"critic_step_comparator": 3.0},
        {"squared_update_path_length": -0.1},
    ],
)
def test_invalid_finite_time_inputs_are_rejected(mutation: dict[str, float]) -> None:
    values = dict(BASE)
    values.update(mutation)
    with pytest.raises(ValueError):
        coupled_finite_time_bound(**values)
