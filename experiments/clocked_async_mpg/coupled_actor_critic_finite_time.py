"""Closed-form consequences of the conditional coupled Lyapunov drift."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CoupledFiniteTimeBound:
    critic_floor: float
    mean_squared_critic_radius: float
    mean_squared_actor_floor: float
    activated_gradient_sum: float
    full_gradient_average: float


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def coupled_finite_time_bound(
    *,
    event_count: int,
    coverage_window: int,
    lyapunov_gap: float,
    actor_progress_scale_min: float,
    actor_radius_upper: float,
    critic_gradient_radius_upper: float,
    critic_contraction: float,
    critic_step_comparator: float,
    critic_target_sensitivity_upper: float,
    critic_weight: float,
    cross_motion_constant: float,
    squared_update_path_length: float,
) -> CoupledFiniteTimeBound:
    """Evaluate the robust finite-time stationarity certificate.

    ``coverage_window`` means every actor block is activated at least once in
    every such event window.  ``squared_update_path_length`` is the sum of the
    squared norms of all applied actor increments, so no constant-step
    approximation is hidden in the interface.
    """

    if isinstance(event_count, bool) or int(event_count) != event_count:
        raise ValueError("event_count must be an integer")
    if isinstance(coverage_window, bool) or int(coverage_window) != coverage_window:
        raise ValueError("coverage_window must be an integer")
    events = int(event_count)
    window = int(coverage_window)
    if events <= 0 or window <= 0 or window > events:
        raise ValueError("require event_count >= coverage_window >= 1")
    gap = _nonnegative("lyapunov_gap", lyapunov_gap)
    scale = _nonnegative("actor_progress_scale_min", actor_progress_scale_min)
    actor_radius = _nonnegative("actor_radius_upper", actor_radius_upper)
    critic_noise = _nonnegative(
        "critic_gradient_radius_upper", critic_gradient_radius_upper
    )
    contraction = _nonnegative("critic_contraction", critic_contraction)
    critic_step = _nonnegative("critic_step_comparator", critic_step_comparator)
    target = _nonnegative(
        "critic_target_sensitivity_upper", critic_target_sensitivity_upper
    )
    weight = _nonnegative("critic_weight", critic_weight)
    motion = _nonnegative("cross_motion_constant", cross_motion_constant)
    path = _nonnegative("squared_update_path_length", squared_update_path_length)
    if scale == 0.0 or contraction == 0.0 or critic_step == 0.0 or weight == 0.0:
        raise ValueError("progress scale, contraction, critic step and weight must be positive")
    if critic_step * contraction > 1.0 + 1e-12:
        raise ValueError("critic_step_comparator exceeds the contraction interval")

    critic_floor = critic_noise / contraction
    critic_mean_square = (
        4.0 * critic_floor**2
        + 2.0 * gap / (weight * critic_step * contraction * events)
    )
    actor_floor_mean_square = (
        8.0 * actor_radius**2
        + 8.0 * weight**2 * target**2 * critic_mean_square
    )
    activated_sum = (
        4.0 * gap / scale + 2.0 * events * actor_floor_mean_square
    )
    denominator = events - window + 1
    full_average = (
        2.0 * window * activated_sum
        + 2.0 * window**2 * motion * path
    ) / denominator
    return CoupledFiniteTimeBound(
        critic_floor=float(critic_floor),
        mean_squared_critic_radius=float(critic_mean_square),
        mean_squared_actor_floor=float(actor_floor_mean_square),
        activated_gradient_sum=float(activated_sum),
        full_gradient_average=float(full_average),
    )
