"""Predictable two-state mixing-certificate controller for EXP-009A."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta

from markov_jump_ms import (
    AGENT_COUNTS_EXPANDING,
    aggregate_same_time_curvature,
    covariance_operator_coefficients,
    homogeneous_delays,
    minimum_decorrelation_gap,
    mixing_tv_after_gap,
    polynomial_matrix,
    registered_expanding_td_model,
    sharp_theorem_steps,
    spectral_radius_with_residual,
    thinned_persistence,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


RESOURCE_BUDGET = 20_000
SERVER_OVERHEAD = 8
PILOT_TRANSITIONS = 2_048
PILOT_ALPHA = 0.01
ADDITIVE_SCALE = 0.2
DIVERGENCE_THRESHOLD = 1e12


def clopper_pearson_upper(
    stays: int, transitions: int, alpha: float = PILOT_ALPHA
) -> float:
    if not 0 <= stays <= transitions:
        raise ValueError("invalid stay count")
    if stays == transitions:
        return 1.0
    return float(beta.ppf(1.0 - alpha, stays + 1, transitions - stays))


def controller_inputs() -> Tuple[Dict[str, np.ndarray], float, float]:
    model = registered_expanding_td_model()
    lipschitz = float(
        max(
            np.linalg.norm(matrix, ord=2)
            for matrix in model["jacobians"]
        )
    )
    monotonicity = float(model["stationary_mean"][0, 0])
    return model, lipschitz, monotonicity


def select_action(
    persistence_upper: float,
    rho: float,
    delay: int,
    pilot_cost: int,
    fixed_q: Optional[int] = None,
) -> Dict[str, float]:
    model, lipschitz, monotonicity = controller_inputs()
    target_delta = monotonicity / (4.0 * lipschitz)
    if persistence_upper >= 1.0:
        return {
            "persistence_upper": 1.0,
            "gap": RESOURCE_BUDGET + 1,
            "delta_upper": 0.5,
            "num_agents": int(fixed_q or 1),
            "eta": 0.0,
            "contraction": 1.0,
            "updates": 0,
            "risk_surrogate": 1.0,
            "pilot_cost": int(pilot_cost),
        }
    persistence_upper = max(0.5, float(persistence_upper))
    gap = minimum_decorrelation_gap(persistence_upper, target_delta)
    delta_upper = mixing_tv_after_gap(persistence_upper, gap)
    candidates = (fixed_q,) if fixed_q is not None else AGENT_COUNTS_EXPANDING
    best = None
    for num_agents in candidates:
        delays = homogeneous_delays(int(num_agents), delay)
        theorem = sharp_theorem_steps(
            model,
            int(num_agents),
            rho,
            delays,
            delta_upper,
        )
        update_cost = gap + SERVER_OVERHEAD + int(num_agents)
        usable = max(0, RESOURCE_BUDGET - int(pilot_cost))
        updates = usable // update_cost
        blocks = updates // (2 * delay + 1)
        contraction = theorem["rate_contraction_coefficient"]
        noise = ADDITIVE_SCALE ** 2 * (
            rho + (1.0 - rho) / float(num_agents)
        )
        residual = (
            theorem["rate_eta"] ** 2
            * noise
            / max(1.0 - contraction, np.finfo(float).eps)
        )
        risk = contraction ** blocks + residual
        row = {
            "persistence_upper": persistence_upper,
            "gap": int(gap),
            "delta_upper": delta_upper,
            "num_agents": int(num_agents),
            "eta": theorem["rate_eta"],
            "contraction": contraction,
            "updates": int(updates),
            "risk_surrogate": float(risk),
            "pilot_cost": int(pilot_cost),
        }
        if best is None or (row["risk_surrogate"], row["num_agents"]) < (
            best["risk_surrogate"],
            best["num_agents"],
        ):
            best = row
    return best


def select_finite_budget_action(
    persistence_upper: float,
    rho: float,
    delay: int,
    pilot_cost: int,
    fixed_q: Optional[int] = None,
) -> Dict[str, float]:
    """Jointly select q and the theorem-safe finite-budget scalar step."""

    model, lipschitz, monotonicity = controller_inputs()
    target_delta = monotonicity / (4.0 * lipschitz)
    if persistence_upper >= 1.0:
        return {
            "persistence_upper": 1.0,
            "gap": RESOURCE_BUDGET + 1,
            "delta_upper": 0.5,
            "num_agents": int(fixed_q or 1),
            "eta": 0.0,
            "contraction": 1.0,
            "updates": 0,
            "risk_surrogate": 1.0,
            "pilot_cost": int(pilot_cost),
        }
    persistence_upper = max(0.5, float(persistence_upper))
    gap = minimum_decorrelation_gap(persistence_upper, target_delta)
    delta_upper = mixing_tv_after_gap(persistence_upper, gap)
    candidates = (fixed_q,) if fixed_q is not None else AGENT_COUNTS_EXPANDING
    best = None
    for num_agents in candidates:
        delays = homogeneous_delays(int(num_agents), delay)
        theorem = sharp_theorem_steps(
            model,
            int(num_agents),
            rho,
            delays,
            delta_upper,
        )
        update_cost = gap + SERVER_OVERHEAD + int(num_agents)
        usable = max(0, RESOURCE_BUDGET - int(pilot_cost))
        updates = usable // update_cost
        blocks = updates // (2 * delay + 1)
        noise = ADDITIVE_SCALE ** 2 * (
            rho + (1.0 - rho) / float(num_agents)
        )
        root = theorem["sharp_root"] * (1.0 - 1e-8)
        effective_monotonicity = theorem["effective_monotonicity"]
        effective_curvature = theorem["effective_curvature"]
        rms_delay = theorem["rms_delay"]

        def contraction(eta: float) -> float:
            base = (
                1.0
                - 2.0 * eta * effective_monotonicity
                + eta * eta * effective_curvature
            )
            factor = (
                np.sqrt(max(base, 0.0))
                + eta * eta * lipschitz * lipschitz * rms_delay
            )
            return float(factor * factor)

        def risk(eta: float) -> float:
            if eta <= 0.0:
                return 1.0
            coefficient = contraction(eta)
            residual = (
                eta * eta
                * noise
                / max(1.0 - coefficient, np.finfo(float).eps)
            )
            return float(coefficient ** blocks + residual)

        grid = root * np.geomspace(1e-5, 1.0, 161)
        values = np.asarray([risk(value) for value in grid])
        index = int(np.argmin(values))
        left = 0.0 if index == 0 else float(grid[index - 1])
        right = root if index == len(grid) - 1 else float(grid[index + 1])
        inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
        first = right - inverse_phi * (right - left)
        second = left + inverse_phi * (right - left)
        first_value = risk(first)
        second_value = risk(second)
        for _ in range(80):
            if first_value <= second_value:
                right = second
                second = first
                second_value = first_value
                first = right - inverse_phi * (right - left)
                first_value = risk(first)
            else:
                left = first
                first = second
                first_value = second_value
                second = left + inverse_phi * (right - left)
                second_value = risk(second)
        eta = 0.5 * (left + right)
        coefficient = contraction(eta)
        row = {
            "persistence_upper": persistence_upper,
            "gap": int(gap),
            "delta_upper": delta_upper,
            "num_agents": int(num_agents),
            "eta": float(eta),
            "contraction": coefficient,
            "updates": int(updates),
            "risk_surrogate": risk(eta),
            "pilot_cost": int(pilot_cost),
        }
        if best is None or (
            row["risk_surrogate"],
            row["num_agents"],
            row["eta"],
        ) < (
            best["risk_surrogate"],
            best["num_agents"],
            best["eta"],
        ):
            best = row
    return best


def select_joint_action(
    persistence_upper: float,
    rho: float,
    delay: int,
    pilot_cost: int,
    fixed_q: Optional[int] = None,
    resource_budget: int = RESOURCE_BUDGET,
) -> Dict[str, float]:
    """Jointly optimize participation, decorrelation gap, and safe step."""

    model, lipschitz, monotonicity = controller_inputs()
    if persistence_upper >= 1.0:
        return {
            "persistence_upper": 1.0,
            "gap": int(resource_budget) + 1,
            "delta_upper": 0.5,
            "num_agents": int(fixed_q or 1),
            "eta": 0.0,
            "contraction": 1.0,
            "updates": 0,
            "risk_surrogate": 1.0,
            "pilot_cost": int(pilot_cost),
        }
    persistence_upper = max(0.5, float(persistence_upper))
    positivity_target = (
        monotonicity / (2.0 * lipschitz) * (1.0 - 1e-8)
    )
    minimum_gap = minimum_decorrelation_gap(
        persistence_upper, positivity_target
    )
    ratios = np.geomspace(1.0, 4.0, 17)
    gaps = sorted(
        {
            max(minimum_gap, int(np.ceil(minimum_gap * ratio)))
            for ratio in ratios
        }
    )
    candidates = (fixed_q,) if fixed_q is not None else AGENT_COUNTS_EXPANDING
    curvatures = {
        int(num_agents): aggregate_same_time_curvature(
            model, int(num_agents), rho
        )[1]
        for num_agents in candidates
    }
    best = None
    for gap in gaps:
        delta_upper = mixing_tv_after_gap(persistence_upper, gap)
        for num_agents in candidates:
            effective_monotonicity = (
                monotonicity - 2.0 * lipschitz * delta_upper
            )
            effective_curvature = (
                curvatures[int(num_agents)]
                + 2.0 * lipschitz * lipschitz * delta_upper
            )
            rms_delay = float(delay)

            delay_coefficient = (
                lipschitz * lipschitz * rms_delay
            )
            if delay_coefficient == 0.0:
                root = (
                    2.0
                    * effective_monotonicity
                    / effective_curvature
                )
            else:
                roots = np.roots(
                    (
                        delay_coefficient ** 2,
                        0.0,
                        -(effective_curvature + 2.0 * delay_coefficient),
                        2.0 * effective_monotonicity,
                    )
                )
                positive = sorted(
                    float(value.real)
                    for value in roots
                    if abs(value.imag) <= 1e-9 and value.real > 0.0
                )
                if not positive:
                    raise RuntimeError(
                        "failed to solve the sharp delay boundary"
                    )
                root = positive[0]
            root *= 1.0 - 1e-8
            update_cost = gap + SERVER_OVERHEAD + int(num_agents)
            usable = max(0, int(resource_budget) - int(pilot_cost))
            updates = usable // update_cost
            blocks = updates // (2 * delay + 1)
            noise = ADDITIVE_SCALE ** 2 * (
                rho + (1.0 - rho) / float(num_agents)
            )

            def contraction(eta: float) -> float:
                base = (
                    1.0
                    - 2.0 * eta * effective_monotonicity
                    + eta * eta * effective_curvature
                )
                factor = (
                    np.sqrt(max(base, 0.0))
                    + eta
                    * eta
                    * lipschitz
                    * lipschitz
                    * rms_delay
                )
                return float(factor * factor)

            def risk(eta: float) -> float:
                if eta <= 0.0:
                    return 1.0
                coefficient = contraction(eta)
                return float(
                    coefficient ** blocks
                    + eta
                    * eta
                    * noise
                    / max(
                        1.0 - coefficient, np.finfo(float).eps
                    )
                )

            grid = root * np.geomspace(1e-5, 1.0, 41)
            base_grid = (
                1.0
                - 2.0 * grid * effective_monotonicity
                + grid * grid * effective_curvature
            )
            factor_grid = (
                np.sqrt(np.maximum(base_grid, 0.0))
                + grid * grid * delay_coefficient
            )
            coefficient_grid = factor_grid * factor_grid
            values = (
                coefficient_grid ** blocks
                + grid
                * grid
                * noise
                / np.maximum(
                    1.0 - coefficient_grid, np.finfo(float).eps
                )
            )
            index = int(np.argmin(values))
            left = (
                float(root * 1e-8)
                if index == 0
                else float(grid[index - 1])
            )
            right = (
                root
                if index == len(grid) - 1
                else float(grid[index + 1])
            )
            optimized = minimize_scalar(
                risk,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-12, "maxiter": 80},
            )
            eta = float(optimized.x)
            coefficient = contraction(eta)
            row = {
                "persistence_upper": persistence_upper,
                "gap": int(gap),
                "delta_upper": delta_upper,
                "num_agents": int(num_agents),
                "eta": float(eta),
                "contraction": coefficient,
                "updates": int(updates),
                "risk_surrogate": risk(eta),
                "pilot_cost": int(pilot_cost),
            }
            if best is None or (
                row["risk_surrogate"],
                row["num_agents"],
                row["gap"],
                row["eta"],
            ) < (
                best["risk_surrogate"],
                best["num_agents"],
                best["gap"],
                best["eta"],
            ):
                best = row
    return best


def exact_policy_metrics(
    action: Dict[str, float],
    true_persistence: float,
    rho: float,
    delay: int,
) -> Dict[str, float]:
    if action["updates"] == 0 or action["eta"] == 0.0:
        return {
            "exact_radius": 1.0,
            "exact_residual": 0.0,
            "expected_final_error": 1.0,
            "used_persistence": true_persistence,
        }
    model = registered_expanding_td_model()
    num_agents = int(action["num_agents"])
    delays = homogeneous_delays(num_agents, delay)
    used_persistence = thinned_persistence(
        true_persistence, int(action["gap"])
    )
    coefficients = covariance_operator_coefficients(
        model, delays, rho, used_persistence
    )["markov"]
    eta = float(action["eta"])
    radius, residual = spectral_radius_with_residual(coefficients, eta)
    operator = polynomial_matrix(coefficients, eta)
    lifted_dimension = delay + 1
    covariance_block = lifted_dimension * lifted_dimension
    initial_matrix = np.ones(
        (lifted_dimension, lifted_dimension), dtype=float
    )
    initial = np.concatenate(
        (
            0.5 * initial_matrix.reshape(-1, order="F"),
            0.5 * initial_matrix.reshape(-1, order="F"),
        )
    )
    noise_variance = ADDITIVE_SCALE ** 2 * (
        rho + (1.0 - rho) / float(num_agents)
    )
    forcing = np.zeros(2 * covariance_block, dtype=float)
    forcing[0] = 0.5 * eta * eta * noise_variance
    forcing[covariance_block] = 0.5 * eta * eta * noise_variance
    augmented = np.zeros(
        (len(initial) + 1, len(initial) + 1), dtype=float
    )
    augmented[:-1, :-1] = operator
    augmented[:-1, -1] = forcing
    augmented[-1, -1] = 1.0
    state = np.concatenate((initial, np.asarray((1.0,))))
    final = np.linalg.matrix_power(
        augmented, int(action["updates"])
    ).dot(state)
    expected = float(final[0] + final[covariance_block])
    return {
        "exact_radius": radius,
        "exact_residual": residual,
        "expected_final_error": expected,
        "used_persistence": used_persistence,
    }


if njit is not None:

    @njit(cache=True, nogil=True)
    def _categorical_four(weights: np.ndarray) -> int:
        uniform = np.random.random()
        cumulative = 0.0
        for index in range(4):
            cumulative += weights[index]
            if uniform <= cumulative:
                return index
        return 3


    @njit(cache=True, nogil=True)
    def _categorical_sum(
        count: int, weights: np.ndarray, values: np.ndarray
    ) -> float:
        """Sample an exact four-category multinomial sum in O(1)."""

        remaining = count
        remaining_probability = 1.0
        total = 0.0
        for index in range(3):
            if remaining == 0:
                break
            conditional = weights[index] / remaining_probability
            selected = np.random.binomial(remaining, conditional)
            total += selected * values[index]
            remaining -= selected
            remaining_probability -= weights[index]
        total += remaining * values[3]
        return total


    @njit(cache=True, nogil=True)
    def _simulate_policy_kernel(
        seed: int,
        weights: np.ndarray,
        jacobians: np.ndarray,
        true_persistence: float,
        rho: float,
        gap: int,
        num_agents: int,
        eta: float,
        delay: int,
        updates: int,
        additive_scale: float,
        divergence_threshold: float,
    ) -> Tuple[float, float, bool]:
        np.random.seed(seed)
        used_persistence = 0.5 * (
            1.0 + (2.0 * true_persistence - 1.0) ** gap
        )
        mode = int(np.random.random() >= 0.5)
        history = np.ones(updates + delay + 1, dtype=np.float64)
        maximum_error = 1.0
        share_probability = np.sqrt(rho)
        noise_deviation = additive_scale * np.sqrt(
            rho + (1.0 - rho) / num_agents
        )
        for update in range(updates):
            if np.random.random() > used_persistence:
                mode = 1 - mode
            common = _categorical_four(weights[mode])
            common_count = np.random.binomial(
                num_agents, share_probability
            )
            independent_count = num_agents - common_count
            aggregate = common_count * jacobians[common]
            aggregate += _categorical_sum(
                independent_count, weights[mode], jacobians
            )
            aggregate /= num_agents
            noise = noise_deviation * np.random.randn()
            current_index = delay + update
            stale_index = current_index - delay
            new_error = history[current_index] - eta * (
                aggregate * history[stale_index] + noise
            )
            history[current_index + 1] = new_error
            squared = new_error * new_error
            if squared > maximum_error:
                maximum_error = squared
            if (
                not np.isfinite(squared)
                or squared > divergence_threshold
            ):
                return divergence_threshold, maximum_error, True
        final_error = history[delay + updates] ** 2
        return final_error, maximum_error, False


def simulate_policy(
    seed: int,
    action: Dict[str, float],
    true_persistence: float,
    rho: float,
    delay: int,
) -> Dict[str, object]:
    if action["updates"] == 0 or action["eta"] == 0.0:
        return {
            "final_error": 1.0,
            "maximum_error": 1.0,
            "diverged": False,
        }
    if njit is None:  # pragma: no cover
        raise RuntimeError("EXP-009A simulation requires numba")
    model = registered_expanding_td_model()
    final, maximum, diverged = _simulate_policy_kernel(
        int(seed),
        model["weights"],
        model["jacobians"].reshape(-1),
        float(true_persistence),
        float(rho),
        int(action["gap"]),
        int(action["num_agents"]),
        float(action["eta"]),
        int(delay),
        int(action["updates"]),
        ADDITIVE_SCALE,
        DIVERGENCE_THRESHOLD,
    )
    return {
        "final_error": float(final),
        "maximum_error": float(maximum),
        "diverged": bool(diverged),
    }
