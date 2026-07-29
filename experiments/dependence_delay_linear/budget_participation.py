"""Resource-matched participation surfaces for delayed Markov learning.

EXP-005A is deliberately an oracle-surface falsification experiment.  It
enumerates a logarithmic participation grid and a scalar step-size grid, while
charging only the selected agents.  No online controller is evaluated here.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from linear_model import make_agent_delays


AGENT_COUNTS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
RHO_VALUES: Tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)
MAX_DELAYS: Tuple[int, ...] = (4, 16)
ALIGNMENTS: Tuple[str, ...] = ("server_time", "sample_time")
SELECTION_RULES: Tuple[str, ...] = ("fastest", "uniform_rank")


@dataclass(frozen=True)
class BudgetConfig:
    num_agents: int = 32
    delay_exponent: float = 1.25
    eta_min: float = 0.0025
    eta_max: float = 0.08
    eta_count: int = 17
    message_budget: float = 6400.0
    primary_overhead: float = 4.0
    sensitivity_overheads: Tuple[float, ...] = (0.0, 16.0)
    wallclock_budget: float = 800.0
    wallclock_delay_weight: float = 0.25
    wallclock_message_weight: float = 0.02

    @property
    def eta_grid(self) -> np.ndarray:
        return np.geomspace(self.eta_min, self.eta_max, self.eta_count)


def selected_indices(
    num_agents: int,
    selected_count: int,
    rule: str,
) -> np.ndarray:
    """Return deterministic selected-agent indices for one participation rule."""

    if selected_count not in AGENT_COUNTS:
        raise ValueError("selected_count must be on the registered grid")
    if selected_count > num_agents:
        raise ValueError("selected_count cannot exceed num_agents")
    if rule == "fastest":
        return np.arange(selected_count, dtype=int)
    if rule == "uniform_rank":
        if selected_count == 1:
            return np.asarray([num_agents // 2], dtype=int)
        return np.rint(
            np.linspace(0, num_agents - 1, selected_count)
        ).astype(int)
    raise ValueError("unknown selection rule: {0}".format(rule))


def selected_delays(
    max_delay: int,
    selected_count: int,
    rule: str,
    config: BudgetConfig,
) -> np.ndarray:
    delays = make_agent_delays(
        max_agents=config.num_agents,
        max_delay=max_delay,
        exponent=config.delay_exponent,
    )
    return delays[selected_indices(config.num_agents, selected_count, rule)]


def resource_specs(config: BudgetConfig) -> Dict[str, Dict[str, float]]:
    specs: Dict[str, Dict[str, float]] = {
        "message_overhead_4": {
            "kind": "message",
            "budget": config.message_budget,
            "overhead": config.primary_overhead,
        },
        "wallclock": {
            "kind": "wallclock",
            "budget": config.wallclock_budget,
            "delay_weight": config.wallclock_delay_weight,
            "message_weight": config.wallclock_message_weight,
        },
    }
    for overhead in config.sensitivity_overheads:
        specs["message_overhead_{0}".format(int(overhead))] = {
            "kind": "message",
            "budget": config.message_budget,
            "overhead": float(overhead),
        }
    return specs


def per_update_cost(
    selected_count: int,
    delays: Iterable[int],
    spec: Dict[str, float],
) -> float:
    values = np.asarray(list(delays), dtype=int)
    if spec["kind"] == "message":
        return float(selected_count + spec["overhead"])
    if spec["kind"] == "wallclock":
        return float(
            1.0
            + spec["delay_weight"] * float(np.max(values))
            + spec["message_weight"] * selected_count
        )
    raise ValueError("unknown resource kind")


def budget_horizon(
    selected_count: int,
    delays: Iterable[int],
    spec: Dict[str, float],
) -> int:
    cost = per_update_cost(selected_count, delays, spec)
    return max(1, int(np.floor(spec["budget"] / cost)))

