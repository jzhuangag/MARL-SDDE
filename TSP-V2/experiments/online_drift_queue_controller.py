"""Online Lyapunov-bandit participation controller.

The controller observes only the selected action's bounded, disjoint
validation progress.  It never reuses a locally calibrated drift coefficient
as a uniform long-horizon certificate.  Known action costs include training
and validation, and a reserve-aware feasibility mask prevents either physical
budget from being exceeded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class OnlineActionCost:
    messages: int
    environment_ticks: int

    def __post_init__(self) -> None:
        if min(self.messages, self.environment_ticks) <= 0:
            raise ValueError("action costs must be positive")


@dataclass(frozen=True)
class OnlineDecision:
    action: int
    probabilities: dict[int, float]
    feasible_actions: tuple[int, ...]
    message_remaining_before: int
    environment_remaining_before: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OnlineObservation:
    action: int
    progress: float
    bounded_reward: float
    message_remaining_after: int
    environment_remaining_after: int
    message_queue: float
    environment_queue: float

    def to_dict(self) -> dict:
        return asdict(self)


class OnlineDriftQueueController:
    """Finite-catalogue controller with EXP3 learning and Lyapunov queues.

    The hard feasibility mask is pathwise.  The queues are a design signal:
    they price actions whose known costs exceed the registered per-decision
    allowances.  The bandit reward is the selected action's observed decrease
    in a disjoint validation Lyapunov risk, mapped from ``[-1,1]`` to
    ``[0,1]``.
    """

    def __init__(
        self,
        *,
        costs: Mapping[int, OnlineActionCost],
        total_messages: int,
        total_environment_ticks: int,
        decisions: int,
        eta: float,
        exploration: float,
        progress_scale: float,
        forgetting: float,
        queue_weight: float,
        seed: int,
    ) -> None:
        if not costs or any(q <= 0 for q in costs):
            raise ValueError("cost catalogue must be nonempty and positive")
        if min(total_messages, total_environment_ticks, decisions) <= 0:
            raise ValueError("budgets and decision horizon must be positive")
        if eta <= 0.0 or progress_scale <= 0.0 or queue_weight <= 0.0:
            raise ValueError("eta, progress_scale, and queue_weight must be positive")
        if not 0.0 <= exploration < 1.0:
            raise ValueError("exploration must lie in [0,1)")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0,1]")
        self.costs = dict(sorted(costs.items()))
        self.actions = tuple(self.costs)
        self.total_messages = int(total_messages)
        self.total_environment_ticks = int(total_environment_ticks)
        self.total_decisions = int(decisions)
        self.message_remaining = int(total_messages)
        self.environment_remaining = int(total_environment_ticks)
        self.decisions_remaining = int(decisions)
        self.eta = float(eta)
        self.exploration = float(exploration)
        self.progress_scale = float(progress_scale)
        self.forgetting = float(forgetting)
        self.queue_weight = float(queue_weight)
        self.message_allowance = total_messages / decisions
        self.environment_allowance = total_environment_ticks / decisions
        self.message_queue = 0.0
        self.environment_queue = 0.0
        self.log_weights = {q: 0.0 for q in self.actions}
        self.rng = np.random.default_rng(seed)
        self._pending: OnlineDecision | None = None

        cheapest_messages = min(c.messages for c in self.costs.values())
        cheapest_environment = min(c.environment_ticks for c in self.costs.values())
        if (
            cheapest_messages * decisions > total_messages
            or cheapest_environment * decisions > total_environment_ticks
        ):
            raise ValueError("budgets cannot support the cheapest action horizon")

    def _reserve_feasible_actions(self) -> tuple[int, ...]:
        if self.decisions_remaining <= 0:
            return ()
        future = self.decisions_remaining - 1
        min_messages = min(c.messages for c in self.costs.values())
        min_environment = min(c.environment_ticks for c in self.costs.values())
        feasible = []
        for q, cost in self.costs.items():
            if (
                cost.messages + future * min_messages <= self.message_remaining
                and cost.environment_ticks + future * min_environment
                <= self.environment_remaining
            ):
                feasible.append(q)
        return tuple(feasible)

    def probabilities(self) -> dict[int, float]:
        """Return current probabilities without consuming randomness."""

        if self._pending is not None:
            raise RuntimeError("observe the pending action before another decision")
        feasible = self._reserve_feasible_actions()
        if not feasible:
            raise RuntimeError("no reserve-feasible action remains")
        logits = []
        for q in feasible:
            cost = self.costs[q]
            normalized_price = (
                self.message_queue * cost.messages / self.message_allowance
                + self.environment_queue
                * cost.environment_ticks
                / self.environment_allowance
            )
            logits.append(self.log_weights[q] - normalized_price / self.queue_weight)
        logits_array = np.asarray(logits, dtype=float)
        logits_array -= float(np.max(logits_array))
        softmax = np.exp(logits_array)
        softmax /= float(np.sum(softmax))
        mixed = (1.0 - self.exploration) * softmax + self.exploration / len(feasible)
        return {q: float(p) for q, p in zip(feasible, mixed, strict=True)}

    def decide(self) -> OnlineDecision:
        probabilities = self.probabilities()
        feasible = tuple(probabilities)
        probability_array = np.asarray([probabilities[q] for q in feasible])
        action = int(self.rng.choice(feasible, p=probability_array))
        decision = OnlineDecision(
            action=action,
            probabilities=probabilities,
            feasible_actions=feasible,
            message_remaining_before=self.message_remaining,
            environment_remaining_before=self.environment_remaining,
        )
        self._pending = decision
        return decision

    def observe(self, *, risk_before: float, risk_after: float) -> OnlineObservation:
        """Consume one bounded, disjoint validation observation."""

        if self._pending is None:
            raise RuntimeError("decide must precede observe")
        if not (0.0 <= risk_before <= 1.0 and 0.0 <= risk_after <= 1.0):
            raise ValueError("validation risks must lie in [0,1]")
        action = self._pending.action
        probability = self._pending.probabilities[action]
        progress = float(risk_before - risk_after)
        centered_gain = float(np.clip(progress / self.progress_scale, -1.0, 1.0))
        bounded_reward = 0.5 * (centered_gain + 1.0)
        # The common 1/2 offset in ``bounded_reward`` carries no action
        # information.  Importance-weighting that offset would add avoidable
        # variance, so exponential weights uses the signed centered gain.
        importance_reward = centered_gain / probability
        for q in self.actions:
            self.log_weights[q] *= self.forgetting
        self.log_weights[action] += self.eta * importance_reward

        cost = self.costs[action]
        self.message_remaining -= cost.messages
        self.environment_remaining -= cost.environment_ticks
        self.decisions_remaining -= 1
        self.message_queue = max(
            0.0, self.message_queue + cost.messages - self.message_allowance
        )
        self.environment_queue = max(
            0.0,
            self.environment_queue
            + cost.environment_ticks
            - self.environment_allowance,
        )
        if self.message_remaining < 0 or self.environment_remaining < 0:
            raise AssertionError("reserve feasibility failed to protect a budget")
        observation = OnlineObservation(
            action=action,
            progress=progress,
            bounded_reward=bounded_reward,
            message_remaining_after=self.message_remaining,
            environment_remaining_after=self.environment_remaining,
            message_queue=self.message_queue,
            environment_queue=self.environment_queue,
        )
        self._pending = None
        return observation


class SlidingWindowDriftQueueController:
    """Conservative sliding-window UCB with Lyapunov budget queues."""

    def __init__(
        self,
        *,
        costs: Mapping[int, OnlineActionCost],
        total_messages: int,
        total_environment_ticks: int,
        decisions: int,
        window: int,
        bonus_scale: float,
        probe_interval: int,
        progress_scale: float,
        queue_weight: float,
    ) -> None:
        if not costs or any(q <= 0 for q in costs):
            raise ValueError("cost catalogue must be nonempty and positive")
        if min(total_messages, total_environment_ticks, decisions) <= 0:
            raise ValueError("budgets and decision horizon must be positive")
        if min(window, probe_interval) <= 0:
            raise ValueError("window and probe interval must be positive")
        if min(bonus_scale, progress_scale, queue_weight) <= 0.0:
            raise ValueError("scales and queue_weight must be positive")
        self.costs = dict(sorted(costs.items()))
        self.actions = tuple(self.costs)
        self.total_decisions = int(decisions)
        self.message_remaining = int(total_messages)
        self.environment_remaining = int(total_environment_ticks)
        self.decisions_remaining = int(decisions)
        self.message_allowance = total_messages / decisions
        self.environment_allowance = total_environment_ticks / decisions
        self.window = int(window)
        self.bonus_scale = float(bonus_scale)
        self.probe_interval = int(probe_interval)
        self.progress_scale = float(progress_scale)
        self.queue_weight = float(queue_weight)
        self.message_queue = 0.0
        self.environment_queue = 0.0
        self.histories: dict[int, list[float]] = {q: [] for q in self.actions}
        self.last_selected: dict[int, int] = {q: -10**9 for q in self.actions}
        self.step = 0
        self._pending: OnlineDecision | None = None

        min_messages = min(cost.messages for cost in self.costs.values())
        min_environment = min(
            cost.environment_ticks for cost in self.costs.values()
        )
        if (
            min_messages * decisions > total_messages
            or min_environment * decisions > total_environment_ticks
        ):
            raise ValueError("budgets cannot support the cheapest action horizon")

    def _reserve_feasible_actions(self) -> tuple[int, ...]:
        if self.decisions_remaining <= 0:
            return ()
        future = self.decisions_remaining - 1
        min_messages = min(c.messages for c in self.costs.values())
        min_environment = min(c.environment_ticks for c in self.costs.values())
        return tuple(
            q
            for q, cost in self.costs.items()
            if cost.messages + future * min_messages <= self.message_remaining
            and cost.environment_ticks + future * min_environment
            <= self.environment_remaining
        )

    def _price(self, action: int) -> float:
        cost = self.costs[action]
        return (
            self.message_queue * cost.messages / self.message_allowance
            + self.environment_queue
            * cost.environment_ticks
            / self.environment_allowance
        ) / self.queue_weight

    def scores(self) -> dict[int, float]:
        if self._pending is not None:
            raise RuntimeError("observe the pending action before another decision")
        feasible = self._reserve_feasible_actions()
        if not feasible:
            raise RuntimeError("no reserve-feasible action remains")
        scores: dict[int, float] = {}
        total_observations = sum(len(v) for v in self.histories.values())
        for q in feasible:
            history = self.histories[q]
            if not history:
                scores[q] = float("inf")
                continue
            sample = np.asarray(history[-self.window :], dtype=float)
            bonus = self.bonus_scale * np.sqrt(
                np.log(max(total_observations, 2)) / sample.size
            )
            scores[q] = float(np.mean(sample) + bonus - self._price(q))
        return scores

    def decide(self) -> OnlineDecision:
        scores = self.scores()
        feasible = tuple(scores)
        untried = tuple(q for q in feasible if not self.histories[q])
        if untried:
            action = min(untried)
        elif self.step > 0 and self.step % self.probe_interval == 0:
            action = min(feasible, key=lambda q: (self.last_selected[q], q))
        else:
            action = min(feasible, key=lambda q: (-scores[q], q))
        probabilities = {q: float(q == action) for q in feasible}
        decision = OnlineDecision(
            action=action,
            probabilities=probabilities,
            feasible_actions=feasible,
            message_remaining_before=self.message_remaining,
            environment_remaining_before=self.environment_remaining,
        )
        self._pending = decision
        return decision

    def observe(self, *, risk_before: float, risk_after: float) -> OnlineObservation:
        if self._pending is None:
            raise RuntimeError("decide must precede observe")
        if not (0.0 <= risk_before <= 1.0 and 0.0 <= risk_after <= 1.0):
            raise ValueError("validation risks must lie in [0,1]")
        action = self._pending.action
        progress = float(risk_before - risk_after)
        centered_gain = float(np.clip(progress / self.progress_scale, -1.0, 1.0))
        bounded_reward = 0.5 * (centered_gain + 1.0)
        history = self.histories[action]
        history.append(centered_gain)
        if len(history) > self.window:
            del history[:-self.window]
        self.last_selected[action] = self.step

        cost = self.costs[action]
        self.message_remaining -= cost.messages
        self.environment_remaining -= cost.environment_ticks
        self.decisions_remaining -= 1
        self.message_queue = max(
            0.0, self.message_queue + cost.messages - self.message_allowance
        )
        self.environment_queue = max(
            0.0,
            self.environment_queue
            + cost.environment_ticks
            - self.environment_allowance,
        )
        if self.message_remaining < 0 or self.environment_remaining < 0:
            raise AssertionError("reserve feasibility failed to protect a budget")
        observation = OnlineObservation(
            action=action,
            progress=progress,
            bounded_reward=bounded_reward,
            message_remaining_after=self.message_remaining,
            environment_remaining_after=self.environment_remaining,
            message_queue=self.message_queue,
            environment_queue=self.environment_queue,
        )
        self.step += 1
        self._pending = None
        return observation

