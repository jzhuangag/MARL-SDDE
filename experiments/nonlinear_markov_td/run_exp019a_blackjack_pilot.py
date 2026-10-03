"""Run the preregistered EXP-019A exact-Blackjack CPU learning pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.nonlinear_markov_td.exp019a_blackjack_config import (
    CONFIG,
    config_sha256,
)
from experiments.nonlinear_markov_td.t029_blackjack_static_scan import (
    add_card,
    card_probabilities,
    continuing_transition_matrix,
    geometric_mean,
    hit_probability,
    reset_distribution,
    variance_factor,
)


def message_cost(q: int) -> int:
    return int(CONFIG["server_overhead_bytes"]) + int(
        CONFIG["bytes_per_parameter"]
    ) * int(CONFIG["wire_parameter_count"]) * q


def usable_updates(
    q: int, target_horizon: int, budget_ray: str, delay_fraction: float
) -> tuple[int, int, int, int]:
    stride = int(CONFIG["thinning_stride"])
    if budget_ray == "message":
        message_budget = target_horizon * message_cost(4)
        environment_budget = 2 * target_horizon * stride
    elif budget_ray == "environment":
        message_budget = target_horizon * message_cost(32)
        environment_budget = target_horizon * stride
    else:
        raise ValueError(budget_ray)
    delay = int(round(delay_fraction * target_horizon))
    updates = max(
        1,
        min(message_budget // message_cost(q), environment_budget // stride)
        - delay,
    )
    return updates, delay, message_budget, environment_budget


def registered_cells() -> list[dict[str, object]]:
    arms: list[dict[str, object]] = []
    for horizon in CONFIG["target_horizons"]:
        for ray in CONFIG["budget_rays"]:
            for delay_fraction in CONFIG["delay_fractions"]:
                for rho in CONFIG["rho_values"]:
                    for q in CONFIG["q_values"]:
                        updates, delay, msg, env = usable_updates(
                            int(q), int(horizon), str(ray), float(delay_fraction)
                        )
                        arms.append(
                            {
                                "target_horizon": int(horizon),
                                "budget_ray": str(ray),
                                "delay_fraction": float(delay_fraction),
                                "delay": int(delay),
                                "rho": float(rho),
                                "q": int(q),
                                "updates": int(updates),
                                "message_budget": int(msg),
                                "environment_budget": int(env),
                                "risk_proxy": variance_factor(int(q), float(rho))
                                / updates,
                            }
                        )
    fallbacks: dict[tuple[int, str], int] = {}
    for horizon in CONFIG["target_horizons"]:
        for ray in CONFIG["budget_rays"]:
            by_q = {int(q): [] for q in CONFIG["q_values"]}
            for arm in arms:
                if arm["target_horizon"] == horizon and arm["budget_ray"] == ray:
                    by_q[int(arm["q"])].append(float(arm["risk_proxy"]))
            fallbacks[(int(horizon), str(ray))] = min(
                by_q, key=lambda q: (geometric_mean(by_q[q]), q)
            )
    cells = []
    for horizon in CONFIG["target_horizons"]:
        for ray in CONFIG["budget_rays"]:
            fallback_q = fallbacks[(int(horizon), str(ray))]
            for delay_fraction in CONFIG["delay_fractions"]:
                for rho in CONFIG["rho_values"]:
                    group = [
                        arm
                        for arm in arms
                        if arm["target_horizon"] == horizon
                        and arm["budget_ray"] == ray
                        and arm["delay_fraction"] == delay_fraction
                        and arm["rho"] == rho
                    ]
                    selected = min(group, key=lambda row: (row["risk_proxy"], row["q"]))
                    fallback = next(row for row in group if row["q"] == fallback_q)
                    cells.append(
                        {
                            "target_horizon": int(horizon),
                            "budget_ray": str(ray),
                            "active": ray in CONFIG["active_budget_rays"],
                            "delay_fraction": float(delay_fraction),
                            "delay": int(selected["delay"]),
                            "rho": float(rho),
                            "selected_q": int(selected["q"]),
                            "fallback_q": int(fallback_q),
                            "selected_updates": int(selected["updates"]),
                            "fallback_updates": int(fallback["updates"]),
                            "message_budget": int(selected["message_budget"]),
                            "environment_budget": int(selected["environment_budget"]),
                        }
                    )
    return cells


def dealer_final_distributions() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    card_probs = card_probabilities()
    cache: dict[tuple[int, bool], dict[int, float]] = {}

    def finish(total: int, usable: bool) -> dict[int, float]:
        key = (total, usable)
        if key in cache:
            return cache[key]
        if total >= 17:
            result = {0 if total > 21 else total: 1.0}
        else:
            result: dict[int, float] = {}
            for card, probability in card_probs.items():
                next_total, next_usable = add_card(total, usable, card)
                for score, mass in finish(next_total, next_usable).items():
                    result[score] = result.get(score, 0.0) + probability * mass
        cache[key] = result
        return result

    distributions = {}
    for upcard in card_probs:
        combined: dict[int, float] = {}
        for hidden, probability in card_probs.items():
            raw = upcard + hidden
            usable = (upcard == 1 or hidden == 1) and raw + 10 <= 21
            total = raw + (10 if usable else 0)
            for score, mass in finish(total, usable).items():
                combined[score] = combined.get(score, 0.0) + probability * mass
        scores = np.asarray(sorted(combined), dtype=np.int16)
        cdf = np.cumsum([combined[int(score)] for score in scores])
        cdf[-1] = 1.0
        distributions[int(upcard)] = (scores, cdf)
    return distributions


def expected_reward_vector(states: list[tuple[int, int, bool]]) -> np.ndarray:
    card_probs = card_probabilities()
    dealer = dealer_final_distributions()
    rewards = np.zeros(len(states), dtype=np.float64)
    for index, state in enumerate(states):
        p_hit = hit_probability(state)
        hit_reward = 0.0
        for card, probability in card_probs.items():
            next_total, _ = add_card(state[0], state[2], card)
            if next_total > 21:
                hit_reward -= probability
        scores, cdf = dealer[state[1]]
        probs = np.diff(np.concatenate([[0.0], cdf]))
        stick_reward = float(
            sum(
                probability * np.sign(state[0] - int(score))
                for score, probability in zip(scores, probs)
            )
        )
        rewards[index] = p_hit * hit_reward + (1.0 - p_hit) * stick_reward
    return rewards


def exact_value() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, bool]], np.ndarray]:
    transition, states, reset = continuing_transition_matrix()
    rewards = expected_reward_vector(states)
    gamma = float(CONFIG["gamma"])
    value = np.linalg.solve(np.eye(len(states)) - gamma * transition, rewards)
    system = np.vstack([transition.T - np.eye(len(states)), np.ones((1, len(states)))])
    target = np.concatenate([np.zeros(len(states)), np.ones(1)])
    stationary, *_ = np.linalg.lstsq(system, target, rcond=None)
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()
    return value, stationary, states, reset


def _shared_uniforms(
    rng: np.random.Generator, common_flags: np.ndarray
) -> np.ndarray:
    private = rng.random(common_flags.shape)
    common = rng.random((common_flags.shape[0], 1))
    return np.where(common_flags, common, private)


def _sample_from_cdf(cdf: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    return np.searchsorted(cdf, uniforms, side="right")


def generate_tapes(
    rho: float,
    seeds: list[int],
    length: int,
    states: list[tuple[int, int, bool]],
    reset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_seeds = len(seeds)
    q_max = max(CONFIG["q_values"])
    stride = int(CONFIG["thinning_stride"])
    rngs = [np.random.default_rng(seed + int(round(1000 * rho))) for seed in seeds]
    flags = np.zeros((n_seeds, q_max), dtype=bool)
    state = np.zeros((n_seeds, q_max), dtype=np.int16)
    reset_cdf = np.cumsum(reset)
    reset_cdf[-1] = 1.0
    for seed_index, rng in enumerate(rngs):
        flags[seed_index] = rng.random(q_max) < math.sqrt(rho)
        state[seed_index] = _sample_from_cdf(
            reset_cdf, _shared_uniforms(rng, flags[seed_index : seed_index + 1])[0]
        )

    state_info = np.asarray(states, dtype=np.int16)
    lookup = np.full((32, 11, 2), -1, dtype=np.int16)
    for index, (total, dealer, usable) in enumerate(states):
        lookup[total, dealer, int(usable)] = index
    card_values = np.asarray(sorted(card_probabilities()), dtype=np.int16)
    card_cdf = np.cumsum([card_probabilities()[int(card)] for card in card_values])
    card_cdf[-1] = 1.0
    dealer_distributions = dealer_final_distributions()
    source = np.empty((n_seeds, q_max, length), dtype=np.int16)
    reward = np.empty((n_seeds, q_max, length), dtype=np.int8)
    target = np.empty((n_seeds, q_max, length), dtype=np.int16)

    for time in range(length):
        for substep in range(stride):
            current = state_info[state]
            p_hit = np.where(current[:, :, 0] < 20, 0.9, 0.1)
            u_action = np.empty((n_seeds, q_max))
            u_card = np.empty_like(u_action)
            u_dealer = np.empty_like(u_action)
            u_reset = np.empty_like(u_action)
            for seed_index, rng in enumerate(rngs):
                flag = flags[seed_index : seed_index + 1]
                u_action[seed_index] = _shared_uniforms(rng, flag)[0]
                u_card[seed_index] = _shared_uniforms(rng, flag)[0]
                u_dealer[seed_index] = _shared_uniforms(rng, flag)[0]
                u_reset[seed_index] = _shared_uniforms(rng, flag)[0]
            hit = u_action < p_hit
            cards = card_values[_sample_from_cdf(card_cdf, u_card)]
            total = current[:, :, 0].astype(np.int16)
            dealer_up = current[:, :, 1].astype(np.int16)
            usable = current[:, :, 2].astype(bool)
            next_total = total + cards
            next_usable = usable.copy()
            drop_ace = usable & (next_total > 21)
            next_total = np.where(drop_ace, next_total - 10, next_total)
            next_usable = np.where(drop_ace, False, next_usable)
            promote_ace = (~usable) & (cards == 1) & (total + 11 <= 21)
            next_total = np.where(promote_ace, total + 11, next_total)
            next_usable = np.where(promote_ace, True, next_usable)
            bust = hit & (next_total > 21)
            terminal = (~hit) | bust
            rewards = np.zeros((n_seeds, q_max), dtype=np.int8)
            rewards[bust] = -1
            stick = ~hit
            for upcard, (scores, cdf) in dealer_distributions.items():
                mask = stick & (dealer_up == upcard)
                if np.any(mask):
                    sampled = scores[_sample_from_cdf(cdf, u_dealer[mask])]
                    rewards[mask] = np.sign(total[mask] - sampled).astype(np.int8)
            next_state = lookup[next_total.clip(0, 31), dealer_up, next_usable.astype(int)]
            reset_state = _sample_from_cdf(reset_cdf, u_reset).astype(np.int16)
            next_state = np.where(terminal, reset_state, next_state).astype(np.int16)
            if substep == stride - 1:
                source[:, :, time] = state
                reward[:, :, time] = rewards
                target[:, :, time] = next_state
            state = next_state
    return source, reward, target


def train_arm(
    source: np.ndarray,
    rewards: np.ndarray,
    target: np.ndarray,
    q: int,
    updates: int,
    delay: int,
    exact: np.ndarray,
    stationary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_seeds = source.shape[0]
    dimension = len(exact)
    weights = np.zeros((n_seeds, dimension), dtype=np.float64)
    ring = np.zeros((delay + 1, n_seeds, dimension), dtype=np.float64)
    checkpoints = np.unique(
        np.linspace(0, updates, int(CONFIG["evaluation_checkpoints"]), dtype=int)
    )
    checkpoint_set = set(int(value) for value in checkpoints)
    initial = float(np.sum(stationary * exact * exact))
    errors = []
    if 0 in checkpoint_set:
        errors.append(np.ones(n_seeds, dtype=np.float64))
    seed_index = np.arange(n_seeds)[:, None]
    alpha = float(CONFIG["step_size"])
    bound = float(CONFIG["coordinate_projection"])
    for time in range(updates):
        stale = ring[max(0, time - delay) % (delay + 1)]
        state_batch = source[:, :q, time]
        target_batch = target[:, :q, time]
        reward_batch = rewards[:, :q, time]
        value = stale[seed_index, state_batch]
        next_value = stale[seed_index, target_batch]
        delta = reward_batch + float(CONFIG["gamma"]) * next_value - value
        gradient = np.zeros_like(weights)
        np.add.at(
            gradient,
            (np.repeat(np.arange(n_seeds), q), state_batch.reshape(-1)),
            (delta / q).reshape(-1),
        )
        weights = np.clip(weights + alpha * gradient, -bound, bound)
        ring[(time + 1) % (delay + 1)] = weights
        if time + 1 in checkpoint_set:
            error = np.sum(stationary * (weights - exact) ** 2, axis=1) / initial
            errors.append(error)
    trajectory = np.stack(errors, axis=1)
    return trajectory.mean(axis=1), trajectory[:, -1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output_dir: Path) -> dict[str, object]:
    value, stationary, states, reset = exact_value()
    if len(states) != int(CONFIG["parameter_count"]):
        raise RuntimeError("registered parameter dimension drift")
    cells = registered_cells()
    seeds = [int(seed) for seed in CONFIG["pilot_seeds"]]
    maximum_length = max(
        max(int(cell["selected_updates"]), int(cell["fallback_updates"]))
        for cell in cells
    )
    rows: list[dict[str, object]] = []
    for rho in CONFIG["rho_values"]:
        source, rewards, target = generate_tapes(
            float(rho), seeds, maximum_length, states, reset
        )
        for cell_id, cell in enumerate(cells):
            if float(cell["rho"]) != float(rho):
                continue
            for policy in ("selected", "fallback"):
                q = int(cell[f"{policy}_q"])
                updates = int(cell[f"{policy}_updates"])
                auc, terminal = train_arm(
                    source,
                    rewards,
                    target,
                    q,
                    updates,
                    int(cell["delay"]),
                    value,
                    stationary,
                )
                for seed, auc_value, terminal_value in zip(seeds, auc, terminal):
                    rows.append(
                        {
                            "cell_id": cell_id,
                            "seed": seed,
                            "policy": policy,
                            **cell,
                            "q": q,
                            "updates": updates,
                            "normalized_msve_auc": float(auc_value),
                            "terminal_normalized_msve": float(terminal_value),
                            "finite": bool(
                                np.isfinite(auc_value) and np.isfinite(terminal_value)
                            ),
                        }
                    )
    frame = pd.DataFrame(rows).sort_values(["cell_id", "seed", "policy"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    frame.to_csv(metrics_path, index=False, lineterminator="\n", float_format="%.17g")

    paired = frame.pivot_table(
        index=["cell_id", "seed"],
        columns="policy",
        values=["normalized_msve_auc", "terminal_normalized_msve"],
        aggfunc="first",
    )
    auc_ratios = (
        paired[("normalized_msve_auc", "selected")]
        / paired[("normalized_msve_auc", "fallback")]
    )
    terminal_ratios = (
        paired[("terminal_normalized_msve", "selected")]
        / paired[("terminal_normalized_msve", "fallback")]
    )
    cell_meta = frame.drop_duplicates("cell_id").set_index("cell_id")
    cell_auc = auc_ratios.groupby(level="cell_id").apply(
        lambda values: float(math.exp(np.log(values).mean()))
    )
    active_ids = cell_meta.index[cell_meta["active"].astype(bool)]
    inactive_ids = cell_meta.index[~cell_meta["active"].astype(bool)]
    aggregate_auc_ratio = float(math.exp(np.log(auc_ratios).mean()))
    terminal_ratio = float(math.exp(np.log(terminal_ratios).mean()))
    active_strict = int((cell_auc.loc[active_ids] < 1.0).sum())
    inactive_ratio = float(
        math.exp(np.log(auc_ratios.loc[inactive_ids]).mean())
    )
    all_finite = bool(frame["finite"].all())
    budget_valid = bool(
        (
            frame["updates"] * frame["q"].map(message_cost)
            <= frame["message_budget"]
        ).all()
        and (
            frame["updates"] * int(CONFIG["thinning_stride"])
            <= frame["environment_budget"]
        ).all()
    )
    gates = {
        "P1_finite_and_complete": all_finite
        and len(frame) == len(cells) * len(seeds) * 2,
        "P2_budget_valid": budget_valid,
        "P3_aggregate_auc_gain": aggregate_auc_ratio
        <= float(CONFIG["aggregate_auc_ratio_gate"]),
        "P4_active_directional_transfer": active_strict / len(active_ids)
        >= float(CONFIG["active_strict_fraction_gate"]),
        "P5_inactive_no_harm": inactive_ratio
        <= float(CONFIG["inactive_auc_ratio_gate"]),
        "P6_terminal_gain": terminal_ratio <= float(CONFIG["terminal_ratio_gate"]),
        "P7_exact_value_residual": bool(
            np.max(
                np.abs(
                    value
                    - (
                        expected_reward_vector(states)
                        + float(CONFIG["gamma"])
                        * continuing_transition_matrix()[0].dot(value)
                    )
                )
            )
            <= 1e-10
        ),
        "P8_no_gpu_or_external_task": True,
    }
    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "config_sha256": config_sha256(),
        "rows": len(frame),
        "seeds": len(seeds),
        "cells": len(cells),
        "active_cells": len(active_ids),
        "inactive_cells": len(inactive_ids),
        "aggregate_auc_ratio": aggregate_auc_ratio,
        "aggregate_auc_improvement": 1.0 - aggregate_auc_ratio,
        "terminal_ratio": terminal_ratio,
        "active_strict_cells": active_strict,
        "active_strict_fraction": active_strict / len(active_ids),
        "inactive_auc_ratio": inactive_ratio,
        "gates": gates,
        "passed_gates": sum(gates.values()),
        "total_gates": len(gates),
        "formal_authorized": all(gates.values()),
        "gpu_authorized": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "metrics_sha256": sha256(metrics_path),
        "summary_sha256": sha256(summary_path),
        "config_sha256": config_sha256(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "validate":
        cells = registered_cells()
        print(
            json.dumps(
                {
                    "config_sha256": config_sha256(),
                    "cells": len(cells),
                    "seeds": len(CONFIG["pilot_seeds"]),
                    "expected_rows": len(cells) * len(CONFIG["pilot_seeds"]) * 2,
                    "gpu_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required in run mode")
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
