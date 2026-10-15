"""Analytic feasibility audit for delayed launch promises, without trajectories.

The confidence radius concerns bounded, unbiased episode-level *team-return*
gradients, not PPO losses. It is a sufficient distribution-free radius. A
failed radius does not rule out a sharper model- or variance-dependent one.
"""

import argparse
from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def bounded_episode_radius(*, episode_bound, episodes, certificates, delta):
    if not math.isfinite(episode_bound) or episode_bound <= 0:
        raise ValueError("positive finite episode-gradient bound required")
    if min(episodes, certificates) < 1 or not 0 < delta < 1:
        raise ValueError("positive counts and a probability in (0,1) required")
    return episode_bound * (1 + math.sqrt(2 * math.log(certificates / delta))) / math.sqrt(episodes)


@dataclass(frozen=True)
class LaunchPromise:
    agent: int
    step: float
    promised_gain: float
    lower_gradient_norm: float
    upper_gradient_norm: float
    source_radius: float


def launch_promise(*, agent, estimate_norm, radius, source_motion,
                   gradient_bound, smoothness, receipt_motion_bound, step_cap):
    values = (estimate_norm, radius, source_motion, gradient_bound,
              smoothness, receipt_motion_bound, step_cap)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("finite nonnegative certificate quantities required")
    if gradient_bound <= 0 or step_cap <= 0 or estimate_norm > gradient_bound:
        raise ValueError("positive caps and bounded episode-mean gradient required")
    total_radius = radius + smoothness * source_motion
    low = max(0.0, estimate_norm - total_radius)
    high = min(gradient_bound, estimate_norm + total_radius)
    coefficient = low * low - smoothness * receipt_motion_bound * high
    curvature = smoothness * gradient_bound * gradient_bound
    if coefficient <= 0:
        step = 0.0
    elif curvature == 0:
        step = step_cap
    else:
        step = min(step_cap, coefficient / curvature)
    gain = coefficient * step - .5 * curvature * step * step
    return LaunchPromise(agent, step, max(0.0, gain), low, high, total_radius)


def select_launches(promises, *, costs, resource_queues, resource_caps, value_weight=1.0,
                    pending_agents=()):
    """Exact small-team launch objective; charges are at launch, never receipt."""
    if len({p.agent for p in promises}) != len(promises):
        raise ValueError("one candidate promise per actor required")
    charges = np.asarray(costs, dtype=float)
    queues, caps = np.asarray(resource_queues, dtype=float), np.asarray(resource_caps, dtype=float)
    if charges.shape != (len(promises), len(queues)) or caps.shape != queues.shape:
        raise ValueError("one resource-cost row per promise required")
    if not all(np.isfinite(x).all() and (x >= 0).all() for x in (charges, queues, caps)):
        raise ValueError("nonnegative finite resource inputs required")
    if not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError("nonnegative value weight required")
    eligible = [i for i, p in enumerate(promises) if p.agent not in pending_agents and p.step > 0]
    candidates = []
    for mask in itertools.product((False, True), repeat=len(eligible)):
        indices = [i for i, selected in zip(eligible, mask) if selected]
        cost = charges[indices].sum(axis=0) if indices else np.zeros_like(queues)
        if np.any(cost > caps):
            continue
        gain = sum(promises[i].promised_gain for i in indices)
        score = value_weight * gain - float(queues @ cost)
        candidates.append((score, tuple(indices), cost, gain))
    best = min(candidates, key=lambda x: (-x[0], len(x[1]), tuple(promises[i].agent for i in x[1])))
    return {"agents": [promises[i].agent for i in best[1]],
            "steps": [promises[i].step for i in best[1]], "launch_drift_score": float(best[0]),
            "launch_resource_charge": best[2].tolist(), "promised_gain": float(best[3])}


def audit():
    delta, certificates, bound = .05, 72, 1.0
    rows = []
    for episodes in (16, 24, 32, 64, 256, 1024, 4096):
        radius = bounded_episode_radius(episode_bound=bound, episodes=episodes,
                                       certificates=certificates, delta=delta)
        for norm in (1.0, .25, .10):
            # Zero source and receipt motion is the most favorable delay case.
            promise = launch_promise(agent=0, estimate_norm=norm, radius=radius,
                                     source_motion=0, gradient_bound=bound, smoothness=1.0,
                                     receipt_motion_bound=0, step_cap=1.0)
            rows.append({"episodes": episodes, "estimate_norm_over_bound": norm,
                         "radius_over_bound": radius, "promise": asdict(promise)})
    constant = (1 + math.sqrt(2 * math.log(certificates / delta))) ** 2
    necessary_counts = {str(norm): math.floor(constant / (norm * norm)) + 1 for norm in (1.0, .25, .1)}
    current = [r for r in rows if r["episodes"] == 16]
    qualified = any(r["promise"]["promised_gain"] > 0 for r in current)
    return {"audit_kind": "analytic_distribution_free_certificate_feasibility",
            "scientific_trajectories": 0, "role": "Sufficient-bound preflight, not an RL performance experiment",
            "delta": delta, "simultaneous_certificates": certificates, "episode_gradient_bound": bound,
            "current_episodes": 16, "current_radius_over_bound": current[0]["radius_over_bound"],
            "current_certificate_nonvacuous": qualified,
            "necessary_episode_counts_for_positive_norm_lower_bound": necessary_counts,
            "minimum_counts_are_sufficient_for_performance": False,
            "rows": rows, "decision": "NO_GO_CURRENT_CERTIFICATE" if not qualified else "CONDITIONAL_INTERFACE_ONLY",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "interpretation": "At 16 episodes the radius exceeds the largest possible empirical gradient norm, so this certificate idles for every action, even with zero staleness. This does not establish a lower bound against variance-dependent estimators or prove asynchronous MARL impossible."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh audit output required")
    report = audit()
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("decision", "current_radius_over_bound",
                     "necessary_episode_counts_for_positive_norm_lower_bound", "scientific_trajectories")}, indent=2))
