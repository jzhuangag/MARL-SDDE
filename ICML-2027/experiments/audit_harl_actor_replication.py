"""CPU operational audit of exact-buffer replication in pinned MAPPO actors.

No environment, benchmark return, response fitting, or performance selection.
The q-dependent row count is model-processing work, not actor transitions.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import gym
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
PIN = "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"


def parameter_vector(actor):
    return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in actor.actor.parameters()])


def run(out: Path):
    upstream = ROOT / "tmp" / "HARL"
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if head != PIN or dirty:
        raise RuntimeError("exact clean pinned HARL source required")
    if out.exists():
        raise FileExistsError("refusing to overwrite an audit")
    sys.path.insert(0, str(upstream))
    from harl.algorithms.actors.mappo import MAPPO

    config = yaml.safe_load((upstream / "harl/configs/algos_cfgs/mappo.yaml").read_text())
    torch.set_num_threads(1)
    rows = []
    for seed in (63000001, 63000002):
        for activation in ("tanh", "relu"):
            for kind in ("Discrete", "Box"):
                args = {**config["model"], **config["algo"], "activation_func": activation}
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                obs = rng.uniform(-1, 1, (200, 12)).astype(np.float32)
                states = np.zeros((200, 1, 128), dtype=np.float32)
                masks = np.ones((200, 1), dtype=np.float32)
                space = gym.spaces.Box(-1, 1, shape=(12,), dtype=np.float32)
                actions_space = gym.spaces.Discrete(5) if kind == "Discrete" else gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
                original = MAPPO(args, space, actions_space, torch.device("cpu"))
                initial_hash = hashlib.sha256(parameter_vector(original).tobytes()).hexdigest()
                with torch.no_grad():
                    sampled, old_logs, _ = original.get_actions(obs, states, masks)
                    actions = sampled.cpu().numpy()
                    old_logs = old_logs.cpu().numpy()
                    old_distribution = original.evaluate_actions(obs, states, actions, masks, None, masks)[2]
                raw_adv = np.sin(obs[:, :1]) + .2 * obs[:, 1:2] * actions[:, :1]
                advantages = ((raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-5)).astype(np.float32)
                reference = None
                for q in (1, 2, 4, 8):
                    actor = copy.deepcopy(original)
                    sample = tuple(np.tile(value, (q,) + (1,) * (value.ndim - 1))
                                   for value in (obs, states, actions, masks, masks, old_logs, advantages)) + (None,)
                    losses = []
                    for _ in range(5):
                        loss, entropy, gradient_norm, ratios = actor.update(sample)
                        losses.append(float(loss.detach()))
                    parameters = parameter_vector(actor)
                    if reference is None:
                        reference = parameters.copy()
                    with torch.no_grad():
                        distribution = actor.evaluate_actions(obs, states, actions, masks, None, masks)[2]
                        kl = torch.distributions.kl_divergence(old_distribution, distribution)
                        if kl.ndim > 1:
                            kl = kl.sum(dim=-1)
                    rows.append({"seed": seed, "activation": activation, "action_space": kind,
                                 "q": q, "base_rows": 200, "stored_rows": 200*q,
                                 "optimizer_steps": 5, "rows_processed": 1000*q,
                                 "initial_parameters_sha256": initial_hash, "losses": losses,
                                 "policy_kl_on_fixed_buffer": float(kl.mean()),
                                 "max_parameter_difference_from_q1": float(np.max(np.abs(parameters-reference)))})
    maximum = max(row["max_parameter_difference_from_q1"] for row in rows)
    report = {"role": "synthetic-buffer CPU operational audit; not MARL return or performance evidence",
              "harl_commit": head, "torch_version": torch.__version__, "rows": rows,
              "configurations": 8, "payloads": len(rows), "replication_tolerance": 1e-5,
              "maximum_parameter_difference": maximum,
              "all_finite": bool(all(np.isfinite(row["losses"]).all() and np.isfinite(row["policy_kl_on_fixed_buffer"]) for row in rows)),
              "replication_within_tolerance": maximum <= 1e-5,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "Exact duplicated data and matched full-batch optimizer steps do not identify a learning-efficiency benefit. Distinct trajectories, optimizer work, and evolving-policy state must be qualified separately."}
    out.mkdir(parents=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
