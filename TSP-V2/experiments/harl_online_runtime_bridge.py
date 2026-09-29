"""Runtime bridge for deterministic evaluation on the pinned HARL runner.

The stopped development runner accessed a non-existent ``runner.env_name``
attribute.  Pinned HARL stores the environment identifier in
``runner.args["env"]``.  This module centralizes that interface and is shared by
the outcome-free runtime qualification and every later scientific runner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch


def runner_environment_name(runner) -> str:
    """Return the environment identifier from the pinned HARL runner API."""

    args = getattr(runner, "args", None)
    if not isinstance(args, Mapping):
        raise TypeError("pinned HARL runner must expose a mapping-valued args")
    env_name = args.get("env")
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("pinned HARL runner args must contain a nonempty env")
    return env_name


def _to_numpy(value):
    return value.detach().cpu().numpy()


@torch.no_grad()
def deterministic_horizon_return(runner, *, seed: int, horizon: int) -> float:
    """Evaluate decentralized policies on a fresh, explicitly seeded stream."""

    if horizon <= 0:
        raise ValueError("evaluation horizon must be positive")
    from harl.utils.envs_tools import make_eval_env

    env_name = runner_environment_name(runner)
    eval_env = make_eval_env(env_name, seed, 1, runner.env_args)
    try:
        obs, _, available_actions = eval_env.reset()
        rnn_states = np.zeros(
            (1, runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((1, runner.num_agents, 1), dtype=np.float32)
        rewards = []
        for _ in range(horizon):
            action_columns = []
            for agent_id in range(runner.num_agents):
                actions, next_state = runner.actor[agent_id].act(
                    obs[:, agent_id],
                    rnn_states[:, agent_id],
                    masks[:, agent_id],
                    available_actions[:, agent_id]
                    if available_actions is not None
                    else None,
                    deterministic=True,
                )
                rnn_states[:, agent_id] = _to_numpy(next_state)
                action_columns.append(_to_numpy(actions))
            joint_actions = np.asarray(action_columns).transpose(1, 0, 2)
            obs, _, reward, dones, _, available_actions = eval_env.step(
                joint_actions
            )
            mean_reward = float(np.mean(reward))
            if not math.isfinite(mean_reward):
                raise RuntimeError("evaluation produced a non-finite reward")
            rewards.append(mean_reward)
            done_env = np.all(dones, axis=1)
            rnn_states[done_env] = 0.0
            masks[:] = 1.0
            masks[done_env] = 0.0
        result = float(np.mean(rewards))
        if not math.isfinite(result):
            raise RuntimeError("evaluation produced a non-finite return")
        return result
    finally:
        eval_env.close()
