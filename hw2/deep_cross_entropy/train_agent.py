import json
import os
import typing as t
from logging import getLogger

import gym
import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam

from .cross_entropy_agents import DeepCrossEntropyAgent
from .utils import get_trajectories, filter_elite_trajectories

logger = getLogger(__name__)


def train_agent(
    agent: DeepCrossEntropyAgent,
    env_name: str,
    loss_fn: nn.Module,
    n_iterations: int,
    n_trajectories: int,
    q_param: float,
    lr: float,
    max_length: int,
    calc_noise_epsilon: t.Callable[[float, int], float],
    initial_noise_epsilon: float = 1.0,
    solved_problem_reward: t.Optional[float] = None,
    is_parallel: bool = False,
    history_file_path: t.Optional[str] = None,
) -> t.List[float]:
    envs = gym.vector.make(env_name, num_envs=n_trajectories, asynchronous=is_parallel)
    optimizer = Adam(agent.network.parameters(), lr)
    device = agent.get_device()
    mean_total_rewards = []
    history = []

    for i_iteration in range(n_iterations):
        noise_epsilon = calc_noise_epsilon(initial_noise_epsilon, i_iteration + 1)
        trajectories = get_trajectories(agent, envs, max_length, noise_epsilon)
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        mean_total_rewards.append(np.mean(total_rewards))

        logger.info(f"Iteration №{i_iteration + 1}")
        logger.info(
            f"Min total reward: {min(total_rewards):.2f}; "
            f"Max total reward: {max(total_rewards):.2f}"
        )
        logger.info(f"Mean total reward: {mean_total_rewards[-1]:.2f}")

        elite_trajectories = filter_elite_trajectories(trajectories, q_param)
        total_rewards = [
            np.sum(trajectory["rewards"]) for trajectory in elite_trajectories
        ]
        logger.info(f"Min elite total reward: {min(total_rewards):.2f}")

        model_outputs = []
        actions = []
        for trajectory in elite_trajectories:
            model_outputs.extend(trajectory["raw_outputs"])
            actions.extend(trajectory["actions"])

        model_outputs = torch.vstack(model_outputs)
        actions = torch.from_numpy(np.array(actions)).to(device)
        loss = loss_fn(model_outputs, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.extend(
            [
                {
                    key: trajectory[key] for key in ["states", "actions", "rewards"]
                }
                for trajectory in trajectories
            ]
        )

        if (
            solved_problem_reward is not None
            and mean_total_rewards[-1] >= solved_problem_reward
        ):
            break

    if history_file_path is not None:
        dir_name = os.path.dirname(history_file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(history_file_path, "w", encoding="utf-8") as history_file:
            for trajectory in history:
                history_file.write(json.dumps(trajectory) + "\n")

    return mean_total_rewards
