import time
import typing as t
from logging import getLogger

import gym
import torch
import numpy as np

from .cross_entropy_agents import DeepCrossEntropyAgent

logger = getLogger(__name__)


def get_trajectories(
    agent: DeepCrossEntropyAgent,
    envs: gym.vector.VectorEnv,
    max_length: int,
    noise_epsilon: float,
    mode: str = "train"
) -> t.List[t.Dict[str, list]]:
    device = agent.get_device()
    trajectories = [
        {"states": [], "actions": [], "rewards": [], "raw_outputs": []}
        for _ in range(envs.num_envs)
    ]
    is_trajectory_done = [False] * envs.num_envs
    curr_states = envs.reset()

    for i_step in range(max_length):
        curr_states = torch.from_numpy(curr_states).to(device)
        actions, raw_outputs = agent.get_actions(curr_states, noise_epsilon, mode)
        actions = actions.detach().cpu().numpy()
        next_states, rewards, dones, _ = envs.step(actions)

        for i_trajectory, trajectory in enumerate(trajectories):
            if is_trajectory_done[i_trajectory]:
                continue

            trajectory["states"].append(curr_states[i_trajectory].tolist())
            trajectory["actions"].append(actions[i_trajectory].item())
            trajectory["raw_outputs"].append(raw_outputs[i_trajectory])
            trajectory["rewards"].append(rewards[i_trajectory].item())
            is_trajectory_done[i_trajectory] = dones[i_trajectory]

        curr_states = next_states
        if np.all(is_trajectory_done):
            break

    return trajectories


def filter_elite_trajectories(
    trajectories: t.List[t.Dict[str, list]], q_param: float
) -> t.List[t.Dict[str, list]]:
    total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param)
    elite_trajectories = []
    for trajectory, total_reward in zip(trajectories, total_rewards):
        if total_reward >= quantile:
            elite_trajectories.append(trajectory)
    return elite_trajectories


def visualize_trajectory(
    agent: DeepCrossEntropyAgent, env: gym.Env, max_length: int, noise_epsilon: float
) -> float:
    state = env.reset(seed=int(time.time()))
    env.render()

    total_reward = 0.0
    n_steps = 0
    for _ in range(max_length):
        state = torch.from_numpy(state).unsqueeze(0)
        actions, _ = agent.get_actions(state, noise_epsilon, mode="eval")
        action = actions.detach().cpu().numpy()[0]

        state, reward, done, _ = env.step(action)
        total_reward += reward
        n_steps += 1

        env.render()

        if done:
            break

    logger.info(f"Number of steps: {n_steps}; Total reward: {total_reward:.2f}")
    return total_reward
