import argparse
import json
from logging import getLogger
import logging
import time

import gym
import torch

from ppo.utils.network import Network
from ppo.agents import ContinuousDistributionAgent, DiscreteDistributionAgent

logger = getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_file_path")
    parser.add_argument("env_name")
    parser.add_argument("agent_type", choices=["continuous", "discrete"])
    parser.add_argument("--max_trajectory_length", default=1000, type=int)
    parser.add_argument("--env_args", default="{}", type=json.loads)
    args = parser.parse_args()

    env = gym.make(args.env_name, **args.env_args)
    if args.agent_type == "continuous":
        agent = ContinuousDistributionAgent(
            Network(
                [
                    env.observation_space.shape[0],
                    128,
                    128,
                    env.action_space.shape[0] * 2
                ]
            ),
            torch.empty(env.observation_space.shape[0]),
            torch.empty(env.observation_space.shape[0]),
        )
    elif args.agent_type == "discrete":
        agent = DiscreteDistributionAgent(
            Network(
                [env.observation_space.shape[0], 128, 128, env.action_space.n]
            )
        )
    else:
        raise NotImplementedError
    agent.load(args.model_file_path)

    state = env.reset(seed=int(time.time()))
    total_reward = 0.0
    n_steps = 0

    for i_trajectory_step in range(args.max_trajectory_length):
        env.render()
        states = torch.from_numpy(state).unsqueeze(0)
        action = agent.sample_actions(states).squeeze(0).numpy()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        n_steps += 1
        if done:
            break

    env.render()
    logger.info(f"Number of steps: {n_steps}")
    logger.info(f"Total reward: {total_reward:.2f}")

