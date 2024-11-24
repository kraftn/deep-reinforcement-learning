import argparse
from logging import getLogger
import logging
import time

import gym
import torch

from sac.agent import SacAgent
from sac.utils import Network

logger = getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_file_path")
    parser.add_argument("--max_trajectory_length", default=1000, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    args = parser.parse_args()

    env = gym.make("LunarLander-v2")
    agent = SacAgent(
        Network(
            [env.observation_space.shape[0], 128, 128, env.action_space.n]
        ),
        temperature=args.temperature,
    )
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

