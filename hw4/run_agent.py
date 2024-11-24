import argparse
import time
from logging import getLogger
import logging

import gym
import numpy as np

from model_free_algorithms.agents import MonteCarloAgent, SarsaAgent, QLearningAgent
from model_free_algorithms.environments import EnvWrapper

logger = getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", choices=["taxi", "acrobot"])
    parser.add_argument(
        "agent_type", choices=["monte_carlo", "sarsa", "q_learning"]
    )
    parser.add_argument("agent_file_path")
    parser.add_argument("--max_trajectory_length", default=10000, type=int)
    parser.add_argument("--epsilon", default=0.0, type=float)
    args = parser.parse_args()

    if args.env_name == "taxi":
        env_name = "Taxi-v3"
        min_state = np.array([0])
        max_state = np.array([499])
        discretization_grid = [500]
        sleep_time = 0.5
    elif args.env_name == "acrobot":
        env_name = "Acrobot-v1"
        min_state = np.array([-1.0, -1.0, -1.0, -1.0, -12.57, -28.27])
        max_state = np.array([1.0, 1.0, 1.0, 1.0, 12.57, 28.27])
        discretization_grid = [3, 3, 3, 3, 30, 60]
        sleep_time = 0.0
    else:
        raise NotImplementedError

    env_wrapper = EnvWrapper(
        gym.make(env_name),
        min_state=min_state,
        max_state=max_state,
        discretization_grid=discretization_grid,
    )

    if args.agent_type == "monte_carlo":
        agent = MonteCarloAgent(env_wrapper)
    elif args.agent_type == "sarsa":
        agent = SarsaAgent(env_wrapper)
    elif args.agent_type == "q_learning":
        agent = QLearningAgent(env_wrapper)
    else:
        raise NotImplementedError

    agent.load(args.agent_file_path)
    history = agent.run(
        max_trajectory_length=args.max_trajectory_length,
        epsilon=args.epsilon,
        visualize=True,
        seed=int(time.time()),
        sleep_time=sleep_time,
    )
    logger.info(f"Total reward: {sum(history['rewards']):.2f}")
