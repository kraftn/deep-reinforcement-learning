import argparse
import logging
import os
from logging import getLogger

import gym
import numpy as np

from model_free_algorithms.agents import MonteCarloAgent, SarsaAgent, QLearningAgent
from model_free_algorithms.environments import EnvWrapper

logger = getLogger(__name__)


def calc_additive_epsilon(i_episode: int, n_episodes: int) -> float:
    return 1 - i_episode / n_episodes


def calc_hyperbolic_epsilon(i_episode: int, n_episodes: int) -> float:
    k = (1 - 0.01) / (n_episodes * 0.01)
    return 1 / (1 + k * i_episode)


def calc_exponential_epsilon(i_episode: int, n_episodes: int) -> float:
    a = 0.01 ** (1 / n_episodes)
    return a ** i_episode


def train_agents(
    env_wrapper: EnvWrapper,
    gamma: float,
    n_episodes: int,
    alpha: float,
    max_trajectory_length: int,
    history_files_dir: str,
    models_dir: str,
):
    env_name = env_wrapper.env.spec.id
    agents = [
        MonteCarloAgent(
            env_wrapper, calc_additive_epsilon, gamma, n_episodes
        ),
        SarsaAgent(
            env_wrapper,
            calc_additive_epsilon,
            gamma,
            n_episodes,
            alpha,
        ),
        QLearningAgent(
            env_wrapper,
            calc_additive_epsilon,
            gamma,
            n_episodes,
            alpha,
        )
    ]

    for agent in agents:
        agent_type = agent.__class__.__name__
        logger.info(f"Consider {agent_type} agent")
        history_file_path = os.path.join(
            history_files_dir, f"{env_name}_{agent_type}.jsonl"
        )
        agent.train(max_trajectory_length, history_file_path)
        agent.save(os.path.join(models_dir, f"{env_name}_{agent_type}.npy"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=["discrete_task", "continuous_task", "compare_epsilon"]
    )
    parser.add_argument(
        "--history_files_dir",
        default="histories"
    )
    parser.add_argument(
        "--agents_dir",
        default="agents"
    )
    args = parser.parse_args()

    if args.task == "discrete_task":
        gamma = 0.99
        n_episodes = 35000
        alpha = 0.25
        max_trajectory_length = 10000

        env_name = "Taxi-v3"
        env = gym.make(env_name)
        env_wrapper = EnvWrapper(
            env,
            min_state=np.array([0]),
            max_state=np.array([499]),
            discretization_grid=[500],
        )

        train_agents(
            env_wrapper,
            gamma,
            n_episodes,
            alpha,
            max_trajectory_length,
            args.history_files_dir,
            args.agents_dir,
        )
    elif args.task == "continuous_task":
        gamma = 0.99
        n_episodes = 5000
        alpha = 0.25
        max_trajectory_length = 500

        env_name = "Acrobot-v1"
        env = gym.make(env_name)
        env_wrapper = EnvWrapper(
            env,
            min_state=np.array([-1.0, -1.0, -1.0, -1.0, -12.57, -28.27]),
            max_state=np.array([1.0, 1.0, 1.0, 1.0, 12.57, 28.27]),
            discretization_grid=[3, 3, 3, 3, 30, 60],
        )

        train_agents(
            env_wrapper,
            gamma,
            n_episodes,
            alpha,
            max_trajectory_length,
            args.history_files_dir,
            args.agents_dir,
        )
    elif args.task == "compare_epsilon":
        gamma = 0.99
        n_episodes = 35000
        max_trajectory_length = 10000

        env_name = "Taxi-v3"
        env = gym.make(env_name)
        env_wrapper = EnvWrapper(
            env,
            min_state=np.array([0]),
            max_state=np.array([499]),
            discretization_grid=[500],
        )

        for calc_epsilon in [
            calc_additive_epsilon,
            calc_exponential_epsilon,
            calc_hyperbolic_epsilon,
        ]:
            logger.info(f"Consider {calc_epsilon.__name__} method")
            agent = MonteCarloAgent(env_wrapper, calc_epsilon, gamma, n_episodes)
            history_file_path = os.path.join(
                args.history_files_dir,
                f"{agent.__class__.__name__}_{calc_epsilon.__name__}.jsonl"
            )
            agent.train(max_trajectory_length, history_file_path)
    else:
        raise NotImplementedError
