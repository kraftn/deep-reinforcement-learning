import argparse
import logging

import gym
import torch

from deep_cross_entropy.cross_entropy_agents import (
    DiscreteCrossEntropyAgent, ContinuousCrossEntropyAgent
)
from deep_cross_entropy.network import AgentNetwork
from deep_cross_entropy.utils import visualize_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", choices=["lunar_lander", "mountain_car"])
    parser.add_argument("model_file_path")
    parser.add_argument("--deterministic_agent", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.env_name == "lunar_lander":
        state_dim, n_actions = 8, 4
        network = AgentNetwork([state_dim, 192, n_actions])
        agent = DiscreteCrossEntropyAgent(
            network, state_dim, n_actions, is_deterministic=args.deterministic_agent
        )
        agent.load_model(args.model_file_path)
        env = gym.make("LunarLander-v2")
        visualize_trajectory(
            agent, env, max_length=1000, noise_epsilon=0.0
        )
    elif args.env_name == "mountain_car":
        state_dim, action_dim = 2, 1
        network = AgentNetwork([state_dim, 64, action_dim])
        agent = ContinuousCrossEntropyAgent(
            network,
            state_dim,
            action_dim,
            min_action=torch.tensor([-1.0], dtype=torch.float32),
            max_action=torch.tensor([1.0], dtype=torch.float32)
        )
        agent.load_model(args.model_file_path)
        env = gym.make("MountainCarContinuous-v0")
        visualize_trajectory(
            agent, env, max_length=1000, noise_epsilon=0.0
        )
    else:
        raise NotImplementedError
