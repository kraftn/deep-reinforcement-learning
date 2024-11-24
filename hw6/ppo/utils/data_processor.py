from collections import defaultdict
import typing as t
from logging import getLogger

import torch
from gym import Env
import numpy as np
from torch.utils.data import TensorDataset
from tqdm.auto import trange

from ppo.agents.abstract_agent import AbstractAgent

logger = getLogger(__name__)


class DataProcessor:
    data_keys_order = (
        "states", "actions", "rewards", "next_states", "dones", "returns"
    )

    def __init__(self, agent: AbstractAgent, env: Env, gamma: float):
        self.agent = agent
        self.env = env
        self.gamma = gamma

    def sample_trajectory(self, max_length: int) -> t.Dict[str, list]:
        state = self.env.reset()
        data = defaultdict(list)
        for i_step in range(max_length):
            states = torch.from_numpy(state).unsqueeze(0).to(self.agent.device)
            action = self.agent.sample_actions(states).squeeze(0).cpu().numpy()
            next_state, reward, done, _ = self.env.step(action)
            data["states"].append(state)
            data["actions"].append(action)
            data["rewards"].append(reward)
            data["next_states"].append(next_state)
            data["dones"].append(float(done))
            state = next_state
            if done:
                break
        data["returns"] = self._calc_returns(np.array(data["rewards"])).tolist()
        return dict(data)

    def prepare_dataset(
        self,
        n_trajectories: int,
        max_trajectory_length: int,
    ) -> TensorDataset:
        data = defaultdict(list)
        total_rewards = []
        for i_trajectory in trange(n_trajectories, desc="Sampling trajectories"):
            trajectory = self.sample_trajectory(max_trajectory_length)
            for key, values in trajectory.items():
                data[key].extend(np.expand_dims(value, axis=0) for value in values)
            total_rewards.append(sum(trajectory["rewards"]))

        logger.info(f"Mean total reward: {np.mean(total_rewards):.2f}")

        return TensorDataset(
            *[
                torch.from_numpy(np.concatenate(data[key], axis=0))
                for key in self.data_keys_order
            ]
        )

    def _calc_returns(self, rewards: np.ndarray) -> np.ndarray:
        n_rewards = rewards.shape[0]
        row_indices, column_indices = np.indices((n_rewards, n_rewards))
        coefs = np.triu(np.power(self.gamma, column_indices - row_indices))
        return coefs.dot(rewards)
