import typing as t

import numpy as np
from tqdm.auto import tqdm

from model_free_algorithms.agents.base_agents import AbstractAgent


class MonteCarloAgent(AbstractAgent):
    def train(
        self, max_trajectory_length: int, history_file_path: t.Optional[str] = None
    ) -> t.List[t.Dict[str, list]]:
        self.q_values = np.zeros_like(self.q_values, dtype=np.float32)
        counter = np.zeros_like(self.q_values, dtype=np.int32)

        history = []
        for i_episode in tqdm(
            range(self.n_episodes),
            total=self.n_episodes,
            desc="Training agent",
        ):
            epsilon = self.calc_epsilon(i_episode, self.n_episodes)
            trajectory = self.run(max_trajectory_length, epsilon)
            g_values = self._calc_g_values(trajectory)
            self._update_q_values(trajectory, g_values, counter)
            history.append(trajectory)

        if history_file_path is not None:
            self._save_history(history, history_file_path)
        return history

    def _calc_g_values(self, trajectory: t.Dict[str, list]) -> t.List[float]:
        trajectory_length = len(trajectory["states"])
        rewards = trajectory["rewards"]

        g_values = [0] * trajectory_length
        g_values[trajectory_length - 1] = rewards[-1]
        for i_step in range(trajectory_length - 2, -1, -1):
            g_values[i_step] = rewards[i_step] + self.gamma * g_values[i_step + 1]

        return g_values

    def _update_q_values(
        self,
        trajectory: t.Dict[str, list],
        g_values: t.List[float],
        counter: np.ndarray
    ):
        trajectory_length = len(trajectory["states"])
        for i_step in range(trajectory_length):
            state = trajectory["states"][i_step]
            action = trajectory["actions"][i_step]
            g_value = g_values[i_step]
            self.q_values[state, action] += (
                (g_value - self.q_values[state, action]) / (counter[state, action] + 1)
            )
            counter[state, action] += 1
