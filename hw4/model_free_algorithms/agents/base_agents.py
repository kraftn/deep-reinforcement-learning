import os
import time
from abc import ABC, abstractmethod
import typing as t
import json

import numpy as np
from tqdm.auto import tqdm

from model_free_algorithms.environments.env_wrapper import EnvWrapper


class AbstractAgent(ABC):
    def __init__(
        self,
        env: EnvWrapper,
        calc_epsilon: t.Callable[[int, int], float] = lambda i, n: 1 - i / n,
        gamma: float = 0.99,
        n_episodes: int = 500,
    ):
        self.env = env
        self.calc_epsilon = calc_epsilon
        self.gamma = gamma
        self.n_episodes = n_episodes

        self.q_values = np.zeros(
            (self.env.num_states, self.env.num_actions), dtype=np.float32
        )

    @abstractmethod
    def train(
        self, max_trajectory_length: int, history_file_path: t.Optional[str] = None
    ):
        raise NotImplementedError

    def run(
        self,
        max_trajectory_length: int,
        epsilon: float,
        visualize: bool = False,
        seed: t.Optional[int] = None,
        sleep_time: float = 0.0,
    ) -> t.Dict[str, list]:
        state = self.env.reset(seed=seed)
        trajectory = {"states": [], "actions": [], "rewards": []}

        for _ in range(max_trajectory_length):
            if visualize:
                self.env.render()
                time.sleep(sleep_time)

            action_dist = self.get_action_dist(state, epsilon)
            action = np.random.choice(self.env.num_actions, p=action_dist)
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            state, reward, done, _ = self.env.step(action)
            trajectory["rewards"].append(reward)
            if done:
                break

        if visualize:
            self.env.render()
            time.sleep(sleep_time)

        return trajectory

    def get_epsilon_greedy_policy(self, epsilon: float) -> np.ndarray:
        policy = np.full_like(
            self.q_values, epsilon / self.env.num_actions, dtype=np.float32
        )
        max_q_actions = self.q_values.argmax(axis=-1)
        policy[list(range(self.env.num_states)), max_q_actions] += 1 - epsilon
        return policy

    def get_action_dist(self, state: int, epsilon: float) -> np.ndarray:
        action_dist = np.full(
            (self.env.num_actions), epsilon / self.env.num_actions, dtype=np.float32
        )
        max_q_action = self.q_values[state].argmax()
        action_dist[max_q_action] += 1 - epsilon
        return action_dist

    def save(self, file_path: str):
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        np.save(file_path, self.q_values)

    def load(self, file_path: str):
        q_values = np.load(file_path)
        if q_values.shape != self.q_values.shape:
            raise RuntimeError(
                f"Mismatched size of q_values. Agent needs {self.q_values.shape} "
                f"shape, but got {q_values.shape} shape."
            )
        self.q_values = q_values
        return self

    @staticmethod
    def _save_history(history: t.List[t.Dict[str, list]], file_path: str):
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as history_file:
            for trajectory in history:
                history_file.write(json.dumps(trajectory) + "\n")


class TemporalDifferenceAgent(AbstractAgent):
    def __init__(
        self,
        env: EnvWrapper,
        calc_epsilon: t.Callable[[int, int], float] = lambda i, n: 1 - i / n,
        gamma: float = 0.99,
        n_episodes: int = 500,
        alpha: float = 0.5,
    ):
        super().__init__(env, calc_epsilon, gamma, n_episodes)
        self.alpha = alpha

    def train(
        self, max_trajectory_length: int, history_file_path: t.Optional[str] = None
    ) -> t.List[t.Dict[str, list]]:
        self.q_values = np.zeros_like(self.q_values, np.float32)
        history = []

        for i_episode in tqdm(
            range(self.n_episodes),
            total=self.n_episodes,
            desc="Training agent",
        ):
            trajectory = self._process_episode(i_episode, max_trajectory_length)
            history.append(trajectory)

        if history_file_path is not None:
            self._save_history(history, history_file_path)
        return history

    @abstractmethod
    def _process_episode(
        self, i_episode: int, max_trajectory_length: int
    ) -> t.Dict[str, list]:
        raise NotImplementedError
