import copy
import json
import os
from abc import ABC, abstractmethod
import typing as t
from logging import getLogger
import math

import gym
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from deep_q_network.utils.memory import Memory
from deep_q_network.agents.dqn_agent import DQNAgent

logger = getLogger(__name__)


class AbstractTrainer(ABC):
    def __init__(
        self,
        n_episodes: int,
        lr: float,
        batch_size: int,
        max_trajectory_length: int,
        n_network_update_steps: int,
        gamma: float,
        epsilon_decay: float,
        device: str,
    ):
        self.n_episodes = n_episodes
        self.lr = lr
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length
        self.n_network_update_steps = n_network_update_steps
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.device = device

    def train(
        self,
        agent: DQNAgent,
        env: gym.Env,
        history_file_path: t.Optional[str] = None,
        n_episodes_per_log: int = 1,
    ):
        agent.reset()
        if list(agent.network.buffers()):
            raise RuntimeError("Agent network mustn't have any buffers.")
        target_network = copy.deepcopy(agent.network).eval()
        optimizer = torch.optim.Adam(agent.network.parameters(), lr=self.lr)
        memory = Memory(self.device)
        history = []
        epsilon = 1.0

        for i_episode in range(self.n_episodes):
            trajectory_state = env.reset()
            trajectory = {"rewards": []}
            loss_values = []

            for i_trajectory_step in range(self.max_trajectory_length):
                trajectory_state, trajectory_reward, trajectory_done = (
                    self._take_env_step(agent, env, memory, trajectory_state, epsilon)
                )
                trajectory["rewards"].append(trajectory_reward)
                epsilon *= math.exp(-self.epsilon_decay)
                if len(memory) < self.batch_size:
                    continue
                for i_network_update_step in range(self.n_network_update_steps):
                    batch = memory.get_batch(self.batch_size)
                    loss = self._update_agent_network(
                        agent, target_network, batch, optimizer
                    )
                    loss_values.append(loss)
                self._update_target_network(agent, target_network)
                if trajectory_done:
                    break

            if i_episode % n_episodes_per_log == 0:
                logger.info(f"Episode №{i_episode + 1}")
                if loss_values:
                    logger.info(f"Mean MSE loss: {np.mean(loss_values):.2f}")
                logger.info(f"Total reward: {sum(trajectory['rewards']):.2f}")
                logger.info(f"Epsilon: {epsilon:.2f}")
            history.append(trajectory)
            
        if history_file_path is not None:
            self._save_history(history, history_file_path)
        return history

    def _update_agent_network(
        self,
        agent: DQNAgent,
        target_network: nn.Module,
        batch: t.Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        states, actions, rewards, next_states, dones = batch

        current_state_q_values = agent.get_q_values(states, mode="train")
        # current_state_q_values = [batch_size, action_dim]
        current_state_q_values = current_state_q_values[
            torch.arange(current_state_q_values.size(0)), actions
        ]
        # current_state_q_values = [batch_size]

        target = self._calc_target(
            agent, target_network, next_states, rewards, dones
        )
        # target = [batch_size]

        loss = F.mse_loss(current_state_q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @abstractmethod
    def _calc_target(
        self,
        agent: DQNAgent,
        target_network: nn.Module,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _update_target_network(
        self,
        agent: DQNAgent,
        target_network: nn.Module,
    ):
        raise NotImplementedError

    def _take_env_step(
        self,
        agent: DQNAgent,
        env: gym.Env,
        memory: Memory,
        state: np.ndarray,
        epsilon: float,
    ) -> t.Tuple[np.ndarray, float, bool]:
        states = torch.from_numpy(state).to(self.device).unsqueeze(0)
        action = agent.get_epsilon_greedy_actions(states, epsilon).item()
        next_state, reward, done, _ = env.step(action)
        memory.add_sample(state.tolist(), action, reward, next_state.tolist(), done)
        return next_state, reward, done

    @staticmethod
    def _save_history(history: t.List[dict], history_file_path: str):
        dir_name = os.path.dirname(history_file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(history_file_path, "w", encoding="utf-8") as history_file:
            for trajectory in history:
                history_file.write(json.dumps(trajectory) + "\n")

    @staticmethod
    def _check_networks(
        network_1: nn.Module, network_2: nn.Module,
    ):
        param_keys_1 = {key_param[0] for key_param in network_1.named_parameters()}
        param_keys_2 = {key_param[0] for key_param in network_2.named_parameters()}
        if param_keys_1.symmetric_difference(param_keys_2):
            raise RuntimeError("Networks have different sets of weights.")
