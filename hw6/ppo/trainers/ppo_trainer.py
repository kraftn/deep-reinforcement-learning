import json
import os
import typing as t
from abc import ABC, abstractmethod
from logging import getLogger

import torch
from torch.utils.data import DataLoader, TensorDataset, StackDataset, Dataset
from torch.optim import Adam, Optimizer
from gym import Env
import numpy as np
from tqdm.auto import trange

from ppo.agents.abstract_agent import AbstractAgent
from ppo.utils.data_processor import DataProcessor
from ppo.utils.network import Network

logger = getLogger(__name__)


class PPOTrainer(ABC):
    def __init__(
        self,
        n_episodes: int,
        n_trajectories_per_episode: int,
        n_epochs_per_episode: int,
        epsilon: float,
        policy_network_lr: float,
        value_network_lr: float,
        batch_size: int,
        gamma: float,
        value_network_hidden_sizes: t.Sequence[int],
        max_trajectory_length: int,
        dataloader_num_workers: int = 4,
    ):
        self.n_episodes = n_episodes
        self.n_trajectories_per_episode = n_trajectories_per_episode
        self.n_epochs_per_episode = n_epochs_per_episode
        self.epsilon = epsilon
        self.policy_network_lr = policy_network_lr
        self.value_network_lr = value_network_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.value_network_hidden_sizes = value_network_hidden_sizes
        self.max_trajectory_length = max_trajectory_length
        self.dataloader_num_workers = dataloader_num_workers

    def train(
        self, agent: AbstractAgent, env: Env, history_file_path: t.Optional[str] = None
    ) -> t.Tuple[t.List[float], t.List[float], t.List[t.Dict[str, list]]]:
        agent.network.reset_parameters()
        policy_network_optimizer = Adam(
            agent.network.parameters(), lr=self.policy_network_lr
        )

        value_network = Network(self.value_network_hidden_sizes)
        value_network = value_network.to(agent.device).train()
        value_network_optimizer = Adam(
            value_network.parameters(), lr=self.value_network_lr
        )

        data_processor = DataProcessor(agent, env, self.gamma)
        history = []

        policy_network_losses, value_network_losses = [], []
        for i_episode in range(self.n_episodes):
            logger.info(f"Episode №{i_episode + 1}")
            dataset = self._prepare_train_dataset(agent, data_processor)
            history.extend(
                self._transform_dataset_into_history(dataset.datasets["trajectory"])
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.dataloader_num_workers
            )
            for _ in trange(
                self.n_epochs_per_episode, desc="Iterating over epochs"
            ):
                policy_network_loss, value_network_loss = self._train_one_epoch(
                    agent,
                    value_network,
                    dataloader,
                    policy_network_optimizer,
                    value_network_optimizer
                )
                policy_network_losses.append(policy_network_loss)
                value_network_losses.append(value_network_loss)
            logger.info(
                "Mean policy network loss: %f",
                np.mean(policy_network_losses[-self.n_epochs_per_episode:])
            )
            logger.info(
                "Mean value network loss: %f",
                np.mean(value_network_losses[-self.n_epochs_per_episode:])
            )

        if history_file_path is not None:
            self._save_history(history, history_file_path)

        return policy_network_losses, value_network_losses, history

    def _prepare_train_dataset(
        self, agent: AbstractAgent, data_processor: DataProcessor,
    ) -> StackDataset:
        trajectories_dataset = data_processor.prepare_dataset(
            self.n_trajectories_per_episode, self.max_trajectory_length
        )

        states, actions, _, _, _, _ = trajectories_dataset.tensors
        states, actions = states.to(agent.device), actions.to(agent.device)
        log_probs = agent.calc_log_probs(states, actions, mode="eval").detach().cpu()
        # log_probs = [dataset_len]
        log_probs_dataset = TensorDataset(log_probs)

        return StackDataset(
            trajectory=trajectories_dataset, log_prob=log_probs_dataset
        )

    def _train_one_epoch(
        self,
        agent: AbstractAgent,
        value_network: Network,
        dataloader: DataLoader,
        policy_network_optimizer: Optimizer,
        value_network_optimizer: Optimizer,
    ) -> t.Tuple[float, float]:
        total_policy_network_loss, total_value_network_loss = 0.0, 0.0
        total_n_samples = 0

        for batch in dataloader:
            trajectory_data = dict()
            for i_key, key in enumerate(DataProcessor.data_keys_order):
                trajectory_data[key] = batch["trajectory"][i_key].to(agent.device)
            old_log_probs = batch["log_prob"][0].to(agent.device)
            # old_log_probs = [batch_size]

            new_log_probs = agent.calc_log_probs(
                trajectory_data["states"], trajectory_data["actions"], mode="train"
            )
            # new_log_probs = [batch_size]
            advantage_values = self._calc_advantage_values(
                value_network, trajectory_data
            )
            # advantage_values = [batch_size]

            policy_network_loss = self._calc_policy_network_loss(
                new_log_probs, old_log_probs, advantage_values
            )
            self._perform_backpropagation(policy_network_loss, policy_network_optimizer)

            value_network_loss = torch.pow(advantage_values, 2).mean()
            self._perform_backpropagation(value_network_loss, value_network_optimizer)

            batch_size = trajectory_data["states"].size(0)
            total_policy_network_loss += policy_network_loss.item() * batch_size
            total_value_network_loss += value_network_loss.item() * batch_size
            total_n_samples += batch_size

        return (
            total_policy_network_loss / total_n_samples,
            total_value_network_loss / total_n_samples,
        )

    @abstractmethod
    def _calc_advantage_values(
        self, value_network: Network, trajectory_data: t.Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def _calc_policy_network_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantage_values: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(new_log_probs - old_log_probs)
        truncated_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        return -1 * (
            torch.min(
                ratio * advantage_values.detach(),
                truncated_ratio * advantage_values.detach(),
            )
        ).mean()

    @staticmethod
    def _perform_backpropagation(loss: torch.Tensor, optimizer: Optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @staticmethod
    def _transform_dataset_into_history(
        dataset: TensorDataset
    ) -> t.List[t.Dict[str, list]]:
        rewards_key_idx = DataProcessor.data_keys_order.index("rewards")
        dones_key_idx = DataProcessor.data_keys_order.index("dones")

        history = []
        trajectory = {"rewards": []}
        for i_sample in range(len(dataset)):
            trajectory["rewards"].append(dataset[i_sample][rewards_key_idx].item())
            done = dataset[i_sample][dones_key_idx]
            if done:
                history.append(trajectory)
                trajectory = {"rewards": []}

        if not dataset[-1][dones_key_idx]:
            raise RuntimeError("Dataset doesn't end with an item with done = True")

        return history

    @staticmethod
    def _save_history(history: t.Sequence[t.Dict[str, list]], path: str):
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as history_file:
            for trajectory in history:
                history_file.write(json.dumps(trajectory) + "\n")
