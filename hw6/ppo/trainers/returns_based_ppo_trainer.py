import typing as t

import torch

from ppo.trainers.ppo_trainer import PPOTrainer
from ppo.utils.network import Network


class ReturnsBasedPPOTrainer(PPOTrainer):
    def _calc_advantage_values(
        self, value_network: Network, trajectory_data: t.Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        states = trajectory_data["states"]
        # states = [batch_size, state_dim]
        v_values = value_network(states).squeeze(1)
        # v_values = [batch_size]
        returns = trajectory_data["returns"]
        # returns = [batch_size]
        advantage_values = returns - v_values
        return advantage_values
