import typing as t

import torch

from ppo.trainers.ppo_trainer import PPOTrainer
from ppo.utils.network import Network


class BellmanEquationBasedPPOTrainer(PPOTrainer):
    def _calc_advantage_values(
        self,
        value_network: Network,
        trajectory_data: t.Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        states, next_states, rewards, dones = (
            trajectory_data["states"],
            trajectory_data["next_states"],
            trajectory_data["rewards"],
            trajectory_data["dones"]
        )
        # states, next_states = [batch_size, state_dim]
        # rewards, dones = [batch_size]
        next_states_v_values = value_network(next_states).squeeze(1).detach()
        # next_states_v_values = [batch_size]
        left_equation_part = rewards + self.gamma * (1 - dones) * next_states_v_values
        # left_equation_part = [batch_size]
        right_equation_part = value_network(states).squeeze(1)
        # right_equation_part = [batch_size]
        advantage_values = left_equation_part - right_equation_part
        return advantage_values
