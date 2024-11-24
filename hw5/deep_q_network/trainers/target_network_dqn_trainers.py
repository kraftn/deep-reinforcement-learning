import torch
from torch import nn

from deep_q_network.agents.dqn_agent import DQNAgent
from deep_q_network.trainers.abstract_dqn_trainer import AbstractTrainer


class HardTargetNetworkTrainer(AbstractTrainer):
    def _calc_target(
        self,
        agent: DQNAgent,
        target_network: nn.Module,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        target_network.eval()
        with torch.no_grad():
            target_q_values = target_network(next_states)
            # target_q_values = [batch_size, action_dim]
        return rewards + (1 - dones) * self.gamma * target_q_values.max(dim=1).values

    def _update_target_network(self, agent: DQNAgent, target_network: nn.Module):
        self._check_networks(agent.network, target_network)
        param_key2agent_param = dict(agent.network.named_parameters())
        param_key2target_param = dict(target_network.named_parameters())
        for param_key, target_param in param_key2target_param.items():
            target_param.data = param_key2agent_param[param_key].data

class LinearTargetNetworkTrainer(AbstractTrainer):
    def __init__(
        self,
        n_episodes: int,
        lr: float,
        batch_size: int,
        max_trajectory_length: int,
        gamma: float,
        epsilon_decay: float,
        target_update_tau: float,
        device: str,
    ):
        super().__init__(
            n_episodes=n_episodes,
            lr=lr,
            batch_size=batch_size,
            max_trajectory_length=max_trajectory_length,
            n_network_update_steps=1,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            device=device
        )
        self.target_update_tau = target_update_tau

    def _update_target_network(self, agent: DQNAgent, target_network: nn.Module):
        self._check_networks(agent.network, target_network)
        param_key2agent_param = dict(agent.network.named_parameters())
        param_key2target_param = dict(target_network.named_parameters())
        for param_key, target_param in param_key2target_param.items():
            target_param.data = (
                self.target_update_tau * param_key2agent_param[param_key].data
                + (1 - self.target_update_tau) * target_param.data
            )


class SoftTargetNetworkTrainer(LinearTargetNetworkTrainer):
    def _calc_target(
        self,
        agent: DQNAgent,
        target_network: nn.Module,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        target_network.eval()
        with torch.no_grad():
            target_q_values = target_network(next_states)
            # target_q_values = [batch_size, action_dim]
        return rewards + (1 - dones) * self.gamma * target_q_values.max(dim=1).values
