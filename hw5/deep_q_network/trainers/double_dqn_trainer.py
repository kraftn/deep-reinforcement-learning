import torch
from torch import nn

from deep_q_network.trainers.target_network_dqn_trainers import (
    LinearTargetNetworkTrainer
)
from deep_q_network.agents.dqn_agent import DQNAgent


class DoubleDQNTrainer(LinearTargetNetworkTrainer):
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
        agent_q_values = agent.get_q_values(next_states, mode="eval")
        # next_state_q_values = [batch_size, action_dim]

        argmax_actions = target_q_values.argmax(dim=1)
        agent_q_values = agent_q_values[
            torch.arange(agent_q_values.size(0)), argmax_actions
        ]

        target = rewards + (1 - dones) * self.gamma * agent_q_values
        return target.detach()
