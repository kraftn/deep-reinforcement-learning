import os
import typing as t

import torch

from deep_q_network.utils.q_network import QNetwork


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: t.Sequence[int],
        device: str,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.network = QNetwork(
            layer_sizes=[state_dim, *hidden_sizes, action_dim]
        ).to(device)

    def get_q_values(self, states: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "train":
            self.network.train()
            q_values = self.network(states)
        elif mode == "eval":
            self.network.eval()
            with torch.no_grad():
                q_values = self.network(states)
        else:
            raise NotImplementedError
        return q_values

    def get_epsilon_greedy_actions(
        self, states: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        q_values = self.get_q_values(states, mode="eval")
        argmax_q_values = q_values.argmax(dim=1)
        action_dist = torch.full_like(
            q_values, epsilon / self.action_dim, device=self.device
        )
        action_dist[torch.arange(action_dist.size(0)), argmax_q_values] += 1 - epsilon
        return torch.multinomial(action_dist, num_samples=1).squeeze(1)

    def reset(self):
        self.network.reset_parameters()

    def save(self, file_path: str):
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.network.state_dict(), file_path)

    def load(self, file_path: str):
        state_dict = torch.load(file_path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)
