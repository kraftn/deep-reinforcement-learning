from logging import getLogger
import typing as t
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = getLogger(__name__)


class DeepCrossEntropyAgent(ABC):
    def __init__(self, network: nn.Module, state_dim: int):
        self.network = network
        self.state_dim = state_dim

    @abstractmethod
    def get_actions(
        self, states: torch.Tensor, epsilon: float, mode: str
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def run_model(self, states: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "train":
            self.network.train()
            logits = self.network(states)
        elif mode == "eval":
            self.network.eval()
            with torch.no_grad():
                logits = self.network(states)
        else:
            raise NotImplementedError
        return logits

    def get_device(self) -> torch.device:
        return list(self.network.parameters())[0].device

    def save_model(self, file_path: str):
        torch.save(self.network.state_dict(), file_path)

    def load_model(self, file_path: str):
        device = self.get_device()
        self.network.load_state_dict(torch.load(file_path, map_location=device))
        self.network.to(device)


class DiscreteCrossEntropyAgent(DeepCrossEntropyAgent):
    def __init__(
        self,
        network: nn.Module,
        state_dim: int,
        n_actions: int,
        is_deterministic: bool = False
    ):
        super().__init__(network, state_dim)
        self.n_actions = n_actions
        self.is_deterministic = is_deterministic

    def get_actions(
        self, states: torch.Tensor, epsilon: float, mode: str
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        logits = self.run_model(states, mode)
        probs = torch.softmax(logits, dim=1)
        logger.debug(f"Actions probabilities: {probs}")

        uniform_probs = torch.ones(
            (probs.shape[0], self.n_actions), device=self.get_device()
        ) / self.n_actions
        probs = (1 - epsilon) * probs + epsilon * uniform_probs

        if not self.is_deterministic:
            actions = torch.multinomial(probs, 1).squeeze(-1)
        else:
            actions = torch.argmax(probs, dim=1)
        return actions, logits


class ContinuousCrossEntropyAgent(DeepCrossEntropyAgent):
    def __init__(
        self,
        network: nn.Module,
        state_dim: int,
        action_dim: int,
        min_action: torch.Tensor,
        max_action: torch.Tensor
    ):
        super().__init__(network, state_dim)
        self.action_dim = action_dim
        device = self.get_device()
        self.min_action = min_action.to(device)
        self.max_action = max_action.to(device)
        self.delta_action = (max_action - min_action).to(device)

    def get_actions(
        self, states: torch.Tensor, epsilon: float, mode: str
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        actions = self.run_model(states, mode)
        logger.debug(f"Actions: {actions}")
        noise = torch.rand(
            (actions.shape[0], self.action_dim,), device=self.get_device()
        )
        noise = (-self.delta_action + noise * 2 * self.delta_action) * epsilon
        return (
            torch.clamp(actions + noise, min=self.min_action, max=self.max_action),
            actions
        )
