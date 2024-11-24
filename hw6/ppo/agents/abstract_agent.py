import typing as t
from abc import ABC, abstractmethod
import os
from logging import getLogger

import torch
from torch.distributions import Distribution

from ppo.utils.network import Network

logger = getLogger(__name__)


class AbstractAgent(ABC):
    def __init__(self, network: Network):
        self.network = network

    def sample_actions(self, states: torch.Tensor) -> torch.Tensor:
        predictions = self._get_predictions(states, mode="eval")
        distribution = self._get_distribution(predictions)
        actions = distribution.sample()
        logger.debug(f"Actions: {actions}")
        return actions

    def calc_log_probs(
        self, states: torch.Tensor, actions: torch.Tensor, mode: str = "train"
    ) -> torch.Tensor:
        predictions = self._get_predictions(states, mode)
        distribution = self._get_distribution(predictions)
        log_probs = distribution.log_prob(actions)
        if log_probs.ndim > 2:
            raise RuntimeError("log probs have ndim, that is greater than 2")
        if log_probs.ndim == 2:
            log_probs = log_probs.sum(dim=1)
        return log_probs

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self.network.device

    @abstractmethod
    def _get_distribution(self, predictions: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def _get_predictions(self, states: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "train":
            self.network.train()
            return self.network(states)
        elif mode == "eval":
            self.network.eval()
            with torch.no_grad():
                return self.network(states)
        else:
            raise NotImplementedError

    @staticmethod
    def _save_state_dict(state_dict: t.Dict[str, torch.Tensor], path: str):
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(state_dict, path)
