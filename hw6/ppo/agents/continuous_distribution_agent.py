from logging import getLogger

import torch
from torch.distributions import Normal, Distribution

from ppo.agents.abstract_agent import AbstractAgent
from ppo.utils.network import Network

logger = getLogger(__name__)


class ContinuousDistributionAgent(AbstractAgent):
    def __init__(
        self,
        network: Network,
        min_action: torch.Tensor,
        max_action: torch.Tensor,
    ):
        super().__init__(network)
        if min_action.ndim == 1:
            min_action = min_action.unsqueeze(0)
        self.min_action = min_action.to(self.device)
        if max_action.ndim == 1:
            max_action = max_action.unsqueeze(0)
        self.max_action = max_action.to(self.device)

    def _get_distribution(self, predictions: torch.Tensor) -> Distribution:
        loc, scale = torch.tensor_split(predictions, 2, dim=-1)
        delta = self.max_action - self.min_action
        loc = (
            self.min_action + delta * (torch.tanh(loc) + 1) / 2
        )
        scale = delta * (torch.tanh(scale) + 1) / 2
        logger.debug(f"Distribution: loc = {loc}; scale = {scale}")
        return Normal(loc=loc, scale=scale)

    def save(self, path: str):
        state_dict = self.network.state_dict()
        state_dict["min_action"] = self.min_action
        state_dict["max_action"] = self.max_action
        self._save_state_dict(state_dict, path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.min_action = state_dict.pop("min_action")
        self.max_action = state_dict.pop("max_action")
        self.network.load_state_dict(state_dict)
