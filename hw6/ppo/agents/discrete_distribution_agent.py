from logging import getLogger

import torch
from torch.distributions import Distribution, Categorical

from ppo.agents.abstract_agent import AbstractAgent

logger = getLogger(__name__)


class DiscreteDistributionAgent(AbstractAgent):
    def _get_distribution(self, predictions: torch.Tensor) -> Distribution:
        logger.debug(f"Distribution: logits = {predictions}")
        return Categorical(logits=predictions)

    def save(self, path: str):
        self._save_state_dict(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
