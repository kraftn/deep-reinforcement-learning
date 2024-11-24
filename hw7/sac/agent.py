import os
from logging import getLogger

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from sac.utils import Network

logger = getLogger(__name__)


class SacAgent:
    def __init__(self, network: Network, temperature: float = 1.0):
        self.network = network
        self.temperature = temperature

    def sample_actions(self, states: torch.Tensor) -> torch.Tensor:
        distribution = self._get_distribution(states, mode="eval")
        return distribution.sample()

    def calc_log_probs(
        self,
        states: torch.Tensor,
        mode: str = "train"
    ) -> torch.Tensor:
        # states = [batch_size, state_dim]
        # actions  = [batch_size]

        dist = self._get_distribution(states, mode)
        actions = torch.arange(
            dist.logits.size(-1), device=self.network.device
        ).unsqueeze(1)
        log_probs = dist.log_prob(actions).T
        # log_probs = [batch_size, num_actions]

        return log_probs

    @property
    def device(self):
        return self.network.device

    def save(self, file_path: str):
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(self.network.state_dict(), file_path)

    def load(self, file_path: str):
        self.network.load_state_dict(
            torch.load(file_path, map_location=self.device)
        )

    def reset(self):
        self.network.reset_parameters()

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

    def _get_distribution(self, states: torch.Tensor, mode: str) -> Categorical:
        # states = [batch_size, state_dim]
        logits = self._get_predictions(states, mode) / self.temperature
        # logits = [batch_size, num_actions]

        probas = F.softmax(logits, dim=-1)
        logger.debug(f"Probabilities of actions: {probas}")

        return Categorical(logits=logits)
