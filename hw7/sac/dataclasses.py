import typing as t
from dataclasses import dataclass

import torch
from torch.optim import Optimizer
import gym

from sac.agent import SacAgent
from sac.utils.network import Network


@dataclass
class TrainArguments:
    num_episodes: int = 1000
    gamma: float = 0.99
    alpha: t.Optional[float] = None
    batch_size: int = 64
    policy_network_lr: float = 5e-4
    q_network_lr: float = 5e-4
    alpha_lr: float = 1e-4
    q_network_update_tau: float = 0.01
    q_network_hidden_sizes: t.Sequence[int] = (128, 128)
    max_trajectory_length: int = 1000
    num_q_networks: int = 2
    max_entropy_ratio: float = 0.98


@dataclass
class Experiment:
    agent: SacAgent
    env: gym.Env
    q_networks: t.Sequence[Network]
    target_q_networks: t.Sequence[Network]
    log_alpha: torch.Tensor
    policy_network_optimizer: Optimizer
    q_network_optimizers: t.Sequence[Optimizer]
    alpha_optimizer: t.Optional[Optimizer]
    target_entropy: t.Optional[torch.Tensor]
    train_alpha: bool

    @property
    def num_actions(self) -> int:
        return self.env.action_space.n
