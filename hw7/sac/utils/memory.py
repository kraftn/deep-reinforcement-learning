import typing as t
import random

import numpy as np
import torch


class Memory:
    batch_dtypes = (
        torch.float32, torch.int64, torch.float32, torch.float32, torch.float32
    )

    def __init__(self, device: str):
        self.device = device
        self.memory = []

    def add_sample(
        self,
        state: t.List[float],
        action: int,
        reward: float,
        next_state: t.List[float],
        done: bool,
    ):
        done = float(done)
        self.memory.append((state, action, reward, next_state, done))

    def get_batch(
        self, batch_size: int
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size > len(self.memory):
            raise RuntimeError("Not enough samples in memory.")
        samples = random.sample(self.memory, batch_size)
        batch_data = list(zip(*samples))
        states, actions, rewards, next_states, dones = [
            torch.tensor(data, dtype=dtype, device=self.device)
            for data, dtype in zip(batch_data, self.batch_dtypes)
        ]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
