import typing as t

import torch
import torch.nn as nn


class AgentNetwork(nn.Module):
    def __init__(
        self, 
        layers_n_features: t.List[int],
        min_state: t.Optional[torch.Tensor] = None,
        max_state: t.Optional[torch.Tensor] = None,
    ):
        super().__init__()
        layers = []
        for in_features, out_features in zip(
            layers_n_features[:-1], layers_n_features[1:]
        ):
            layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
        layers = layers[:-1]
        self.layers = nn.Sequential(*layers)

        if min_state is None or max_state is None:
            mean_state = torch.zeros((layers_n_features[0],))
            std_state = torch.ones((layers_n_features[0],))
        else:
            mean_state = (min_state + max_state) / 2
            std_state = max_state - mean_state

        zeros = torch.zeros_like(std_state, device=std_state.device)
        if torch.any(torch.isclose(std_state, zeros)):
            raise ValueError("std_state vector contains zero elements")

        self.mean_state = nn.Parameter(mean_state, requires_grad=False)
        self.std_state = nn.Parameter(std_state, requires_grad=False)

    def forward(self, x):
        x = (x - self.mean_state) / self.std_state
        return self.layers(x)
