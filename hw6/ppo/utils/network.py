import typing as t
from logging import getLogger

import torch
from torch import nn

logger = getLogger(__name__)


class Network(nn.Module):
    def __init__(self, layer_sizes: t.Sequence[int]):
        if len(layer_sizes) < 2:
            raise ValueError(
                "Length of layer_sizes must be greater than or equal to 2."
            )
        super().__init__()
        layers = []
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
        layers = layers[:-1]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def reset_parameters(self):
        for layer in self.network.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            else:
                logger.warning(
                    f"Layer {layer} doesn't have reset_parameters method"
                )

    @property
    def device(self) -> torch.device:
        return list(self.network.parameters())[0].device
