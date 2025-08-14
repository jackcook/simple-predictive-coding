from typing import Literal

import torch
from torch import nn


def MLPLayer(input_size: int, output_size: int) -> nn.Module:  # noqa: N802
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LeakyReLU(),
    )


class MLP(nn.Module):
    """A multi-layer perceptron."""

    def __init__(
        self,
        in_features: int = 28 * 28,
        num_classes: int = 10,
        channel_sizes: list[int] = [128, 128],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                *[
                    MLPLayer(
                        in_features if i == 0 else channel_sizes[i - 1],
                        channel_sizes[i],
                    )
                    for i in range(len(channel_sizes))
                ],
                MLPLayer(channel_sizes[-1], num_classes),
            ],
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = [x]

        for layer in self.layers:
            x = layer(x)
            activations.append(x)

        return activations


class VGG5(nn.Module):
    """A VGG-5 convolutional neural network."""

    # Default parameters taken from Table 5 of https://arxiv.org/pdf/2407.01163
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channel_sizes: list[int] = [128, 256, 512, 512],
        kernel_sizes: list[int] = [3, 3, 3, 3],
        strides: list[int] = [1, 1, 1, 1],
        paddings: list[int] = [1, 1, 1, 0],
        pool_window_size: int = 2,
        pool_stride: int = 2,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels if i == 0 else channel_sizes[i - 1],
                            channel_sizes[i],
                            kernel_size=kernel_sizes[i],
                            stride=strides[i],
                            padding=paddings[i],
                        ),
                        nn.BatchNorm2d(channel_sizes[i]),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=pool_window_size, stride=pool_stride),
                    )
                    for i in range(len(channel_sizes))
                ],
                nn.Flatten(),
                nn.Sequential(
                    nn.Linear(channel_sizes[-1], hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, num_classes),
                ),
            ],
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = [x]

        for layer in self.layers:
            x = layer(x)
            activations.append(x)

        return activations


def get_model(model_id: Literal["mlp", "vgg5"]) -> nn.Module:
    if model_id == "mlp":
        return MLP()
    if model_id == "vgg5":
        return VGG5()
    raise ValueError(f"Invalid model ID: {model_id}")
