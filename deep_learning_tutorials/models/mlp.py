from functools import partial, reduce
from operator import mul
from typing import Sequence, Tuple, Union

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from deep_learning_tutorials.models.layers import bn_activation_dropout_block


class MultiLayerPerceptron(nn.Module):
    """Multi-layer perceptron network."""

    def __init__(
        self,
        in_shape: Union[int, Sequence[int]],
        out_shape: Union[int, Sequence[int]],
        hidden_layers: Tuple[int, ...],
        activation: DictConfig,
        bn: DictConfig = None,
        dropout: DictConfig = None,
    ):  # noqa: D205,D212,D415
        """
        Args:
            in_shape: Shape of the network's input. If this is not 1-dimensional, the input will be automatically
                flattened at the start of the forward pass.
            out_shape: Shape of the network's output. If this is not 1-dimensional, the output will be automatically
                reshaped at the end of the forward pass.
            hidden_layers: Number of hidden neurons in each layer. If left empty, will correspond to a single FC layer
                linear model between the input and output.
            activation: Configuration options for the activation layer.
            bn: Configuration options for the (optional) batchnorm layers.
            dropout: Configuration options for the (optional) dropout layers.
        """
        super().__init__()
        if not isinstance(out_shape, Sequence):  # Ensure the dimension is a sequence (even w/ a single element)
            out_shape = [out_shape]
        self.out_shape = out_shape

        def flatten_shape(shape: Union[int, Sequence[int]]) -> int:
            if isinstance(shape, Sequence):
                shape = reduce(mul, shape)
            return shape

        aux_layers_builder = partial(bn_activation_dropout_block, activation=activation, bn=bn, dropout=dropout)
        layers = (flatten_shape(in_shape), *hidden_layers)
        self.model = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(layers[i - 1], layers[i]), *aux_layers_builder(layers[i]))
                for i in range(1, len(layers))
            ],
            nn.Linear(layers[-1], flatten_shape(out_shape)),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        # Ensure the input is flat
        x = torch.flatten(x, start_dim=1)
        x = self.model(x)
        # Reshape the output to the expected output shape
        x = torch.reshape(x, (-1, *self.out_shape))
        return x
