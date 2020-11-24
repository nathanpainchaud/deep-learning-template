from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn


def bn_activation_dropout_block(
    num_features: int,
    activation: DictConfig,
    bn: DictConfig = None,
    dropout: DictConfig = None,
) -> List[nn.Module]:
    """Generates a BatchNorm -> Activation -> Dropout block of layers, based on the configuration for each layer.

    Args:
        num_features: Number of features the batchnorm would apply to.
        activation: Configuration options for the activation layer.
        bn: Configuration options for the (optional) batchnorm layer.
        dropout: Configuration options for the (optional) dropout layer.

    Returns:
        BatchNorm -> Activation -> Dropout block of layers.
    """
    aux_layers = []
    if bn_layer := instantiate(bn, num_features=num_features):
        aux_layers.append(bn_layer)
    aux_layers.append(instantiate(activation))
    if dropout_layer := instantiate(dropout):
        aux_layers.append(dropout_layer)
    return aux_layers
