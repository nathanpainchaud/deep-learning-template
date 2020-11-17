from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from torch import nn

from deep_learning_tutorials import BaseModel
from deep_learning_tutorials.models.mlp import MultiLayerPerceptron


class FullyConnectedAutoencoder(BaseModel):
    """Fully-connected autoencoder model, for basic datasets like MNIST."""

    def __init__(self, data_params: Mapping[str, Any], num_latent_dims: int, hidden_layers: Tuple[int, ...], **kwargs):
        super().__init__(data_params, required_data_attrs=[("shape", Union[int, Sequence[int]])])
        self.encoder = MultiLayerPerceptron(self.data_params.shape, num_latent_dims, hidden_layers, **kwargs)
        self.decoder = MultiLayerPerceptron(num_latent_dims, self.data_params.shape, hidden_layers[::-1], **kwargs)

    @property
    def model(self) -> Dict[str, nn.Module]:  # noqa: D102
        return {"encoder": self.encoder, "decoder": self.decoder}
