from typing import Dict, Literal

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.core.decorators import auto_move_data
from torch import Tensor, nn

from deep_learning_template.tasks.generic import TrainValTask


class PixelWiseAutoencodingTask(TrainValTask):
    """Task that trains an autoencoder to reconstruct images in a pixel-wise manner."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss: DictConfig, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss = instantiate(loss)

    @auto_move_data
    def forward(  # noqa: D102
        self, x: Tensor, task: Literal["encode", "decode", "reconstruct"] = "reconstruct"
    ) -> Tensor:
        if task in ["encode", "reconstruct"]:
            x = self.encoder(x)
        if task in ["decode", "reconstruct"]:
            x = self.decoder(x)
        return x

    def train_val_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:  # type: ignore # noqa: D102
        # Unpack input
        x, _ = batch

        # Forward
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # Loss
        loss = self.loss(x_hat, x)

        # Metrics to log
        return {"loss": loss}
