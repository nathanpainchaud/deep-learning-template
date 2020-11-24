from abc import ABC
from typing import Any, Mapping

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn

from deep_learning_template.utils.config import freeze_config, unfreeze_config


class BaseTask(LightningModule, ABC):
    """Abstract base class for a task, that performs generic setup based on the config."""

    def __init__(self, cfg: DictConfig, data_params: Mapping[str, Any]):  # noqa: D205,D212,D415
        """
        Args:
            cfg: Application configuration.
            data_params: Hyper-parameters provided by the datamodule.
        """
        super().__init__()
        self.save_hyperparameters(cfg)

        # Temporarilly re-allow modification of config to add `data_params` field
        unfreeze_config(cfg)
        self.hparams["data_params"] = data_params
        freeze_config(cfg)

    def configure_optimizers(self):  # noqa: D102
        # TODO Add support for learning rate scheduler in optimizer config
        optimizers = []
        for optimizer in self.hparams.optimizers.optimizers:
            # Convert & copy the optimizer config to be able to pop the `'name'` entry w/o affecting the base config
            optimizer = dict(optimizer)
            module_name = optimizer.pop("name")
            module: nn.Module = getattr(self, module_name) if module_name != "default" else self
            optimizers.append(instantiate(optimizer, params=module.parameters()))
        return optimizers
