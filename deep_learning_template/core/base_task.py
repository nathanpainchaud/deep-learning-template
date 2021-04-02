from abc import ABC
from typing import IO, Any, Callable, Dict, Mapping, Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.core.saving import ModelIO
from torch import nn

from deep_learning_template.utils.config import freeze_config, unfreeze_config


class BaseTask(LightningModule, ABC):
    """Abstract base class for a task, that performs generic setup based on the config."""

    def __init__(self, cfg: DictConfig, data_params: Mapping[str, Any]):
        """Initializes class instance.

        Args:
            cfg: Application configuration.
            data_params: Hyper-parameters provided by the datamodule.
        """
        super().__init__()

        # Temporarily re-allow modification of config to add data hyper-parameters
        unfreeze_config(cfg)
        cfg["data"] = data_params
        freeze_config(cfg)

        self.save_hyperparameters(cfg)

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

    @classmethod
    def load_from_checkpoint(  # noqa: D102
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        # Import locally to avoid circular import
        from deep_learning_template import BaseModel

        # Load the configuration from the checkpoint
        hparams = torch.load(checkpoint_path)[ModelIO.CHECKPOINT_HYPER_PARAMS_KEY]

        # Rebuild the model's architecture from the configuration stored in the checkpoint
        model = instantiate(hparams.model.model, data_params=hparams.data)
        model_modules = model.model if isinstance(model, BaseModel) else {"model": model}

        # Forward the model's networks to the pipeline that restores the task
        task = super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs,
            **model_modules,
        )

        # Freeze the task's configuration after it's finished loading
        freeze_config(task.hparams)

        return task

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Makes the configuration editable before saving it.

        This is required to be able to override arbitrary values when loading it.

        Args:
            checkpoint: Checkpoint to save.
        """
        unfreeze_config(checkpoint[ModelIO.CHECKPOINT_HYPER_PARAMS_KEY])
