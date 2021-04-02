from pathlib import Path
from typing import Type

import torch
from hydra.utils import get_class
from pytorch_lightning.core.saving import ModelIO

from deep_learning_template.core import BaseTask


def load_task_from_checkpoint(checkpoint_path: Path):
    """Loads a task checkpoint, casting it to the appropriate type.

    The module's class is automatically determined based on the hyperparameters saved in the checkpoint.

    Args:
        checkpoint_path: Path to the task checkpoint to load.

    Returns:
        Task loaded from the checkpoint, casted to its original type.
    """
    hparams = torch.load(checkpoint_path)[ModelIO.CHECKPOINT_HYPER_PARAMS_KEY]
    task_cls: Type[BaseTask] = get_class(hparams["task"]["task"]["_target_"])
    return task_cls.load_from_checkpoint(str(checkpoint_path), ckpt_path=checkpoint_path)
