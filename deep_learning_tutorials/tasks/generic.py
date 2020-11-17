from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor

from deep_learning_tutorials import BaseTask
from deep_learning_tutorials.utils.format.native import prefix


class TrainValTask(BaseTask, ABC):
    """Abstract class for task whose train and validation loops are similar."""

    @abstractmethod
    def train_val_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Handles steps for both training and validation loops, assuming the behavior should be the same.

        For models where the behavior in training and validation is different, then override ``training_step`` and
        ``validation_step`` directly (in which case ``train_val_step`` doesn't need to be implemented).

        Returns:
            Mapping between metric names and their values. It must contain at least a ``'loss'``, as that is the value
            optimized in training and monitored by callbacks during validation.
        """

    def training_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self.train_val_step(*args, **kwargs), "train_")
        self.log_dict(result, **self.hparams.logging.train)
        # Add reference to 'train_loss' under 'loss' keyword, requested by PL to know which metric to optimize
        result["loss"] = result["train_loss"]
        return result

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self.train_val_step(*args, **kwargs), "val_")
        self.log_dict(result, **self.hparams.logging.validation)
        return result
