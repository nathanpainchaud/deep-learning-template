from typing import Any, Dict

from pl_bolts.datamodules import MNISTDataModule as pl_MNISTDataModule

from deep_learning_template import BaseDataModule


class MNISTDataModule(pl_MNISTDataModule, BaseDataModule):
    """Wrapper around the MNIST datamodule provided by Bolts that conforms to our project's datamodule API."""

    @property
    def hyper_parameters(self) -> Dict[str, Any]:  # noqa: D102
        return {"shape": self.dims}
