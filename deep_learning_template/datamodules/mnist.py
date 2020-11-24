from typing import Any, Dict

from pl_bolts.datamodules import mnist_datamodule

from deep_learning_template import BaseDataModule


class MNISTDataModule(mnist_datamodule.MNISTDataModule, BaseDataModule):
    """Wrapper around the MNIST datamodule provided by Bolts that conforms to our project's datamodule API."""

    @property
    def hyper_parameters(self) -> Dict[str, Any]:  # noqa: D102
        return {"shape": self.dims}
