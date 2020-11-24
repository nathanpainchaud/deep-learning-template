from typing import Any, Dict

from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    """Abstract base class for a datamodule, that defines the interface to implement to work with models and tasks."""

    @property
    def hyper_parameters(self) -> Dict[str, Any]:
        """Parameters related to the data that might be useful to configure models and tasks."""
        return {}
