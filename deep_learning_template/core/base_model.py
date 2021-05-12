from abc import ABC
from typing import Any, Dict, Mapping, Sequence, Tuple, Type

from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for a model, that defines the interface to implement to work with tasks."""

    def __init__(self, data_params: Mapping[str, Any], required_data_attrs: Sequence[Tuple[str, Type]] = None):
        """Initializes class instance.

        Args:
            data_params: Hyper-parameters provided by the datamodule.
            required_data_attrs: Hyper-parameters (w/ their types) that should be provided by the datamodule for the
                model to know what architecture to initialize.
        """
        super().__init__()
        self._validate_data_params(data_params, required_data_attrs=required_data_attrs)
        self.data_params = AttributeDict(data_params)

    @staticmethod
    def _validate_data_params(
        data_params: Mapping[str, Any], required_data_attrs: Sequence[Tuple[str, Type]] = None
    ) -> None:
        if required_data_attrs:
            errors = []
            for data_attr, attr_type in required_data_attrs:
                if data_attr not in data_params:
                    errors.append(
                        f"Model is missing parameter '{data_attr}' of type '{attr_type}' that should be provided by "
                        "the datamodule's `hyper_parameters` property."
                    )
                # TODO Check the type of the data attribute when it's not missing
            if errors:
                err_msg = "\n".join([f"{err_count+1}) {err_msg}" for err_count, err_msg in enumerate(errors)])
                raise MisconfigurationException(f"Error(s) encountered when initializing model: \n{err_msg}")

    @property
    def model(self) -> Dict[str, nn.Module]:
        """The modules, or submodules, making up the model's architecture."""
        return {"model": self}
