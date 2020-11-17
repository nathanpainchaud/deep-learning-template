from typing import List

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn

from deep_learning_tutorials import BaseDataModule, BaseModel, BaseTask


def validate_cfg(cfg: DictConfig) -> None:
    """Performs static checks on the provided application configuration.

    Args:
        cfg: Application configuration.
    """
    _validate_task_model_data_combo(cfg)
    _validate_optimizers_conf(cfg.optimizers)


def _validate_task_model_data_combo(cfg: DictConfig) -> None:
    """Validates that the task, model and datamodule requested are compatible with each other.

    Args:
        cfg: Application configuration.
    """
    task = cfg.task.task._target_
    for config_group in [cfg.model, cfg.datamodule]:
        if not any(task == authorized_task._target_ for authorized_task in config_group.authorized_tasks):
            raise MisconfigurationException(f"Provided task '{task}' isn't supported by config group {config_group}")


def _validate_optimizers_conf(optimizers_cfg: DictConfig) -> None:
    """Validates that the generated optimizers' configuration is valid by itself.

    Args:
        optimizers_cfg: Optimizers configuration.
    """
    names = []
    for optim_conf in optimizers_cfg.optimizers:
        not_name = not hasattr(optim_conf, "name")
        if not_name:
            raise MisconfigurationException(
                f"'name' attribute should be within the optimizer config group: {optim_conf} "
            )
        else:
            names.append(optim_conf.name)
    if len(names) != len(set(names)):
        raise MisconfigurationException(
            f"Each optimizer name should be unique. Here is the list of optimizer names: {names}"
        )


def initialize_task(cfg: DictConfig, data_module: LightningDataModule) -> BaseTask:
    """Instantiates a task based on an application configuration and a datamodule.

    Args:
        cfg: Application configuration.
        data_module: Data module.

    Returns:
        Task instantiated based on the provided configuration and datamodule.
    """
    # We can use a base `LightningDataModule` if no hyperparameters from the datamodule are required
    data_hyper_parameters = data_module.hyper_parameters if isinstance(data_module, BaseDataModule) else {}

    # We can use a base `nn.Module` if we only require a single end-to-end module
    model: nn.Module = instantiate(cfg.model.model, data_params=data_hyper_parameters)
    model_modules = model.model if isinstance(model, BaseModel) else {"model": model}

    # Initialize the task based on the configuration, as well as the model and data
    task: BaseTask = instantiate(cfg.task.task, cfg=cfg, data_params=data_hyper_parameters, **model_modules)
    return task


def initialize_loggers(cfg: DictConfig, *args, **kwargs) -> List[pl.callbacks.Callback]:
    """Instantiates loggers based on an application configuration.

    Args:
        cfg: Application configuration.
        *args: Positional parameters to pass along to every logger.
        **kwargs: Keyword parameters to pass along to every logger.

    Returns:
        Loggers instantiated based on the provided configuration.
    """
    loggers = []
    if cfg.log:
        for logger in cfg.loggers.loggers:
            loggers.append(instantiate(logger, *args, **kwargs))
    return loggers
