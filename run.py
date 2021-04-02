import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from deep_learning_template.core import initialization as init
from deep_learning_template.utils.config import freeze_config
from deep_learning_template.utils.serialization import load_task_from_checkpoint

os.environ["HYDRA_FULL_ERROR"] = "1"

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra application's main function, that builds the model and trains/tests it on the dataset.

    Args:
        cfg: Application-wide configuration generated by Hydra.
    """
    freeze_config(cfg)
    log.info(f"Hydra app running with config: \n{OmegaConf.to_yaml(cfg)}")
    init.validate_cfg(cfg)

    data_module = instantiate(cfg.datamodule.datamodule)

    if cfg.ckpt_path:
        trainer = Trainer(resume_from_checkpoint=cfg.ckpt_path)
        task = load_task_from_checkpoint(cfg.ckpt_path)
    else:
        trainer = instantiate(cfg.trainer.trainer, logger=init.initialize_loggers(cfg))
        task = init.initialize_task(cfg, data_module)

    if cfg.trainer.train:
        trainer.fit(task, datamodule=data_module)
        log.info("Training complete.")
    elif not cfg.ckpt_path:
        log.warning(
            "Trainer set to skip training (`train` flag is `False`) without a checkpoint being provided. "
            "Downstream operations will use randomly initialized weights for the model."
        )

    if cfg.trainer.test:
        trainer.test(task, datamodule=data_module)
        log.info("Testing complete.")


if __name__ == "__main__":
    main()
