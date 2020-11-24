from omegaconf import DictConfig, OmegaConf


def freeze_config(cfg: DictConfig) -> None:
    """Freezes everything below the config node, i.e. raises the config's read-only and struct flags.

    Args:
        cfg: Root node below which to freeze the config.
    """
    OmegaConf.set_struct(cfg, True)  # Disable read and write access to unknown fields in config
    OmegaConf.set_readonly(cfg, True)  # Make config structure read-only


def unfreeze_config(cfg: DictConfig) -> None:
    """Unfreezes everything below the config node, i.e. lowers the config's read-only and struct flags.

    Args:
        cfg: Root node below which to unfreeze the config.
    """
    OmegaConf.set_struct(cfg, False)  # Re-enable read and write access to unknown fields in config
    OmegaConf.set_readonly(cfg, False)  # Re-allow write access to config structure
