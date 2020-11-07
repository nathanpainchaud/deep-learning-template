import logging
import os

os.environ["HYDRA_FULL_ERROR"] = "1"

log = logging.getLogger(__name__)


if __name__ == "__main__":
    log.info("Program is running ...")
