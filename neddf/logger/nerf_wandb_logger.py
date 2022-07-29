import os
from typing import Dict, Optional

import wandb

from .nerf_logger_abstract import NeRFLoggerAbstract


class NeRFWandBLogger(NeRFLoggerAbstract):
    """Logger for Weights & Biases."""

    is_initialized = False

    def __init__(
        self,
        project: Optional[str] = None,
        log_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        config: Optional[Dict] = None,
    ) -> None:
        """

        Args:
            project (Optional[str], optional): Project name of wandb
            log_dir (Optional[str], optional): Directory to log output
            offline (Optional[bool], optional): Set true to offline mode
            config (Optional[Dict], optional): Other wandb configurations
        """
        super().__init__()

        if not NeRFWandBLogger.is_initialized:
            if offline:
                os.environ["WANDB_MODE"] = "offline"
            else:
                os.environ["WANDB_MODE"] = "run"
            NeRFWandBLogger.is_initialized = True

        if log_dir is None:
            self.wandb_d = wandb.init(project=project, config=config)
        else:
            wandb.tensorboard.patch(root_logdir=log_dir)
            self.wandb_d = wandb.init(
                project=project, sync_tensorboard=True, config=config
            )

        assert self.wandb_d is not None

    def __del__(self) -> None:
        if isinstance(self.wandb_d, wandb.sdk.wandb_run.Run):
            self.wandb_d.finish()

    def _next_impl(self, data: Dict) -> None:
        """@implement

        Args:
            data (Dict): Registered data to be written passed from Super class
        """
        wandb.log(data)


if __name__ == "__main__":
    print("!!!under construction!!!")

    # import random
    # from time import sleep

    # logger = NeRFWandBLogger("test", "log", False, {})
    # for _ in range(20):
    #     logger.write_batchstart()
    #     sleep(1.0 + random.random())
    #     logger.write_batchend()
    #     logger.write(1.0 + random.random(), random.random())
    #     logger.next()
    #     sleep(random.random())
