from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from .base_logger import BaseLogger


class NeRFTBLogger(BaseLogger):
    """Logger using TensorBoard."""

    def __init__(self) -> None:
        """

        Args:
            log_dir (str): Directory to log output
        """
        super().__init__()
        log_dir = "log"
        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir)  # type: ignore # noqa

    def _next_impl(self, data: Dict) -> None:
        """@implement

        Args:
            data (Dict): Registered data to be written passed from Super class
        """
        for k in data:
            self.writer.add_scalar(k, data[k], self.niter)  # type: ignore # noqa


if __name__ == "__main__":
    import random
    from time import sleep

    logger = NeRFTBLogger()
    for _ in range(20):
        logger.write_batchstart()
        sleep(1.0 + random.random())
        logger.write_batchend()
        loss_dict = {"loss1": torch.tensor([1.0])}
        logger.write(1.0 + random.random(), random.random(), loss_dict)
        logger.next()
        sleep(random.random())
