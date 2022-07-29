from typing import Dict

from torch.utils.tensorboard import SummaryWriter

from .nerf_logger_abstract import NeRFLoggerAbstract


class NeRFTBLogger(NeRFLoggerAbstract):
    """Logger using TensorBoard."""

    def __init__(self, log_dir: str) -> None:
        """

        Args:
            log_dir (str): Directory to log output
        """
        super().__init__()
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

    logger = NeRFTBLogger("log")
    for _ in range(20):
        logger.write_batchstart()
        sleep(1.0 + random.random())
        logger.write_batchend()
        logger.write(1.0 + random.random(), random.random())
        logger.next()
        sleep(random.random())
