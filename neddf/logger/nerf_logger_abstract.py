from abc import abstractmethod
from time import time
from typing import Dict


class NeRFLoggerAbstract:
    """NeRFLoggerAbstract

    Abstruct logger to be implemented.
    """

    def __init__(self) -> None:
        self.loss: float = 0.0
        self.psnr: float = 0.0
        self.loggerstart: float = time()
        self.batchstart: float = self.loggerstart
        self.prev_batchend: float = self.loggerstart
        self.batchend: float = self.loggerstart
        self.niter: int = 0

    def reset(self) -> None:
        """Reset registered data."""
        self.loss = 0.0
        self.psnr = 0.0
        self.niter = 0
        self.loggerstart = time()
        self.batchstart = self.loggerstart
        self.prev_batchend = self.loggerstart
        self.batchend = self.loggerstart

    def write(self, loss: float, psnr: float) -> None:
        """Register data of an iteration to be written.

        Args:
            loss (float): loss for the iteration
            psnr (float): PSNR for the iteration
        """
        self.loss = loss
        self.psnr = psnr

    def write_batchstart(self) -> None:
        """Register start time of an iteration. Please execute before an iteration."""
        self.prev_batchend = self.batchend
        self.batchstart = time()

    def write_batchend(self) -> None:
        """Register end time of an iteration. Please execute after an iteration."""
        self.batchend = time()

    def next(self) -> None:
        """Submit data to a logger backend."""
        self._next_impl(
            {
                "loss": self.loss,
                "PSNR": self.psnr,
                "iteration duration": self.batchend - self.batchstart,
                "dataload duration": self.batchstart - self.prev_batchend,
                "total duration": self.batchend - self.loggerstart,
            }
        )
        self.niter += 1

    @abstractmethod
    def _next_impl(self, data: Dict) -> None:
        """Abstrruct method to be implemented.

        Args:
            data (Dict): Parameters for implemented logger
        """
        None
