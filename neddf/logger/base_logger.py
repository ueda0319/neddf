from abc import ABC, abstractmethod
from time import time
from typing import Dict

from torch import Tensor


class BaseLogger(ABC):
    """BaseLogger

    Abstruct logger to be implemented.
    """

    def __init__(self) -> None:
        """Initializer configure common attributes."""
        self.loss: float = 0.0
        self.psnr: float = 0.0
        self.loss_dict: Dict[str, float] = {}
        self.loggerstart: float = time()
        self.batchstart: float = self.loggerstart
        self.prev_batchend: float = self.loggerstart
        self.batchend: float = self.loggerstart
        self.niter: int = 0

    def reset(self) -> None:
        """Reset registered data."""
        self.loss = 0.0
        self.psnr = 0.0
        self.loss_dict = {}
        self.niter = 0
        self.loggerstart = time()
        self.batchstart = self.loggerstart
        self.prev_batchend = self.loggerstart
        self.batchend = self.loggerstart

    def write(self, loss: float, psnr: float, loss_dict: Dict[str, Tensor]) -> None:
        """Register data of an iteration to be written.

        Args:
            loss (float): loss for the iteration
            psnr (float): PSNR for the iteration
            loss_dict (Dict[str, Tensor]): other objective function's values for the iteration

        """
        self.loss = loss
        self.psnr = psnr
        self.loss_dict = {key: float(loss_dict[key].item()) for key in loss_dict}

    def write_batchstart(self) -> None:
        """Register start time of an iteration. Please execute before an iteration."""
        self.prev_batchend = self.batchend
        self.batchstart = time()

    def write_batchend(self) -> None:
        """Register end time of an iteration. Please execute after an iteration."""
        self.batchend = time()

    def next(self) -> None:
        """Submit data to a logger backend."""
        log_dict: Dict[str, float] = {
            "loss": self.loss,
            "PSNR": self.psnr,
            "iteration duration": self.batchend - self.batchstart,
            "total duration": self.batchend - self.loggerstart,
        }
        for key in self.loss_dict:
            log_dict["objective/{}".format(key)] = self.loss_dict[key]

        self._next_impl(log_dict)
        self.niter += 1

    @abstractmethod
    def _next_impl(self, data: Dict) -> None:
        """Abstrruct method to be implemented.

        Args:
            data (Dict): Parameters for implemented logger
        """
        raise NotImplementedError()
