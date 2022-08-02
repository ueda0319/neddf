from typing import Dict, Final, Tuple, Literal

import torch
from torch import Tensor


class Sampling:
    def __init__(
        self,
        sample_pos: Tensor,
        sample_dir: Tensor,
        diag_variance: Tensor,
    ) -> None:
        self.sample_pos: Tensor = sample_pos
        self.sample_dir: Tensor = sample_dir
        self.diag_variance: Tensor = diag_variance

    @property
    def device(self) -> torch.device:
        return self.sample_pos.device

