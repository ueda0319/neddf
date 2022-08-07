import torch
from neddf.loss.base_loss import BaseLoss
from torch import Tensor


class RangeLoss(BaseLoss):
    """RangeLoss.

    This class inheriting base_loss calculate penalty for values outside range.

    Attributes:
        weight (float): weight for color loss
        weight_coarse (float): weight for color loss in coarse model
    """

    def __init__(
        self,
        weight: float = 1.0,
        weight_coarse: float = 0.1,
    ) -> None:
        super().__init__(
            key_output="range_penalty",
            key_target="range_penalty",
            key_loss="range_penalty",
            weight=weight,
            weight_coarse=weight_coarse,
        )

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        res = torch.mean(output)
        return res
