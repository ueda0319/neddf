import torch
from neddf.loss.base_loss import BaseLoss
from torch import Tensor


class ColorLoss(BaseLoss):
    """MaskBCELoss.

    This class inheriting base_loss calculate penalty of color difference.
    The formulation is from original nerf paper.

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
            key_output="color",
            key_target="color",
            key_loss="color",
            weight=weight,
            weight_coarse=weight_coarse,
        )

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        """loss

        This method calculate loss in each tensor
        BaseLoss.forward call this method two times(coarse and fine) in a iterration
        Note that output and target should take same shape

        Args:
            output (Tensor): output values or rendered value in neuralfields
            target (Tensor): target values, which always take ground truth

        Returns:
            Tensor[1, float]: calcurated loss value
        """
        res = torch.mean(torch.square(output - target))
        return res
