import torch
from neddf.loss.base_loss import BaseLoss
from torch import Tensor


class MaskBCELoss(BaseLoss):
    """MaskBCELoss.

    This class inheriting base_loss calculate penalty of mask difference.
    The formulation is Binary Cross Entropy(BCE), from NeuS paper.

    Attributes:
        weight (float): weight for mask loss
        weight_coarse (float): weight for mask loss in coarse model
    """

    def __init__(
        self,
        weight: float = 1.0,
        weight_coarse: float = 0.1,
    ) -> None:
        super().__init__(
            key_output="transmittance",
            key_target="mask",
            key_loss="mask",
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
        mask_output: Tensor = torch.clamp(1.0 - output, 1e-6, 1.0 - 1e-6)
        res = -torch.mean(
            target * torch.log(mask_output)
            + (1.0 - target) * torch.log(1.0 - mask_output)
        )
        return res
