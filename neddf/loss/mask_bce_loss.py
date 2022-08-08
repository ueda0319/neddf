import torch
from neddf.loss.base_loss import BaseLoss
from torch import Tensor


class MaskBCELoss(BaseLoss):
    """MaskBCELoss.

    This class inheriting base_loss calculate penalty of mask difference.
    The formulation is Binary Cross Entropy(BCE), from NeuS paper.
    (https://arxiv.org/abs/2106.10689)

    Attributes:
        weight (float): weight for mask loss.
        weight_coarse (float): weight for mask loss in coarse model.
        key_output (str): dictionary key in output, set to 'transmittance'.
        key_target (str): dictionary key in target, set to 'mask'.
        key_loss (str): dictionary key in return, set to 'mask'.
    """

    def __init__(
        self,
        weight: float = 1.0,
        weight_coarse: float = 0.1,
    ) -> None:
        """Initializer

        Args:
            weight (float): weight of this loss function
            weight_coarse (float): weight of this loss function in coarse model

        """
        super().__init__(
            key_output="transmittance",
            key_target="mask",
            key_loss="mask",
            weight=weight,
            weight_coarse=weight_coarse,
        )

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Loss

        Calculate loss value in the objective function

        Args:
            output (Tensor): Inference result of network
            target (Tensor): The target values output should have taken

        Returns:
            Tensor[1, float]: Calculated loss value

        """
        mask_output: Tensor = torch.clamp(1.0 - output, 1e-6, 1.0 - 1e-6)
        res = -torch.mean(
            target * torch.log(mask_output)
            + (1.0 - target) * torch.log(1.0 - mask_output)
        )
        return res
