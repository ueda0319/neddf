import torch
from neddf.loss.base_loss import BaseLoss
from torch import Tensor


class ColorLoss(BaseLoss):
    """ColorLoss.

    This class inheriting base_loss calculate penalty of color difference.
    The formulation is from original nerf paper.
    (https://arxiv.org/abs/2003.08934)

    Attributes:
        weight (float): weight of this loss function.
        weight_coarse (float): weight of this loss function in coarse model.
        key_output (str): dictionary key in output, set to 'color'.
        key_target (str): dictionary key in target, set to 'color'.
        key_loss (str): dictionary key in return, set to 'color'.
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
            key_output="color",
            key_target="color",
            key_loss="color",
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
        res = torch.mean(torch.square(output - target))
        return res
