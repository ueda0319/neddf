import torch
from neddf.loss.base_loss import BaseLoss
from torch import Tensor


class FieldsConstraintLoss(BaseLoss):
    """FieldsConstraintLoss.

    This class inheriting base_loss calculate penalty for constraints of field.
    The formulation is from neddf paper.

    Attributes:
        weight (float): weight for color loss.
        weight_coarse (float): weight for color loss in coarse model.
        key_output (str): dictionary key in output, set to 'fields_penalty'.
        key_target (str): dictionary key in target, set to 'fields_penalty'.
        key_loss (str): dictionary key in return, set to 'fields_penalty'.
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
            key_output="fields_penalty",
            key_target="fields_penalty",
            key_loss="fields_penalty",
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
        res = torch.mean(output)
        return res
