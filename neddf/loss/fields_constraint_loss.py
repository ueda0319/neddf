import torch
from neddf.loss.base_loss import BaseLoss
from torch import Tensor


class FieldsConstraintLoss(BaseLoss):
    """FieldsConstraintLoss.

    This class inheriting base_loss calculate penalty for constraints of field.
    The formulation is from neddf paper.

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
            key_output="fields_penalty",
            key_target="fields_penalty",
            key_loss="fields_penalty",
            weight=weight,
            weight_coarse=weight_coarse,
        )

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        res = torch.mean(output)
        return res