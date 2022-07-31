from typing import Dict

import torch
from torch import Tensor

from neddf.loss.base_loss import BaseLoss


class MaskBCELoss(BaseLoss):
    """MaskBCELoss.

    This class inheriting base_loss calculate penalty of mask difference.
    The formulation is Binary Cross Entropy(BCE), from NeuS paper.

    Attributes:
        weight (float): weight for mask loss
        weight_coarse (float): weight for mask loss in coarse model
    """

    def forward(
        self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        assert "transmittance" in outputs
        assert "mask" in targets
        loss_dict: Dict[str, Tensor] = {}
        loss_dict["mask"] = self.loss(outputs["transmittance"], targets["mask"])
        if self.weight_coarse > 0.0:
            assert "transmittance_coarse" in outputs
            loss_dict["mask_coarse"] = self.loss(
                outputs["transmittance_coarse"], targets["mask"]
            )

        return loss_dict

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        mask_output: Tensor = torch.clamp(1.0 - output, 1e-6, 1.0 - 1e-6)
        res = -torch.mean(
            target * torch.log(mask_output)
            + (1.0 - target) * torch.log(1.0 - mask_output)
        )
        return res
