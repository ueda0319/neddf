from typing import Dict

import torch
from torch import Tensor

from neddf.loss.base_loss import BaseLoss


class ColorLoss(BaseLoss):
    """MaskBCELoss.

    This class inheriting base_loss calculate penalty of color difference.
    The formulation is from original nerf paper.

    Attributes:
        weight (float): weight for color loss
        weight_coarse (float): weight for color loss in coarse model
    """

    def forward(
        self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        assert "color" in outputs
        assert "color" in targets
        loss_dict: Dict[str, Tensor] = {}
        loss_dict["color"] = torch.mean(
            torch.square(outputs["color"] - targets["color"])
        )
        if self.weight_coarse > 0.0:
            assert "color_coarse" in outputs
            loss_dict["color_coarse"] = torch.mean(
                torch.square(outputs["color_coarse"] - targets["color"])
            )

        return loss_dict
