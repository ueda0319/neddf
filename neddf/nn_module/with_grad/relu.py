from typing import Tuple

import torch
from torch import Tensor


class ReLUGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: torch.autograd.function._ContextMethodMixin,
        x: Tensor,
        J: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """forward

        Forward calculation for ReLUGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            x (Tensor[batch_size, input_ch, float]): input features
            J (Tensor[batch_size, 3, input_ch, float]): gradients of input features
        """
        mask = x >= 0
        y = x * mask
        G = J * mask.unsqueeze(1)
        ctx.save_for_backward(mask)  # type: ignore
        return y, G

    @staticmethod
    def backward(  # type: ignore
        ctx,
        dLdy: Tensor,
        dLdG: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """backward

        Backward calculation for LinearGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            dLdy (Tensor[batch_size, output_ch, float]): output features
            dLdG (Tensor[batch_size, 3, output_ch, float]): gradients of output features

        Contexts:
            mask (Tensor[batch_size, input_ch, bool])
        """
        mask = ctx.saved_tensors  # type: ignore
        dLdx = dLdy * mask
        dLdJ = dLdG * mask.unsqueeze(1)

        return dLdx, dLdJ
