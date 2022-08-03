from typing import Optional, Tuple
import torch
from torch import nn, Tensor


class ReLUGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function._ContextMethodMixin,
        x: Tensor,
        J: Tensor,
    ):
        """forward

        Forward calculation for ReLUGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            x (Tensor[batch_size, input_ch, float]): input features
            J (Tensor[batch_size, 3, input_ch, float]): gradients of input features
        """
        mask = (x >= 0)
        y = x * mask
        G = J * mask.unsqueeze(1)
        ctx.save_for_backward(mask)
        return y, G

    @staticmethod
    def backward(
        ctx,
        dLdy: Tensor,
        dLdG: Tensor,
    ):
        """backward

        Backward calculation for LinearGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            dLdy (Tensor[batch_size, output_ch, float]): output features
            dLdG (Tensor[batch_size, 3, output_ch, float]): gradients of output features
        
        Contexts:
            mask (Tensor[batch_size, input_ch, bool])
        """
        mask = ctx.saved_tensors
        dLdx = dLdy * mask
        dLdJ = dLdG * mask.unsqueeze(1)

        return dLdx, dLdJ
