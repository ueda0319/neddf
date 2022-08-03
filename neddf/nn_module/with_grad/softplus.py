from typing import Optional, Tuple
import torch
from torch import nn, Tensor


class SoftplusGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function._ContextMethodMixin,
        x: Tensor,
        J: Tensor,
        threshold: float = 20.0,
    ):
        """forward

        Forward calculation for SoftplusGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            x (Tensor[batch_size, input_ch, float]): input features
            J (Tensor[batch_size, 3, input_ch, float]): gradients of input features
        """
        bs, input_ch = x.shape
        mask = x > threshold
        ex = torch.exp(x)
        iex = torch.exp(-x)
        y = torch.log(1.0 + ex)
        y[mask] = x[mask]

        dydx = 1.0 / (1.0 + iex)
        dydx[mask] = 1.0

        dGdJ = dydx.unsqueeze(1).expand_as(J)
        G = dGdJ * J

        ctx.save_for_backward(J, dydx, dGdJ, mask)
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
        J, dydx, dGdJ, mask = ctx.saved_tensors
        d2x = (1 - dydx) * dydx
        d2x[mask] = 0.0
        dGdx = d2x * torch.sum(J * dLdG, 2)
        dLdx = dLdy * dydx + dGdx
        dLdJ = dLdG * dGdJ
        return dLdx, dLdJ, None
