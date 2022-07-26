from typing import Optional, Tuple

import torch
from torch import Tensor


class SoftplusGradFunction(torch.autograd.Function):
    """SoftplusGradFunction

    This class inheriting torch.autograd.Function.
    This function calculate ReLU with first order gradient as forward propagation.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: torch.autograd.function._ContextMethodMixin,
        x: Tensor,
        J: Tensor,
        threshold: float = 20.0,
    ) -> Tuple[Tensor, Tensor]:
        """forward

        Forward calculation for SoftplusGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            x (Tensor[batch_size, input_ch, float]): input features
            J (Tensor[batch_size, 3, input_ch, float]): gradients of input features
            threshold (float): threshold to treat linear

        Returns:
            Tuple[
                Tensor[batch_size, input_ch, float]
                Tensor[batch_size, 3, input_ch, float]
            ]

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

        ctx.save_for_backward(J, dydx, dGdJ, mask)  # type: ignore
        return y, G

    @staticmethod
    def backward(  # type: ignore
        ctx,
        dLdy: Tensor,
        dLdG: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """backward

        Backward calculation for LinearGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            dLdy (Tensor[batch_size, output_ch, float]): output features
            dLdG (Tensor[batch_size, 3, output_ch, float]): gradients of output features

        Contexts:
            d2x (Tensor[batch_size, input_ch, bool])
            J (Tensor[batch_size, 3, input_ch, bool])
            dGgJ (Tensor[batch_size, 3, input_ch, bool])
            mask (Tensor[batch_size, input_ch, bool])

        Returns:
            Tuple[
                Tensor[batch_size, input_ch, float]
                Tensor[batch_size, 3, input_ch, float]
                None
            ]

        """
        J, dydx, dGdJ, mask = ctx.saved_tensors  # type: ignore
        d2x = (1 - dydx) * dydx
        d2x[mask] = 0.0
        dGdx = d2x * torch.sum(J * dLdG, 1)
        dLdx = dLdy * dydx + dGdx
        dLdJ = dLdG * dGdJ
        return dLdx, dLdJ, None
