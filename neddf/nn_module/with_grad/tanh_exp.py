from typing import Optional, Tuple

import torch
from torch import Tensor


class TanhExpGradFunction(torch.autograd.Function):
    """TanhExpGradFunction

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

        Forward calculation for TanhExpGradFunction.

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
        mask = x > threshold
        ex = torch.exp(x)
        tx = torch.tanh(ex)
        y = x * tx
        y[mask] = x[mask]

        dydx = tx - x * ex * (tx ** 2 - 1)
        dydx[mask] = 1.0

        dGdJ = dydx.unsqueeze(1).expand_as(J)
        G = dGdJ * J

        d2x = ex * (-x + 2 * ex * x * tx - 2) * (tx ** 2 - 1)
        d2x[mask] = 0.0
        dGdx = J * d2x.unsqueeze(1)
        ctx.save_for_backward(dydx, dGdx, dGdJ)  # type: ignore
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
            dLdy (Tensor[batch_size, input_ch, float]): output features
            dLdG (Tensor[batch_size, 3, input_ch, float]): gradients of output features

        Contexts:
            dydx (Tensor[batch_size, input_ch, bool])
            dGdx (Tensor[batch_size, 3, input_ch, bool])
            dGdJ (Tensor[batch_size, 3, input_ch, bool])

        Returns:
            Tuple[
                Tensor[batch_size, input_ch, float]
                Tensor[batch_size, 3, input_ch, float]
                None
            ]

        """
        dydx, dGdx, dGdJ = ctx.saved_tensors  # type: ignore
        dLdx = dLdy * dydx + torch.sum(dLdG * dGdx, 1)
        dLdJ = dLdG * dGdJ

        return dLdx, dLdJ, None
