from typing import Optional, Tuple

import torch
from torch import Tensor


class TanhExpGradFunction(torch.autograd.Function):
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
        ctx.save_for_backward(d2x, J, dydx, dGdJ, mask)  # type: ignore
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
            mask (Tensor[batch_size, input_ch, bool])
        """
        d2x, J, dydx, dGdJ, mask = ctx.saved_tensors  # type: ignore
        dGdx = d2x * torch.sum(J * dLdG, 1)
        dLdx = dLdy * dydx + dGdx
        dLdJ = dLdG * dGdJ

        return dLdx, dLdJ, None
