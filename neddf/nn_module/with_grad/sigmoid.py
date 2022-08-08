from typing import Optional, Tuple

import torch
from torch import Tensor


class SigmoidGradFunction(torch.autograd.Function):
    """LinearGradFunction

    This class inheriting torch.autograd.Function.
    This function calculate Sigmoid with first order gradient as forward propagation.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: torch.autograd.function._ContextMethodMixin,
        x: Tensor,
        J: Tensor,
        s: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """forward

        Forward calculation for SigmoidGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            x (Tensor[batch_size, input_ch, float]): input features
            J (Tensor[batch_size, 3, input_ch, float]): gradients of input features
            s (float): efficient of softplus

        Returns:
            Tuple[
                Tensor[batch_size, input_ch, float]
                Tensor[batch_size, 3, input_ch, float]
            ]

        """
        bs, input_ch = x.shape
        tx = (1.0 + torch.tanh(s * x * 0.5)) * 0.5
        y = tx
        dydx = s * tx * (1 - tx)
        dGdJ = dydx.unsqueeze(2).expand_as(J)
        G = dGdJ * J

        ctx.save_for_backward(J, tx, dydx, dGdJ)
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
            J (Tensor[batch_size, 3, input_ch, bool])
            tx (Tensor[batch_size, input_ch, bool])
            dydx (Tensor[batch_size, input_ch, bool])
            dGdJ (Tensor[batch_size, 3, input_ch, bool])

        Returns:
            Tuple[
                Tensor[batch_size, input_ch, float]
                Tensor[batch_size, 3, input_ch, float]
                None
            ]

        """
        J, tx, dydx, dGdJ = ctx.saved_tensors
        d2x = dydx * (1.0 - 2 * tx)
        dGdx = d2x * torch.sum(J * dLdG, 1)
        dLdx = dLdy * dydx + dGdx
        dLdJ = dLdG * dGdJ

        return dLdx, dLdJ, None
