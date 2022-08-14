from typing import Tuple

import torch
from torch import Tensor


class LeakyReLUGradFunction(torch.autograd.Function):
    """LeakyReLUGradFunction

    This class inheriting torch.autograd.Function.
    This function calculate ReLU with first order gradient as forward propagation.
    """

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

        Returns:
            Tuple[
                Tensor[batch_size, input_ch, float]
                Tensor[batch_size, 3, input_ch, float]
            ]

        """
        scale = torch.ones_like(x)
        scale[(x<0)] = 0.01
        y = x * scale
        G = J * scale.unsqueeze(1)
        ctx.save_for_backward(scale)  # type: ignore
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
            scale (Tensor[batch_size, input_ch, bool])

        Returns:
            Tuple[
                Tensor[batch_size, input_ch, float]
                Tensor[batch_size, 3, input_ch, float]
            ]

        """
        scale = ctx.saved_tensors[0]  # type: ignore
        dLdx = dLdy * scale
        dLdJ = dLdG * scale.unsqueeze(1)

        return dLdx, dLdJ
