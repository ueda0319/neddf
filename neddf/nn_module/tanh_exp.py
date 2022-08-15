import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


class tanhExp(torch.autograd.Function):
    """tanhExp

    This class calculate tanhExp, proposed as activation function.
    (https://arxiv.org/abs/2003.09855)
    This class inheriting torch.autograd.Function.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:  # type: ignore
        """forward

        Forward calculation for tanhExp.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            x (Tensor[batch_size, input_ch, float]): input features

        Returns:
            Tensor[batch_size, input_ch, float]

        """
        ex = torch.exp(x)
        tx = torch.tanh(ex)
        y = x * tx
        y[x > 20.0] = x[x > 20.0]
        ctx.save_for_backward(x, ex, tx)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y: Tensor) -> Tensor:  # type: ignore
        """backward

        Backward calculation for LinearGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            y (Tensor[batch_size, input_ch, float]): gradient of output features

        Contexts:
            x (Tensor[batch_size, input_ch, bool]): input
            ex (Tensor[batch_size, input_ch, bool]): exp(x)
            tx (Tensor[batch_size, input_ch, bool]): tanh(exp(x))

        Returns:
            Tensor[batch_size, input_ch, float]

        """
        x, ex, tx = ctx.saved_tensors  # type: ignore
        d: Tensor = tx - x * ex * (tx ** 2 - 1)
        d[x > 20.0] = 1.0
        return d * y
