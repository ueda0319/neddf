import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


class tanhExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:  # type: ignore
        ex = torch.exp(x)
        tx = torch.tanh(ex)
        y = x * tx
        y[x > 20.0] = x[x > 20.0]
        ctx.save_for_backward(x, ex, tx)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, y: Tensor) -> Tensor:  # type: ignore
        x, ex, tx = ctx.saved_tensors  # type: ignore
        d: Tensor = tx - x * ex * (tx**2 - 1)
        d[x > 20.0] = 1.0
        return d * y
