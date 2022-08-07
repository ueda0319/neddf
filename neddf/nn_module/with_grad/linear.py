from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class LinearGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: torch.autograd.function._ContextMethodMixin,
        x: Tensor,
        J: Tensor,
        weight_t: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """forward

        Forward calculation for LinearGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            x (Tensor[batch_size, input_ch, float]): input features
            J (Tensor[batch_size, 3, input_ch, float]): gradients of input features
            weight_t (Tensor[input_ch, output_ch, float]): transposed weights
            bias (Tensor[output_ch, float]): bias
        """
        y = x.mm(weight_t)
        if bias is not None:
            y += bias.unsqueeze(0).expand_as(y)
        G = J.matmul(weight_t)

        ctx.save_for_backward(x, J, weight_t, bias)  # type: ignore
        return y, G

    @staticmethod
    def backward(  # type: ignore
        ctx: torch.autograd.function._ContextMethodMixin,
        dLdy: Tensor,
        dLdG: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """backward

        Backward calculation for LinearGradFunction.

        Args:
            ctx (_ContextMethodMixin): ctx for keep values used in backward
            dLdy (Tensor[batch_size, output_ch, float]): output features
            dLdG (Tensor[batch_size, 3, output_ch, float]): gradients of output features

        Contexts:
            x (Tensor[batch_size, input_ch, float]): input features
            J (Tensor[batch_size, 3, input_ch, float]): gradients of input features
            weight_t (Tensor[input_ch, output_ch, float]): weights
            bias (Tensor[output_ch, float]): bias
        """
        x, J, weight_t, bias = ctx.saved_tensors  # type: ignore
        input_ch, output_ch = weight_t.shape
        grad_x = grad_J = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:  # type: ignore
            grad_x = dLdy.mm(weight_t.t())
        if ctx.needs_input_grad[1]:  # type: ignore
            grad_J = dLdG.matmul(weight_t.t())
        if ctx.needs_input_grad[2]:  # type: ignore
            # grad_weight = x.t().mm(dLdy) + torch.sum(J.transpose(1, 2).matmul(dLdG), 0)
            grad_weight = x.t().mm(dLdy) + J.view(-1, input_ch).t().matmul(dLdG.view(-1,output_ch))
        if bias is not None and ctx.needs_input_grad[3]:  # type: ignore
            grad_bias = dLdy.sum(0)

        return grad_x, grad_J, grad_weight, grad_bias


class LinearGradLayer(nn.Module):
    def __init__(self, input_ch: int = 128, output_ch: int = 128):
        super(LinearGradLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_ch, output_ch))
        self.bias = nn.Parameter(torch.randn(output_ch))
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.bias, 0.0)
        self.input_ch = input_ch
        self.output_ch = output_ch

    def forward(self, x: Tensor, J: Tensor) -> Tuple[Tensor, Tensor]:
        """forward

        Forward calculation for LinearGradLayer.

        Args:
            x (Tensor[batch_size, input_ch, float]): input features
            J (Tensor[batch_size, 3, input_ch, float]): gradients of input features

        Returns:
            Tuple[
                Tensor[batch_size, input_ch, float]
                Tensor[batch_size, 3, input_ch, float]
            ]
        """
        return LinearGradFunction.apply(x, J, self.weight, self.bias)  # type: ignore

    def withoutGrad(self, x: Tensor) -> Tensor:
        y = x.mm(self.weight)
        y += self.bias.unsqueeze(0).expand_as(y)
        return y
